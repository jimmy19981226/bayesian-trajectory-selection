"""Prepare, generate, score, select, fine-tune, and evaluate."""

from pathlib import Path
import itertools
from .io import read_jsonl, write_jsonl, write_json, manifest
from .modeling import load_model, student_prompt, encode_example, score_trajectory, verify_dropout, CompletionCollator
from .selection import select_winners
from .evaluation import first_boxed_answer, reference_answer, summarize_predictions


def normalized_problem(text):
    return ' '.join(text.split())


def assert_no_overlap(train, test):
    heldout = {normalized_problem(r['problem']) for r in test}
    overlap = [r['problem_id'] for r in train if normalized_problem(r['problem']) in heldout]
    if overlap:
        raise ValueError(f'Training/benchmark overlap found: {overlap[:10]}')


def subset(rows, args):
    return itertools.islice(rows, args.start, None if args.limit is None else args.start + args.limit)


def check_output(path):
    if Path(path).exists():
        raise FileExistsError(f'{path} exists; choose a new output to preserve previous runs')


def training_precision(device):
    """Choose mixed precision for the selected device, not unrelated accelerators."""
    import torch
    cuda = device == 'cuda'
    bf16 = cuda and torch.cuda.is_bf16_supported()
    return {'bf16': bf16, 'fp16': cuda and not bf16, 'use_cpu': device == 'cpu'}


def prepare(args):
    from datasets import load_dataset
    check_output(args.output)
    train = load_dataset('rasbt/math_full_minus_math500', split='train', revision=args.revision)
    test = load_dataset('HuggingFaceH4/MATH-500', split='test', revision=args.math_revision)
    rows = []
    for i, row in enumerate(train):
        rows.append({'problem_id': str(row.get('unique_id', i)), 'problem': row['problem'],
                     'subject': row.get('type'), 'level': row.get('level')})
    if len(rows) != 12000:
        raise ValueError(f'Expected the original 12,000-row pool, got {len(rows)}; inspect dataset revision')
    if len({r['problem_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate training problem IDs')
    assert_no_overlap(rows, test)
    write_jsonl(args.output, subset(rows, args))
    manifest(args.output + '.manifest.json', args, train_fingerprint=train._fingerprint,
             test_fingerprint=test._fingerprint, full_training_rows=len(rows), overlap_count=0)


def generate(args):
    import torch
    from transformers import set_seed
    check_output(args.output)
    set_seed(args.seed)
    model, tokenizer = load_model(args.model, revision=args.revision, quantize=args.quantize, device=args.device)
    def candidates():
        for row in subset(read_jsonl(args.input), args):
            messages = [{'role': 'user', 'content': student_prompt(row['problem'])}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(model.device)
            for candidate in range(args.candidates):
                with torch.no_grad():
                    output = model.generate(**inputs, do_sample=True, temperature=args.temperature,
                              top_p=args.top_p, max_new_tokens=args.max_new_tokens,
                              pad_token_id=tokenizer.pad_token_id)
                response = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                if not response.strip():
                    raise ValueError(f'Empty teacher response for {row["problem_id"]}')
                yield {**row, 'candidate_id': str(candidate), 'response': response}
    write_jsonl(args.output, candidates())
    manifest(args.output + '.manifest.json', args, model_commit=getattr(model.config, '_commit_hash', None))


def score(args):
    from transformers import set_seed
    check_output(args.output)
    set_seed(args.seed)
    model, tokenizer = load_model(args.model, revision=args.revision, dropout=args.dropout,
                                   quantize=args.quantize, device=args.device)
    probe = tokenizer('The solution to 2x + 5 = 15 begins with', return_tensors='pt').input_ids.to(model.device)
    verification = verify_dropout(model, probe)
    def scores():
        for row in subset(read_jsonl(args.input), args):
            example = encode_example(tokenizer, row['problem'], row['response'], args.max_length)
            yield {**row, **score_trajectory(model, example, passes=args.passes,
                                            top_m=args.top_m, token_chunk_size=args.token_chunk_size)}
    write_jsonl(args.output, scores())
    manifest(args.output + '.manifest.json', args, **verification,
             model_commit=getattr(model.config, '_commit_hash', None))


def select(args):
    check_output(args.output)
    baseline, epistemic, report = select_winners(read_jsonl(args.input), args.candidates)
    folder = Path(args.output)
    write_jsonl(folder / 'baseline.jsonl', baseline)
    write_jsonl(folder / 'epistemic.jsonl', epistemic)
    write_json(folder / 'selection.json', report)
    manifest(folder / 'manifest.json', args)
    print(report)


def train(args):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments, set_seed
    check_output(args.output)
    set_seed(args.seed)
    model, tokenizer = load_model(args.model, revision=args.revision, quantize=args.quantize, device=args.device)
    if args.quantize:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha,
        lora_dropout=args.lora_dropout, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj'], task_type='CAUSAL_LM'))
    examples = [encode_example(tokenizer, r['problem'], r['response'], args.max_length)
                for r in read_jsonl(args.train_file)]
    if not examples:
        raise ValueError('Empty training set')
    truncated = sum(e['truncated'] for e in examples)
    # Keep only Trainer model inputs; right padding does not mask real EOS labels.
    data = [{k: e[k] for k in ('input_ids', 'attention_mask', 'labels')} for e in examples]
    settings = TrainingArguments(output_dir=args.output, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, seed=args.seed, data_seed=args.seed,
        lr_scheduler_type='linear', warmup_steps=0, weight_decay=0,
        logging_steps=10, save_strategy='epoch', save_total_limit=2, report_to='none',
        **training_precision(model.device.type), gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False})
    trainer = Trainer(model=model, args=settings, train_dataset=data,
                      data_collator=CompletionCollator(tokenizer.pad_token_id))
    outcome = trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    manifest(Path(args.output) / 'manifest.json', args, training_rows=len(data),
             truncated_rows=truncated, train_metrics=outcome.metrics,
             model_commit=getattr(model.config, '_commit_hash', None))


def evaluate(args):
    import torch
    from datasets import load_dataset
    from transformers import set_seed
    check_output(args.output)
    set_seed(args.seed)
    model, tokenizer = load_model(args.model, revision=args.revision, quantize=args.quantize, device=args.device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    dataset = (load_dataset('HuggingFaceH4/MATH-500', split='test', revision=args.dataset_revision)
               if args.benchmark == 'math500' else
               load_dataset('openai/gsm8k', 'main', split='test', revision=args.dataset_revision))
    folder = Path(args.output)
    def predictions():
        for i, row in enumerate(subset(dataset, args), start=args.start):
            problem = row['problem'] if args.benchmark == 'math500' else row['question']
            inputs = tokenizer(student_prompt(problem), return_tensors='pt', add_special_tokens=False).to(model.device)
            with torch.no_grad():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                                         pad_token_id=tokenizer.pad_token_id)
            response = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            predicted = first_boxed_answer(response)
            reference = reference_answer(row, args.benchmark)
            yield {'problem_id': str(row.get('unique_id', i)), 'problem': problem,
                   'response': response, 'prediction': predicted, 'reference': reference,
                   'correct': predicted is not None and predicted == reference,
                   'subject': row.get('subject', row.get('type')), 'level': row.get('level')}
    write_jsonl(folder / 'predictions.jsonl', predictions())
    result = summarize_predictions(read_jsonl(folder / 'predictions.jsonl'))
    write_json(folder / 'metrics.json', result)
    manifest(folder / 'manifest.json', args, dataset_fingerprint=dataset._fingerprint,
             model_commit=getattr(model.config, '_commit_hash', None))
    print(result['overall'])
