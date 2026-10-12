"""End-to-end CLI smoke test on a local random model and synthetic datasets."""
import json
import pytest
torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')
datasets = pytest.importorskip('datasets')
pytest.importorskip('peft')
from cs677.cli import main
from cs677.io import read_jsonl


def test_all_pipeline_stages_offline(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    vocabulary = {word: i for i, word in enumerate([
        '[PAD]', '[EOS]', '[UNK]', 'Solve', 'the', 'problem', 'step', 'by', '.',
        'Put', 'final', 'answer', 'in', 'boxed', 'Problem', 'Solution', ':',
        'What', 'is', '1', '+', '2', '?', '3', '4', '5', '{', '}', '\\', 'Add', 'gives'])}
    backend = Tokenizer(WordLevel(vocabulary, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend,
        pad_token='[PAD]', eos_token='[EOS]', unk_token='[UNK]')
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    config = transformers.Qwen2Config(vocab_size=len(vocabulary), hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=128, pad_token_id=0, eos_token_id=1)
    torch.manual_seed(10)
    model = transformers.Qwen2ForCausalLM(config)
    model.generation_config.eos_token_id = None
    model_path = tmp_path / 'local-model'
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    train_data = datasets.Dataset.from_list([
        {'problem': f'What is {i} + 1?', 'unique_id': i, 'type': 'Algebra', 'level': 'Level 1'}
        for i in range(12000)])
    math_data = datasets.Dataset.from_list([
        {'problem': 'What is 1 + 2 + 3?', 'answer': '6', 'subject': 'Algebra', 'level': 1},
        {'problem': 'What is 4 + 5 + 6?', 'answer': '15', 'subject': 'Algebra', 'level': 2}])
    gsm_data = datasets.Dataset.from_list([
        {'question': 'What is 1 + 1?', 'answer': 'Add.\n#### 2'},
        {'question': 'What is 2 + 2?', 'answer': 'Add.\n#### 4'}])
    def local_dataset(name, *args, **kwargs):
        if name == 'rasbt/math_full_minus_math500':
            return train_data
        if name == 'HuggingFaceH4/MATH-500':
            return math_data
        if name == 'openai/gsm8k':
            return gsm_data
        raise AssertionError(name)
    monkeypatch.setattr(datasets, 'load_dataset', local_dataset)
    problems, candidates, scores = [str(tmp_path / name) for name in ['problems.jsonl', 'candidates.jsonl', 'scores.jsonl']]
    common = ['--model', str(model_path), '--device', 'cpu']
    main(['prepare', '--output', problems, '--limit', '2'])
    main(['generate', '--input', problems, '--output', candidates, '--candidates', '2', '--max-new-tokens', '8', *common])
    assert len(list(read_jsonl(candidates))) == 4
    main(['score', '--input', candidates, '--output', scores, '--passes', '2', '--max-length', '128', *common])
    selected = tmp_path / 'selected'
    main(['select', '--input', scores, '--output', str(selected), '--candidates', '2'])
    for method in ['baseline', 'epistemic']:
        adapter = tmp_path / method
        main(['train', '--train-file', str(selected / f'{method}.jsonl'), '--output', str(adapter),
              '--epochs', '1', '--batch-size', '1', '--gradient-accumulation', '1', '--rank', '2',
              '--alpha', '4', '--max-length', '128', *common])
        assert (adapter / 'adapter_model.safetensors').is_file()
        assert json.loads((adapter / 'manifest.json').read_text())['training_rows'] == 2
        for benchmark in ['math500', 'gsm8k']:
            result = tmp_path / f'{method}-{benchmark}'
            main(['evaluate', '--adapter', str(adapter), '--benchmark', benchmark,
                  '--output', str(result), '--max-new-tokens', '8', *common])
            assert json.loads((result / 'metrics.json').read_text())['overall']['total'] == 2
            rows = list(read_jsonl(result / 'predictions.jsonl'))
            assert all(r['correct'] == (r['prediction'] is not None and r['prediction'] == r['reference']) for r in rows)
