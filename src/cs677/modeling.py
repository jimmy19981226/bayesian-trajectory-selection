"""Hugging Face integration. Heavy imports stay optional for the offline demo."""

from contextlib import contextmanager
import numpy as np
from .uncertainty import top_m_mean


def student_prompt(problem):
    return 'Solve the problem step by step. Put the final answer in \\boxed{}.\n\nProblem: ' + problem + '\n\nSolution:\n'


def encode_example(tokenizer, problem, response, max_length=4096):
    """Explicit token boundary shared by scoring and SFT; prompt labels are masked."""
    prompt = tokenizer.encode(student_prompt(problem), add_special_tokens=False)
    completion = tokenizer.encode(response, add_special_tokens=False)
    if not prompt or not completion:
        raise ValueError('Prompt and response must have tokens')
    if len(prompt) >= max_length:
        raise ValueError('Problem leaves no room for response tokens')
    if tokenizer.eos_token_id is not None:
        completion.append(tokenizer.eos_token_id)
    original_length = len(prompt) + len(completion)
    completion = completion[:max_length - len(prompt)]
    return {'input_ids': prompt + completion,
            'attention_mask': [1] * (len(prompt) + len(completion)),
            'labels': [-100] * len(prompt) + completion,
            'prompt_length': len(prompt), 'truncated': original_length > max_length}


def load_model(name, *, revision=None, dropout=None, quantize=False, device='auto'):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if quantize and device != 'cuda':
        raise ValueError('The reconstructed NF4 path requires CUDA; omit --quantize on Mac/CPU')
    config = AutoConfig.from_pretrained(name, revision=revision)
    if dropout is not None:
        if config.model_type != 'qwen2':
            raise ValueError('MC scorer is validated for Qwen2/Qwen2.5 models only')
        config.attention_dropout = dropout
    dtype = torch.float32 if device == 'cpu' else (torch.bfloat16 if device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16)
    options = {'config': config, 'revision': revision, 'dtype': dtype, 'attn_implementation': 'eager'}
    if quantize:
        options.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                       bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4',
                       bnb_4bit_compute_dtype=dtype), device_map='auto')
    model = AutoModelForCausalLM.from_pretrained(name, **options)
    if not quantize:
        model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError('Tokenizer requires a padding or EOS token')
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer


@contextmanager
def dropout_scoring(model):
    import torch
    states = [(module, module.training) for module in model.modules()]
    try:
        # Qwen uses functional attention dropout, so enabling nn.Dropout alone is insufficient.
        model.train()
        with torch.no_grad():
            yield
    finally:
        for module, training in states:
            module.training = training


def verify_dropout(model, input_ids):
    import torch
    with dropout_scoring(model):
        a = model.base_model(input_ids=input_ids, use_cache=False).last_hidden_state
        b = model.base_model(input_ids=input_ids, use_cache=False).last_hidden_state
    difference = float((a.float() - b.float()).abs().max())
    if not np.isfinite(difference) or difference == 0:
        raise RuntimeError('MC dropout is inactive or nonfinite; refusing to score a deterministic model')
    return {'max_hidden_state_difference': difference, 'dropout_verified': True}


def score_trajectory(model, example, *, passes=10, top_m=50, token_chunk_size=32):
    """Score completion tokens only, using correctly shifted causal predictions.

    Store K hidden-state sequences on CPU, then project vocabulary logits in chunks.
    This avoids retaining a [K, sequence, vocabulary] probability tensor.
    """
    import torch
    if passes < 2 or token_chunk_size < 1:
        raise ValueError('At least two dropout passes and a positive chunk size are required')
    if model.config.model_type != 'qwen2':
        raise ValueError('Scorer supports Qwen2/Qwen2.5 causal models')
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([example['input_ids']], device=device)
    start = example['prompt_length'] - 1
    stop = ids.shape[1] - 1
    if start < 0 or stop <= start:
        raise ValueError('No completion tokens to score')
    hidden = []
    with dropout_scoring(model):
        for _ in range(passes):
            h = model.base_model(input_ids=ids, use_cache=False).last_hidden_state
            hidden.append(h[0, start:stop].detach().cpu())
        head = model.get_output_embeddings()
        head_device = head.weight.device
        total_parts, expected_parts = [], []
        for offset in range(0, stop - start, token_chunk_size):
            sum_probs = None
            sum_entropy = None
            for h in hidden:
                logits = head(h[offset:offset + token_chunk_size].to(head_device)).float()
                logp = torch.log_softmax(logits, dim=-1)
                p = logp.exp()
                e = -(p * logp).sum(-1)
                sum_probs = p.clone() if sum_probs is None else sum_probs + p
                sum_entropy = e.clone() if sum_entropy is None else sum_entropy + e
            mean = sum_probs / passes
            total = -(mean * mean.clamp_min(torch.finfo(mean.dtype).tiny).log()).sum(-1)
            total_parts.append(total.cpu().numpy())
            expected_parts.append((sum_entropy / passes).cpu().numpy())
    total = np.concatenate(total_parts)
    expected = np.concatenate(expected_parts)
    mi = np.maximum(total - expected, 0)
    return {'total_entropy': top_m_mean(total, top_m),
            'mutual_information': top_m_mean(mi, top_m),
            'expected_entropy_mean': float(expected.mean()),
            'scored_tokens': len(total), 'truncated': example['truncated']}


class CompletionCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        import torch
        width = max(len(e['input_ids']) for e in examples)
        batch = {}
        for key, fill in [('input_ids', self.pad_token_id), ('attention_mask', 0), ('labels', -100)]:
            batch[key] = torch.tensor([e[key] + [fill] * (width - len(e[key])) for e in examples])
        return batch
