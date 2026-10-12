"""Actual tiny Qwen models, initialized locally; no checkpoints or network needed."""

import numpy as np
import pytest
torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')
from cs677.modeling import score_trajectory, verify_dropout, encode_example, CompletionCollator
from cs677.uncertainty import decompose, top_m_mean
from cs677.pipeline import training_precision


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_non_cuda_training_does_not_enable_cuda_precision(device, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)

    def unexpected_cuda_query():
        pytest.fail('Non-CUDA training queried CUDA precision support')

    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', unexpected_cuda_query)
    assert training_precision(device) == {
        'bf16': False, 'fp16': False, 'use_cpu': device == 'cpu'
    }


@pytest.mark.parametrize('bf16_supported', [True, False])
def test_cuda_training_chooses_one_precision(bf16_supported, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: bf16_supported)
    assert training_precision('cuda') == {
        'bf16': bf16_supported, 'fp16': not bf16_supported, 'use_cpu': False
    }


def tiny_model(dropout=.2):
    torch.manual_seed(123)
    config = transformers.Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=128, attention_dropout=dropout, pad_token_id=0,
        eos_token_id=1, bos_token_id=2)
    config._attn_implementation = 'eager'
    return transformers.Qwen2ForCausalLM(config).eval()


def test_chunked_scores_match_full_causal_logits_and_restore_mode():
    model = tiny_model()
    example = {'input_ids': [2, 3, 4, 5, 6], 'prompt_length': 3, 'truncated': False}
    ids = torch.tensor([example['input_ids']])
    verify_dropout(model, ids)
    assert not model.training
    torch.manual_seed(77)
    actual = score_trajectory(model, example, passes=3, top_m=1, token_chunk_size=1)
    torch.manual_seed(77)
    model.train()
    with torch.no_grad():
        probabilities = np.stack([model(ids, use_cache=False).logits[0, 2:4].float().softmax(-1).numpy() for _ in range(3)])
    expected = decompose(probabilities)
    assert actual['scored_tokens'] == 2
    for key in ['total_entropy', 'mutual_information']:
        np.testing.assert_allclose(actual[key], top_m_mean(expected[key], 1), atol=1e-6)
    assert all(p.grad is None for p in model.parameters())


def test_deterministic_dropout_fails_verification():
    with pytest.raises(RuntimeError, match='inactive'):
        verify_dropout(tiny_model(0), torch.tensor([[2, 3, 4, 5]]))


def test_collator_masks_prompt_and_padding_but_keeps_eos():
    class Tokenizer:
        eos_token_id = 1
        def encode(self, text, add_special_tokens=False):
            return [3, 4] if text.startswith('Solve') else [5]
    example = encode_example(Tokenizer(), 'question', 'response')
    assert example['labels'] == [-100, -100, 5, 1]
    short = {k: v[:-1] for k, v in example.items() if k in ('input_ids', 'attention_mask', 'labels')}
    batch = CompletionCollator(1)([example, short])
    assert batch['labels'].tolist() == [[-100, -100, 5, 1], [-100, -100, 5, -100]]
    assert encode_example(Tokenizer(), 'q', 'r', max_length=3)['truncated']


def test_lora_train_save_reload(tmp_path):
    peft = pytest.importorskip('peft')
    model = tiny_model(0)
    adapter = peft.get_peft_model(model, peft.LoraConfig(r=2, lora_alpha=4,
                    target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    examples = [{'input_ids': [2, 3, 4, 1], 'attention_mask': [1] * 4, 'labels': [-100, -100, 4, 1]}]
    before = {n: p.detach().clone() for n, p in adapter.named_parameters() if p.requires_grad}
    trainer = transformers.Trainer(model=adapter, train_dataset=examples,
        data_collator=CompletionCollator(0), args=transformers.TrainingArguments(
        output_dir=str(tmp_path / 'trainer'), use_cpu=True, max_steps=1,
        per_device_train_batch_size=1, learning_rate=.01, report_to='none', save_strategy='no'))
    trainer.train()
    assert any(not torch.equal(before[n], p) for n, p in adapter.named_parameters() if n in before)
    adapter.eval()
    ids = torch.tensor([[2, 3, 4]])
    with torch.no_grad():
        expected = adapter(ids).logits
    adapter.save_pretrained(tmp_path / 'adapter')
    restored = peft.PeftModel.from_pretrained(tiny_model(0), tmp_path / 'adapter').eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(ids).logits, expected)
