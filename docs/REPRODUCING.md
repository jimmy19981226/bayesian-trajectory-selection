# Reproduction and validation guide

## Two levels of reproducibility

The reported result tables can be reproduced from the archived paper and the structured transcriptions. The experiments can be **rerun using a reconstructed implementation**, but exact numerical reproduction of the original training run is not guaranteed because original trajectories, weights, settings, and logs are missing.

The configuration JSON documents evidence and defaults; it is not a live CLI configuration loader. CLI flags are the executable source of run settings. Both baseline and epistemic commands must use the same training and evaluation flags.

## Installation

The README gives the recommended virtual environment setup. Only NumPy is needed for the demo, numerical functions, and selection. Install the `ml` extra for model/dataset operations and `quantization` for the CUDA bitsandbytes path. The base package supports Python 3.11+.

```bash
python -m pip install -e '.[ml,dev]'
HF_HUB_OFFLINE=1 pytest -q
```

Tiny-model tests create random Qwen2 models locally. No downloaded pretrained checkpoint is required. If the machine has unrelated globally installed pytest plugins, disable plugin auto-discovery:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HF_HUB_OFFLINE=1 python -m pytest -q
```

## Data contracts

Files are UTF-8 JSONL, one object per line. Never mix unrelated runs in one file.

Prepared problem:

```json
{"problem_id":"0","problem":"What is 1 + 1?","subject":"Algebra","level":"Level 1"}
```

Teacher candidate (additional fields may be retained):

```json
{"problem_id":"0","candidate_id":"0","problem":"What is 1 + 1?","response":"Adding gives \\boxed{2}."}
```

Scoring adds `total_entropy`, `mutual_information`, `expected_entropy_mean`, `scored_tokens`, and `truncated`. The first two are separately aggregated top-m trajectory scores; `expected_entropy_mean` is a diagnostic mean across all completion tokens and must not be subtracted from the trajectory-level entropy score to recover MI.

`select` requires exactly five unique candidate IDs per problem by default, finite scores, and consistent question text. Its output is `baseline.jsonl`, `epistemic.jsonl`, a selection-disagreement report, and a manifest. Use `--candidates` if deliberately running a different candidate-count experiment.

No training data or model weights are committed. Data belongs under ignored `data/`; run artifacts under ignored `runs/`.

## Prompting and causal alignment

The reconstructed student prompt is in `student_prompt()`. The base Qwen model receives plain text. The instruction-tuned teacher uses its chat template around the same problem/instruction. Teacher generation uses sampling; evaluation uses greedy decoding. These are explicit reconstruction choices, not recovered settings.

Scoring and SFT tokenize prompt and completion separately and concatenate IDs. This establishes an explicit boundary, avoiding ambiguous token merges across the prompt/response boundary. Scoring excludes prompt tokens and uses hidden states one position before each completion token. SFT masks prompt and padding labels with -100. EOS remains a training target even when its ID equals the padding ID.

Examples longer than `--max-length` are right-truncated. Scored rows contain a `truncated` flag, and training manifests contain the count. A problem that leaves no completion room raises an error. Inspect truncation rates; they can affect reasoning coverage and final-answer supervision.

## MC dropout implementation and memory

Qwen implements attention dropout functionally, so simply enabling `nn.Dropout` modules is insufficient. The scorer puts the model in training mode inside `torch.no_grad()` and restores previous module modes afterward. It disables the KV cache for repeated full teacher-forced passes. A two-pass hidden-state probe must show nonzero finite change before scoring proceeds.

The scorer performs K transformer passes, retaining completion hidden states on CPU, then applies the output projection in token chunks. This avoids retaining K full vocabulary probability tensors. With K=10, about 4096 tokens, hidden width 2048, and two-byte hidden states, the retained hidden states alone are about 160 MiB. Attention activations, model weights, and vocabulary projection buffers require additional memory. `--token-chunk-size` controls the last component; the default is 32.

The implementation uses eager attention for predictable dropout behavior. Its quadratic attention memory can be substantial at 4096 tokens. The 72B teacher is not a laptop-sized workload. Full 3B scoring/training and 60,000 candidate generations should be planned for suitable accelerator resources.

The recovered notebook quantized scoring with NF4. Add `--quantize` to match that aspect. This implementation limits the quantized path to CUDA; unquantized CPU/MPS are available for smaller workloads. Training quantization is optional and is not confirmed by the original paper. If enabled, `prepare_model_for_kbit_training` prepares the frozen base before LoRA attachment.

## Revisions, manifests, and outputs

Pass `--revision COMMIT` to generation, scoring, training, and evaluation to pin the model. Preparation accepts `--revision` for the training dataset and `--math-revision` for the held-out overlap check. Evaluation accepts `--dataset-revision`. Omitted revisions use upstream defaults and can change over time.

Each successful command writes a manifest containing explicit arguments and installed package versions. Relevant steps also record input SHA-256 hashes, dataset fingerprints, and resolved model commit hashes. These records improve future reproducibility; they cannot infer lost historical versions. Different accelerator/library combinations may still differ even with identical seeds.

Generation, scoring, and evaluation support `--start` and `--limit` to process slices. Each invocation resets the RNG from its seed, so independently generated shards need a deliberate seed policy. `select` must receive complete candidate groups. Output paths must be new. JSONL data is streamed; interrupted files may contain incomplete runs and must be inspected before reuse. Automatic resume and distributed training launch configuration are outside this reconstruction.

## Evaluation audit

Each evaluation folder contains the full generated response, extracted answer, reference, strict correctness flag, and subject/difficulty metadata per problem. The reported metric deliberately does not apply equivalence normalization. Run a separate labeled analysis if adding a symbolic grader; do not overwrite strict results.

Example strict mismatches: `0.5` versus `\frac{1}{2}`, `2.0` versus `2`, and ` 2 ` versus `2`. The first boxed result is used even if a later box corrects it, matching the reported procedure. Unclosed first boxes fail extraction.

## Historical result figure

```bash
python scripts/plot_reported_results.py
```

This reads `results/reported/results.json`, checks aggregate counts, and writes the README chart. It uses a 0–100% axis and labels the values as reported, so the tiny differences are not visually exaggerated.

## What was validated

Numerical tests check entropy with zero probabilities, analytic disagreement cases, top-m behavior, chi-squared KKT conditions, data selection and ties, overlap rejection, nested boxed-answer parsing, historical count consistency, and original-file checksums.

Model tests compare chunked uncertainty calculations with full causal logits on an actual tiny Qwen2 model, verify dropout failure when disabled, check prompt/padding/EOS loss masks, and train/save/reload a tiny LoRA adapter. The end-to-end offline integration test exercises the reconstructed CLI against a locally created Qwen model and synthetic datasets. These checks establish implementation behavior on tiny fixtures; they do not validate full-size GPU/NF4 execution or reproduce MATH-500/GSM8K accuracy.
