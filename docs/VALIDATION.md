# Restoration validation

Validated locally on 2026-10-12 using Python 3.13 on macOS, with CPU-only tiny-model tests.

| Check | Outcome |
| --- | --- |
| Core mathematics, selection, parser, overlap checks, original checksums | Passed |
| MC dropout and chunked/full-logit equivalence on tiny Qwen2 | Passed |
| Completion labels, prompt/padding masks, EOS handling | Passed |
| Tiny LoRA training, adapter save and reload | Passed |
| CLI prepare → generate → score → select → two adapter training runs → both benchmarks | Passed using synthetic data and a local random model |
| Complete local test suite | **24 passed** |
| Editable package installation and installed `cs677 demo` command | Passed |
| Historical result count consistency | Passed |
| Local documentation links | No broken targets |
| README chart | Rendered and visually inspected |

Command used for tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q
```

Plugin auto-discovery was disabled because an unrelated global course-grading pytest plugin failed during test collection. No project test was disabled.

Observed versions: NumPy 2.3.4, PyTorch 2.11.0, Transformers 5.3.0, Datasets 4.8.3, PEFT 0.21.2, Accelerate 1.13.0, pytest 7.4.4, Matplotlib 3.11.2. These are the **restoration validation environment**, not the lost experiment's dependencies. The project's dependency ranges allow compatible updates and are not an exact lockfile.

Not validated here: full-size 72B generation, complete 3B training, CUDA NF4 quantization, multi-device execution, and reproduction of historical benchmark scores. Those require the relevant hardware, data generation, and new experiment runs. The random-model integration test assesses software behavior, not mathematical reasoning quality.
