# Bayesian Epistemic Uncertainty for Reasoning Trajectory Selection

**James Chen · CS 677 Final Project**

Can a small language model learn mathematical reasoning more efficiently when its training trajectories are selected using epistemic uncertainty instead of total predictive entropy?

This project compares two ways of choosing teacher-generated solutions for LoRA fine-tuning of Qwen2.5-3B. A Qwen2.5-72B-Instruct teacher generates five solutions per problem. The student scores each solution using Monte Carlo dropout, and two selection rules produce separate training sets: one based on total entropy and one based on mutual information (MI).

> **Restoration status:** The original paper, slides, and partial notebook have been recovered and preserved. The Python pipeline is a new reconstruction of their methods. The original 60,000 trajectories, trained adapters, full source code, and per-question evaluation logs were not available in the recovered files. All benchmark numbers below are **reported historical results**, not newly reproduced measurements. See [recovery provenance](docs/RECOVERY.md).

[Original paper](sources/final-paper.pdf) · [Presentation](sources/final-presentation.pptx) · [Recovered notebook](sources/original-notebook.ipynb) · [Theory](docs/THEORY.md) · [Reproduction guide](docs/REPRODUCING.md) · [Validation](docs/VALIDATION.md)

## Main result

MI selection changed the chosen trajectory for **8,105 of 12,000 problems (67.54%)**, but the reported accuracy gains were small: one additional correct answer on MATH-500 and two on GSM8K.

| Benchmark | Total entropy baseline | Epistemic MI | Change |
| --- | ---: | ---: | ---: |
| MATH-500 | 46.20% (231/500) | 46.40% (232/500) | +0.20 percentage points |
| GSM8K | 79.30% (1046/1319) | 79.45% (1048/1319) | +0.15 percentage points |

Source: original paper, Table 1, page 3. These small differences do not establish statistically reliable superiority. Multiple seeds and paired prediction logs would be needed for a stronger comparison.

![Reported benchmark results](docs/assets/reported-accuracy.png)

## Method and theory

### 1. Teacher-generated reasoning trajectories

The recovered notebook loads the training split of [`rasbt/math_full_minus_math500`](https://huggingface.co/datasets/rasbt/math_full_minus_math500), a 12,000-problem pool that excludes MATH-500. The paper reports five Qwen2.5-72B-Instruct solutions per problem, yielding **60,000 candidates**. The restored preparation command also checks normalized problem text against MATH-500 before writing the training pool.

### 2. Monte Carlo dropout

Qwen2.5-3B scores a fixed candidate solution under teacher forcing using **K = 10** stochastic passes. Attention dropout remains active, while gradients are disabled and model weights remain unchanged. The notebook sets attention dropout to **0.1** and loads the student with **4-bit NF4 quantization**, double quantization, and bfloat16 compute.

For token position $t$, vocabulary token $v$, and dropout sample $k$:

```math
p_t^{(k)}(v)=p(v\mid x,y_{\lt t},\theta^{(k)}), \qquad
\bar p_t(v)=\frac{1}{K}\sum_{k=1}^{K}p_t^{(k)}(v).
```

### 3. Separate total and epistemic uncertainty

The predictive entropy of the mean distribution measures total uncertainty:

```math
H_{\mathrm{total}}(t)=-\sum_v \bar p_t(v)\log\bar p_t(v).
```

The mean entropy within individual passes is:

```math
H_{\mathrm{expected}}(t)=\frac{1}{K}\sum_{k=1}^{K}\left[-\sum_v p_t^{(k)}(v)\log p_t^{(k)}(v)\right].
```

Their difference is the mutual-information estimate:

```math
\mathrm{MI}(t)=H_{\mathrm{total}}(t)-H_{\mathrm{expected}}(t).
```

If each pass is uncertain in the same way, total entropy can be high while MI is near zero. If the passes confidently disagree, MI is high. The project hypothesizes that this disagreement identifies useful gaps in the student's knowledge.

This is a model-dependent proxy, not a direct measurement of missing mathematical knowledge. The Bayesian interpretation depends on the dropout approximation. Enabling dropout in a pretrained model does not, by itself, establish calibrated Bayesian uncertainty. [Gal & Ghahramani (2016)](https://proceedings.mlr.press/v48/gal16.html) provides the theoretical motivation.

### 4. Top-50 aggregation and selection

For either token score $u_t$, take the mean of the 50 largest scores (or all scored tokens when a trajectory has fewer than 50):

```math
S_u(y\mid x)=\frac{1}{m'}\sum_{t\in\mathrm{Top}_{m'}(u)}u_t,
\qquad m'=\min(50,T).
```

The final experiment selects exactly one trajectory per problem:

```math
y_H^*=\arg\max_{y\in\mathcal C(x)}S_{H_{\mathrm{total}}}(y\mid x),\qquad
y_{\mathrm{MI}}^*=\arg\max_{y\in\mathcal C(x)}S_{\mathrm{MI}}(y\mid x).
```

Each training set contains **12,000 examples**. The baseline in the final paper is entropy of the **MC-averaged predictive distribution**. The notebook's earlier plan mentions a point-estimate entropy comparison; that is not the same baseline.

### 5. LoRA distillation

Each selected dataset trains a separate LoRA adapter on the same student with the same reported hyperparameters:

| Setting | Value | Evidence |
| --- | --- | --- |
| Student | Qwen/Qwen2.5-3B | Paper and notebook |
| Teacher | Qwen/Qwen2.5-72B-Instruct | Paper and slides |
| MC passes / top tokens | 10 / 50 | Paper |
| Attention dropout for scoring | 0.1 | Notebook |
| LoRA rank / alpha | 16 / 32 | Paper |
| Epochs | 2 | Paper |
| Per-device batch size | 4 | Paper |
| Gradient accumulation | 4 | Paper |
| Learning rate | 2 × 10⁻⁴ | Paper |
| Maximum sequence length | 4096 | Paper |

The paper mentions LoRA dropout but does not give its probability or target modules. The reconstruction uses **0.05** and Qwen attention/MLP projection modules. Prompts, seed 42, decoding settings, completion-only loss, EOS handling, right truncation, optimizer schedule, and other unrecovered details are explicit reconstruction choices in [configs/paper.json](configs/paper.json).

### 6. Strict evaluation

Generate one response per benchmark problem, extract the **first** `\boxed{...}` answer, and compare it with the reference by exact string equality. The restored parser handles nested braces, such as `\boxed{\frac{1}{2}}`. Missing or malformed boxes are incorrect. It does not normalize mathematical expressions, numeric formatting, or whitespace inside the box. GSM8K references use the answer after `####`.

```math
\mathrm{ACC@1}=\frac{\#\text{correct single-response answers}}{\#\text{evaluated questions}}.
```

### Earlier proposal: chi-squared regularization

The notebook proposed normalizing top-MI utility and applying a chi-squared-regularized selection rule. It does **not** preserve that rule's equation, normalization, hyperparameters, or implementation. The final paper instead describes the per-problem argmax method above.

For completeness, the restored library includes an **optional illustrative implementation**, separate from the reported experiment:

```math
q^*=\arg\max_{q\in\Delta_n}\left[\sum_iq_i u_i-\frac{\lambda}{2}\sum_i\frac{(q_i-p_i)^2}{p_i}\right],
```

```math
q_i^*=p_i\left[1+\frac{u_i-\eta}{\lambda}\right]_+, \qquad \sum_iq_i^*=1.
```

Here $p_i>0$ is a prior and $\eta$ enforces normalization. [`chi_squared_weights`](src/cs677/selection.py) solves the threshold numerically. This supplies a usable version of the proposed idea, but it is **not recovered original code and has no reported benchmark results**. It is not applied by the paper reproduction commands. Derivation and assumptions are in [THEORY.md](docs/THEORY.md).

## Complete reported results

### MATH-500 by subject

| Subject | N | Total entropy baseline | Epistemic MI |
| --- | ---: | ---: | ---: |
| Algebra | 124 | 65.3% (81/124) | 64.5% (80/124) |
| Counting & Probability | 38 | 34.2% (13/38) | 36.8% (14/38) |
| Geometry | 41 | 36.6% (15/41) | 29.3% (12/41) |
| Intermediate Algebra | 97 | 25.8% (25/97) | 30.9% (30/97) |
| Number Theory | 62 | 71.0% (44/62) | 74.2% (46/62) |
| Prealgebra | 82 | 53.7% (44/82) | 47.6% (39/82) |
| Precalculus | 56 | 16.1% (9/56) | 19.6% (11/56) |

Source: paper, Table 2, pages 3–4. MI improved four subjects and declined in three; gains were not uniform.

### MATH-500 by difficulty

| Difficulty | N | Total entropy baseline | Epistemic MI |
| --- | ---: | ---: | ---: |
| Level 1 | 43 | 72.1% (31/43) | 74.4% (32/43) |
| Level 2 | 90 | 62.2% (56/90) | 65.6% (59/90) |
| Level 3 | 105 | 53.3% (56/105) | 53.3% (56/105) |
| Level 4 | 128 | 40.6% (52/128) | 43.0% (55/128) |
| Level 5 | 134 | 26.9% (36/134) | 22.4% (30/134) |

Source: paper, Table 3, page 4. MI improved Levels 1, 2, and 4, tied Level 3, and declined on Level 5.

The JSON and CSV transcriptions are in [`results/reported/`](results/reported/). Both MATH-500 breakdowns sum to the overall counts (231 baseline, 232 MI). Original per-example predictions were not recovered, so paired significance tests and confidence intervals on the paired difference cannot be reconstructed from these tables alone.

## Run locally

Python 3.11 or newer is required. The offline demo needs only NumPy; it downloads no model or dataset.

```bash
git clone https://github.com/jimmy19981226/bayesian-trajectory-selection.git
cd bayesian-trajectory-selection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cs677 demo --output runs/demo
pytest tests/test_core.py -q
```

The demo intentionally contrasts equally uncertain predictions with disagreeing predictions. Its outputs are labeled synthetic and are not benchmark measurements.

For the model pipeline:

```bash
python -m pip install -e '.[ml,dev]'
# Optional, for the recovered CUDA NF4 scoring path:
python -m pip install -e '.[quantization]'

cs677 prepare --output data/problems.jsonl
cs677 generate --input data/problems.jsonl --output data/candidates.jsonl
cs677 score --input data/candidates.jsonl --output data/scored.jsonl --quantize
cs677 select --input data/scored.jsonl --output data/selected

cs677 train --train-file data/selected/baseline.jsonl --output runs/baseline
cs677 train --train-file data/selected/epistemic.jsonl --output runs/epistemic

cs677 evaluate --adapter runs/baseline --benchmark math500 --output runs/baseline-math500
cs677 evaluate --adapter runs/epistemic --benchmark math500 --output runs/epistemic-math500
cs677 evaluate --adapter runs/baseline --benchmark gsm8k --output runs/baseline-gsm8k
cs677 evaluate --adapter runs/epistemic --benchmark gsm8k --output runs/epistemic-gsm8k
```

**Hardware:** the default 72B teacher requires substantial accelerator memory. Full generation and 600,000 stochastic student passes are expensive. They were not run as part of restoration. Use existing teacher JSONL data or a suitable GPU machine for the complete study. `--quantize` is a CUDA-only path in this implementation; omit it on a Mac/CPU. A smaller teacher selected with `--model` is a different experiment. Start with `--limit 2` on `prepare`, and use a new output path for every run.

Run `cs677 COMMAND --help` for parameters. [REPRODUCING.md](docs/REPRODUCING.md) covers file schemas, memory, revisions, assumptions, and validation. Output files/directories are protected against accidental reuse. Long jobs write completed JSONL records progressively; automatic resume is not implemented.

## Project layout

```text
sources/                 Original paper, presentation, partial notebook, checksums
src/cs677/               Reconstructed numerical core and model pipeline
configs/paper.json       Reported settings and separate reconstruction defaults
results/reported/        Historical result transcriptions in JSON and CSV
docs/                    Theory, recovery record, reproducibility notes, figures
tests/                   Numerical, data, grading, tiny-Qwen, and LoRA tests
.github/workflows/       Offline core and tiny-model CI checks
```

## Interpretation and future work

The paper's conclusion is a promising proof of concept with modest evidence. Changing the uncertainty measure substantially changed the training examples but barely changed strict accuracy. Alternative teacher trajectories may have similar SFT value, or the metric and experiment may be insufficient to reveal meaningful differences; these are interpretations, not established mechanisms.

The proposed next steps are stronger answer normalization/equivalence checking, diversity-aware selection, larger ablations and selection budgets, multiple training seeds, and better preservation of uncertainty during distillation. Useful additional controls include random selection, a deterministic entropy baseline, dropout-rate and pass-count ablations, and correctness filtering of teacher solutions. These extensions have not been evaluated in the recovered study.

## References

The original paper cites:

1. Wei et al. (2022). [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903).
2. Zelikman et al. (2022). [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465).
3. Cobbe et al. (2021). [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168).
4. Hendrycks et al. (2021). [Measuring Mathematical Problem Solving With the MATH Dataset](https://arxiv.org/abs/2103.03874).
5. Kendall & Gal (2017). [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://proceedings.neurips.cc/paper/2017/hash/2650d6089a6d640c5e85b2b88265dc2b-Abstract.html).
6. Gal & Ghahramani (2016). [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://proceedings.mlr.press/v48/gal16.html).

Implementation sources: [Qwen2.5-3B model card](https://huggingface.co/Qwen/Qwen2.5-3B), [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct), [PEFT LoRA documentation](https://huggingface.co/docs/peft/en/package_reference/lora), [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), and [GSM8K](https://huggingface.co/datasets/openai/gsm8k). Model and dataset licenses are separate; weights and downloaded datasets are not distributed here.
