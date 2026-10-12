# Experimental method

[Back to the project](../README.md) · [Mathematical derivations](THEORY.md) · [Run the pipeline](REPRODUCING.md)

The final paper defines the reported experiment. Settings recovered from the notebook and choices made during reconstruction are identified separately below.


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

The paper mentions LoRA dropout but does not give its probability or target modules. The reconstruction uses **0.05** and Qwen attention/MLP projection modules. Prompts, seed 42, decoding settings, completion-only loss, EOS handling, right truncation, optimizer schedule, and other unrecovered details are explicit reconstruction choices in [configs/paper.json](../configs/paper.json).

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

Here $p_i>0$ is a prior and $\eta$ enforces normalization. [`chi_squared_weights`](../src/cs677/selection.py) solves the threshold numerically. This supplies a usable version of the proposed idea, but it is **not recovered original code and has no reported benchmark results**. It is not applied by the paper reproduction commands. Derivation and assumptions are in [THEORY.md](THEORY.md).

## References


The original paper cites:

1. Wei et al. (2022). [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903).
2. Zelikman et al. (2022). [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465).
3. Cobbe et al. (2021). [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168).
4. Hendrycks et al. (2021). [Measuring Mathematical Problem Solving With the MATH Dataset](https://arxiv.org/abs/2103.03874).
5. Kendall & Gal (2017). [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://proceedings.neurips.cc/paper/2017/hash/2650d6089a6d640c5e85b2b88265dc2b-Abstract.html).
6. Gal & Ghahramani (2016). [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://proceedings.mlr.press/v48/gal16.html).

Implementation sources: [Qwen2.5-3B model card](https://huggingface.co/Qwen/Qwen2.5-3B), [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct), [PEFT LoRA documentation](https://huggingface.co/docs/peft/en/package_reference/lora), [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), and [GSM8K](https://huggingface.co/datasets/openai/gsm8k). Model and dataset licenses are separate; weights and downloaded datasets are not distributed here.
