# Theory and the proposed selection rules

This document restates the final paper's uncertainty method and derives a clearly labeled optional version of the notebook's earlier chi-squared proposal. The equations in the presentation were embedded as images; the original PPTX preserves them.

## Motivation: uncertainty is not a single quantity

Let $x$ be a math problem and $y=(y_1,\dots,y_T)$ a fixed teacher trajectory. A student predicts a distribution over the vocabulary at each step conditioned on $x,y_{<t}$. A broad distribution can reflect multiple valid ways to continue, unusual wording, or missing knowledge. The project's hypothesis is that separating disagreement between model samples from uncertainty within each sample will improve data curation.

In a Bayesian treatment, parameters have a posterior $p(\theta\mid\mathcal D)$ and the posterior predictive distribution averages over parameters. MC dropout replaces this posterior with an approximate distribution induced by stochastic dropout masks. The restored implementation keeps Qwen attention dropout active for repeated teacher-forced passes with no gradient updates.

## Entropy decomposition

For a fixed context $c_t=(x,y_{<t})$, write $p_k(v)=p(y_t=v\mid c_t,\theta_k)$ and $\bar p=K^{-1}\sum_kp_k$.

$$H[\bar p]=-\sum_v\bar p(v)\log\bar p(v),\qquad
\mathbb E_kH[p_k]=-\frac1K\sum_k\sum_vp_k(v)\log p_k(v).$$

The empirical mutual information is

$$I(y_t;\theta\mid c_t,\mathcal D)\approx H[\bar p]-\mathbb E_kH[p_k]
=\frac1K\sum_kD_{\mathrm{KL}}(p_k\Vert\bar p).$$

The KL identity shows that MI is nonnegative in exact arithmetic and zero when all samples give the same distribution. Logs are natural, so values are in nats. MI is at most $\min(\log K,\log|V|)$ for this finite ensemble. Floating-point subtraction can produce tiny negatives; the implementation clamps those to zero.

Expected entropy is commonly interpreted as an aleatoric component and MI as an epistemic component **within this approximation**. Neither is a causal diagnosis of an individual token's uncertainty. Dropout rate, model calibration, prior training, finite K, quantization, and distribution shift can affect the estimate. In particular, the recovered notebook adds dropout at scoring time; it does not demonstrate posterior calibration for pretrained Qwen weights.

Two diagnostic examples over a two-token vocabulary:

| MC predictions | Total entropy | Expected entropy | MI |
| --- | ---: | ---: | ---: |
| Every pass is (0.5, 0.5) | ln 2 | ln 2 | 0 |
| Half are (1, 0), half (0, 1) | ln 2 | 0 | ln 2 |

The same total entropy can therefore arise from very different patterns of disagreement. The numerical tests cover these cases.

## Teacher forcing, alignment, and aggregation

Each pass conditions on the same observed trajectory prefix. It does not sample a new response. In a causal LM, logits at position $j$ predict the token at $j+1$. If a prompt contains P tokens and the full example has L tokens, completion scoring uses hidden-state positions $P-1,\ldots,L-2$. The reconstruction excludes prompt tokens and includes the completion's EOS if it fits; these masking details were not preserved in the sources.

For an uncertainty sequence $u_1,\dots,u_T$, the project averages its top 50 values. This emphasizes uncertain regions and avoids averaging them away among easy tokens. It can still be sensitive to outliers and trajectory length; no recovered ablation establishes that 50 is optimal. The implementation averages all tokens when T < 50.

The baseline independently ranks total-entropy tokens. The MI method independently ranks MI tokens. **Subtracting the top-50 expected-entropy mean from the top-50 total-entropy mean is not the same operation** as taking the top-50 mean of tokenwise MI; the implementation uses the latter.

The paper selects the argmax among five candidates for each problem. There is no global budget optimization or stochastic weighting in that reported experiment. The restored tie-break chooses the lexicographically smallest candidate ID, an implementation choice because the original tie handling is unknown.

## Optional chi-squared proposal reconstruction

The notebook mentions normalized utility and a chi-squared-regularized closed-form selection rule, but preserves neither a formula nor working code. The following is a standard, explicit instantiation supplied during restoration. It must not be attributed to the original experiment.

Given utilities $u_i$, a strictly positive reference distribution $p_i$, and $\lambda>0$, solve

$$\max_{q\ge0,\,\sum_iq_i=1}\;q^\top u-\frac\lambda2D_{\chi^2}(q\Vert p),\qquad
D_{\chi^2}(q\Vert p)=\sum_i\frac{(q_i-p_i)^2}{p_i}.$$

The factor $1/2$ and divergence direction are part of this reconstruction. The objective is strictly concave on the simplex. For an active coordinate, the derivative of the Lagrangian gives

$$u_i-\lambda\frac{q_i-p_i}{p_i}-\eta=0.$$

Combining stationarity with nonnegativity yields

$$q_i=p_i\max\left(0,1+\frac{u_i-\eta}{\lambda}\right).$$

Choose $\eta$ so $\sum_iq_i=1$. The left side is continuous and monotone in $\eta$, so bisection finds the unique probability vector. This is the method implemented by `chi_squared_weights`; it is a closed-form coordinate rule with a numerically solved normalization threshold.

If all coordinates remain active, $\eta=\sum_ip_iu_i$ and

$$q_i=p_i\left(1+\frac{u_i-\mathbb E_pu}{\lambda}\right).$$

Equal utilities return the prior. Large regularization favors the prior; small regularization concentrates mass on maxima. A constant offset in all utilities leaves q unchanged. Scaling utilities changes the effective regularization, so a normalization choice must be reported. The library deliberately does not impose an undocumented normalization; callers may, for example, use min-max scaling with a constant-vector fallback.

```python
import numpy as np
from cs677.selection import chi_squared_weights

raw = np.array([0.04, 0.08, 0.15, 0.06, 0.10])
utilities = (raw - raw.min()) / np.ptp(raw) if np.ptp(raw) else np.zeros_like(raw)
weights = chi_squared_weights(utilities, regularization=1.0)
```

This returns weights, not an already selected dataset. Sampling, budget allocation, diversity constraints, and any resulting benchmark outcomes are unimplemented research choices. The offline demo exposes the weights only to illustrate the proposed mechanism.

## Distillation objective

For a selected dataset $\mathcal S$, supervised fine-tuning minimizes completion-token negative log-likelihood:

$$\mathcal L(\phi)=-\sum_{(x,y)\in\mathcal S}\sum_t\log p_{\theta,\phi}(y_t\mid x,y_{<t}).$$

Base parameters $\theta$ are frozen and LoRA parameters $\phi$ are trained. A projection becomes $W'=W+(\alpha/r)BA$ with low-rank matrices A and B. The paper reports r=16 and alpha=32. The restored training command uses completion-only labels and masks prompt/padding positions; the source does not state the original label mask.

## What the results can support

The two reported training datasets differ on 67.54% of problem-level winners. Benchmark gains are +1/500 and +2/1319 answers. Aggregate counts cannot recover which predictions changed, their overlap, or run-to-run variability. Subject and difficulty patterns are descriptive; causal explanations about knowledge gaps remain hypotheses. Exact-string grading is particularly sensitive to harmless formatting differences.

The paper and slides propose better grading, diversity-aware curation, wider budgets/ablations, and improved uncertainty preservation. Multiple seeds and raw prediction logs would make future comparisons auditable.

Primary theoretical references: [Gal & Ghahramani, 2016](https://proceedings.mlr.press/v48/gal16.html) and [Kendall & Gal, 2017](https://proceedings.neurips.cc/paper/2017/hash/2650d6089a6d640c5e85b2b88265dc2b-Abstract.html). The optional chi-squared derivation above is supplied explicitly as part of the restoration.
