# Reported results

[Back to the project](../README.md) · [Original paper](../sources/final-paper.pdf) · [Machine-readable tables](../results/reported/)

These are historical results transcribed from the final paper. The reconstructed implementation has not independently reproduced them. Original trajectories, adapters, and per-example evaluation logs were not recovered.

## Overall comparison


MI selection changed the chosen trajectory for **8,105 of 12,000 problems (67.54%)**, but the reported accuracy gains were small: one additional correct answer on MATH-500 and two on GSM8K.

| Benchmark | Total entropy baseline | Epistemic MI | Change |
| --- | ---: | ---: | ---: |
| MATH-500 | 46.20% (231/500) | 46.40% (232/500) | +0.20 percentage points |
| GSM8K | 79.30% (1046/1319) | 79.45% (1048/1319) | +0.15 percentage points |

Source: original paper, Table 1, page 3. These small differences do not establish statistically reliable superiority. Multiple seeds and paired prediction logs would be needed for a stronger comparison.

![Reported benchmark results](assets/reported-accuracy.png)


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

The JSON and CSV transcriptions are in [`results/reported/`](../results/reported/). Both MATH-500 breakdowns sum to the overall counts (231 baseline, 232 MI). Original per-example predictions were not recovered, so paired significance tests and confidence intervals on the paired difference cannot be reconstructed from these tables alone.

