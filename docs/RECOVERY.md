# Recovery record

Restored on 2026-10-12 from three files in the author's Downloads folder. A filename search in Downloads, Documents, and Desktop found no other relevant CS 677 project files. The connected GitHub repository listing did not contain an existing CS 677 repository. This was a targeted search, not forensic disk recovery.

## Preserved originals

| Original filename | Archive | What survived |
| --- | --- | --- |
| CS677 Fianl Paper.pdf | [final-paper.pdf](../sources/final-paper.pdf) | Six-page final paper: method, settings, all result tables, discussion, references |
| CS677 Final Presentation.pptx | [final-presentation.pptx](../sources/final-presentation.pptx) | Nine-slide presentation, including equations and charts stored as images |
| CS677 Final Project.ipynb | [original-notebook.ipynb](../sources/original-notebook.ipynb) | Eleven cells with setup code, recorded outputs, and a five-phase proposal |

All three copies are byte-identical to their Downloads sources. SHA-256 hashes are in [manifest.json](../sources/manifest.json). Text extracts are convenience files; they do not replace the original layouts, equations, or chart images.

## Recovered executable code

The notebook imports Torch, Transformers, and Datasets; sets Qwen2.5-3B attention dropout to 0.1; loads NF4 4-bit weights with double quantization and bfloat16 compute; activates training mode with gradients disabled; compares two stochastic forward passes; and loads the 12,000-problem training pool.

Its recorded dropout check reports unequal logits and a summed absolute logit difference of 3,506,176. The notebook labels this “variance,” but it is a summed absolute difference, not a statistical variance estimate. That old diagnostic is not a new validation result. The recorded logits have shape `[1, 22, 151936]`.

The remaining phases are Markdown descriptions. No candidate-generation loop, MI computation, selection implementation, SFT loop, or benchmark evaluator survives in the notebook.

## Reconstructed components

| Component | Source basis | Restoration status |
| --- | --- | --- |
| Data preparation and MATH-500 overlap check | Notebook dataset ID; paper problem count | New implementation |
| Five teacher candidates per problem | Paper page 2; slide 5 | New implementation |
| Ten MC passes and entropy decomposition | Paper page 2; slide 4 equations | New implementation |
| Top-50 token aggregation | Paper page 2 | New implementation |
| Per-problem argmax selection | Paper page 2 | New implementation |
| Chi-squared regularized weighting | Notebook phase 3 idea only | Optional illustrative derivation, not original formula |
| Two LoRA adapters | Paper page 2 | New implementation with documented defaults |
| First-boxed strict ACC@1 | Paper page 3; slide 6 | New implementation |
| Historical result JSON/CSV and chart | Paper Tables 1–3 | Transcription, not rerun |
| Tests and GitHub Actions | Added during restoration | New validation infrastructure |

The final paper is authoritative for the **reported experiment**. The notebook is authoritative for the surviving setup and earlier proposal. Disagreements are retained explicitly: the notebook suggests a point-estimate entropy baseline and chi-squared selection; the paper instead compares MC predictive entropy with MC mutual information using argmax winners.

## Missing artifacts and unknowns

The recovered files contain no teacher trajectory dataset, original scores, adapter weights, checkpoints, optimizer state, per-question predictions, run logs, or dependency lockfile. Exact original reproduction cannot be guaranteed.

Unspecified details include exact teacher prompt; teacher sampling temperature/top-p; seeds; decoding limits and policy; original model/dataset revisions; LoRA dropout probability and target modules; optimizer and schedule; precision during SFT; prompt/response token masking; EOS treatment; truncation; tie handling; and whether teacher trajectories were correctness-filtered. All restoration defaults are listed separately in [configs/paper.json](../configs/paper.json). No teacher correctness filtering is added to the paper pipeline because it was not described.

The restored implementation writes run settings, package versions, dataset fingerprints where available, resolved model commit hashes, and input file hashes. Future runs should also pin dataset/model revisions explicitly and retain raw predictions.

## Interpretation

The reconstruction restores a usable research project and its complete surviving scientific record. It does not claim to recover the deleted original implementation byte for byte, or to reproduce the historical benchmark outcomes. The full experiment needs substantial compute and has not been rerun during restoration.
