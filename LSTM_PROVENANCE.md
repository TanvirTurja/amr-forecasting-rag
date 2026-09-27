# LSTM run-to-run provenance

The single-step LSTM is the only nondeterministic model in the benchmark: cuDNN's
recurrent kernels are not run-to-run reproducible on GPU, even with `SEED = 42`
and identical hardware. Two executions of `training.ipynb` on the same machine
(RTX 5060 Laptop GPU) produced different LSTM results while every deterministic
model reproduced bit-exactly:

| Execution | LSTM test MAE | 95% cluster CI | RMSE | R2 | XGBoost − LSTM paired test |
|---|---|---|---|---|---|
| **R1** — 2026-09-26 02:37 (artifacts: `model_mae_bootstrap_ci.csv`) | **7.89** | 6.99–8.91 | 11.16* | 0.845* | Δ = −1.34, p < 0.001* |
| **R2** — 2026-09-26 11:58 (notebook saved outputs + `model_comparison_results.csv`) | **7.06** | 6.26–8.02 | 10.56 | 0.861 | Δ = −0.51, p = 0.030 |

\* The R1 execution's per-row test predictions were not persisted. Only its MAE
and cluster CI survive, in `model_mae_bootstrap_ci.csv`; the starred values are
quoted from the manuscript and cannot be regenerated from shipped artifacts.

## What this means

- **The manuscript (under review) reports the R1 row** (MAE 7.89, CI 6.99–8.91).
  Its MAE and CI are backed by `model_mae_bootstrap_ci.csv`; its RMSE, R², sMAPE,
  and the XGBoost−LSTM contrast come from the unpersisted R1 run.
- **The shipped notebook execution (R2) and `model_comparison_results.csv` report
  7.06 / 10.56 / 0.861** with paired p = 0.030. The two CSVs therefore disagree on
  the LSTM row by construction; all other rows are float-identical across R1/R2,
  which is the bit-reproducibility evidence for the deterministic models.
- Observed same-hardware run-to-run spread: **0.83 pp** — larger than every model
  contrast reported in the paper except persistence−XGBoost (0.76). This is why
  the paper states that the tree ensembles and linear models, which are
  deterministic, carry the benchmark's conclusions.
- In both runs the qualitative position of the LSTM is identical: it trails
  persistence, the hybrid, linear, ridge, XGBoost, and LightGBM, and XGBoost
  beats it significantly. No conclusion in the paper depends on which run is
  quoted.

## Policy going forward

To prevent this class of provenance gap, any future execution of `training.ipynb`
should persist per-row test predictions for every model (e.g.,
`test_predictions.csv`) so that every reported number — including the LSTM's —
can be regenerated from shipped artifacts.
