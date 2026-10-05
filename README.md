# AMR Trend Forecasting with Policy Q&A

This project forecasts antimicrobial resistance (AMR) rates using WHO GLASS surveillance data and pairs those forecasts with a local RAG-based Q&A system that can answer policy questions grounded in WHO documents.

The paper is on arXiv: **[arxiv.org/abs/2602.22673](https://arxiv.org/abs/2602.22673)**. Version 3 (posted 26 September 2026) is a correction: a leakage audit found four feature-construction defects in the original benchmark and withdrew its headline claim. The original v1/v2 text is preserved below the correction notice; every number in this README comes from the corrected analysis.

---

## What this does

**Part 1 — Forecasting benchmark, with a leakage audit**

A leakage audit of the original benchmark found four feature-construction defects in the
legacy panel: a lag feature equal to the target itself on 8.7% of test rows (D1),
specimen-crossed lags (D2), surveillance quality flags computed over the full 2020–2023
period (D3), and full-panel imputation (D4). `Build_AMR_Dataset.ipynb` rebuilds the panel
from the raw WHO GLASS exports with all four repaired; `training.ipynb` then replays the
defective pipeline on the legacy CSV as a regression test (it reproduces the published
6.130 XGBoost MAE exactly) and benchmarks the corrected panel under a strict temporal
split (train 2021, validation 2022, test 2023).

On the corrected panel, three reference forecasts and six learning algorithms are compared
with country-clustered bootstrap intervals and paired tests:

- Reference forecasts: persistence (lag-1), per-series train mean, global train mean
- Models: Linear, Ridge, XGBoost, LightGBM, MLP, LSTM
- Plus a parameter-free **hybrid selector**: persistence where the series has a prior-year
  observation, XGBoost on cold-start series.

**Result:** the original claim that XGBoost cut error by 85.3% versus a naive baseline was
an artifact of the defects and is withdrawn. On the leakage-free panel a one-year
persistence forecast (MAE 5.79 pp) significantly outperforms XGBoost (6.55 pp;
paired country-cluster bootstrap p = 0.001), and the hybrid selector is best overall
(5.74 pp). Machine learning earns its keep only on the 13% of test series with no
prior-year observation. Feature attribution uses XGBoost gain importance (prior-year
resistance 54.0%); see `xgb_feature_importance.csv`.

**Part 2 — Policy Q&A**

A Retrieval-Augmented Generation (RAG) pipeline built with ChromaDB, BM25+dense retrieval
(reciprocal rank fusion), and a local Gemma 4 model translates the forecasts into policy
answers grounded in six foundational WHO policy documents, with mandatory source tags.
Its evaluation instrument is itself validated: `judge_validation.ipynb` measures three
local LLM judges on 100 deliberately corrupted + 25 clean answers and only uses a judge
that clears pre-registered gates (detection ≥ 0.80, false-flag rate ≤ 0.20). The
validated judge scores 21 of 25 real answers clean; the four flags are attribution issues
queued for expert adjudication. It runs fully locally — no API keys, no cloud services.

---

## Files

```
Build_AMR_Dataset.ipynb               # Rebuilds the audited panel from raw WHO GLASS exports
AMR_Training_Dataset_Fixed.csv        # Audited panel (6,276 rows, 2021–2023; 2020 kept for lags)
AMR_Training_Dataset_Final.csv        # Legacy defective panel (5,909 rows) — kept for the replay test
training.ipynb                        # Stage A defective replay + corrected benchmark + ablations
rag_pipeline.ipynb                    # RAG system for policy Q&A (25 demo questions)
judge_validation.ipynb                # Corruption-validated judge audit (checkpointed, 375 calls)
LSTM_PROVENANCE.md                    # Why the LSTM row differs across runs/artifacts
model_comparison_results.csv          # Corrected benchmark results (shipped run)
model_mae_bootstrap_ci.csv            # 95% country-cluster bootstrap CIs
regional_mae_extended.csv             # Regional MAE (hybrid selector)
xgb_feature_importance.csv            # XGBoost gain importance
ablation_corrected.csv                # Feature-group ablations
AMR_Data/ + income_group_world_bank.xlsx   # Raw WHO GLASS exports and income groups
legacy/                               # Defective-pipeline outputs (old figures, legacy CSVs)
```

---

## Running the code

**Requirements**

```bash
pip install pandas numpy scikit-learn xgboost lightgbm torch matplotlib seaborn category_encoders chromadb sentence-transformers pymupdf langchain-text-splitters rank_bm25 ollama
```

**Steps**

1. Clone the repo. Both `AMR_Training_Dataset_Final.csv` and `AMR_Training_Dataset_Fixed.csv` must be present for `training.ipynb` (Stage A replays the defective pipeline on the former).
2. (Optional) Rebuild the audited panel from scratch with `Build_AMR_Dataset.ipynb` — requires `AMR_Data/` and `income_group_world_bank.xlsx`.
3. Run `training.ipynb` end to end: it replays the defective benchmark (expect XGBoost MAE 6.130, MATCH), then produces the corrected benchmark, regime analysis, and ablations.
4. Run `rag_pipeline.ipynb` for the policy Q&A system (requires [Ollama](https://ollama.com) with the generator model pulled: `ollama pull gemma4:e4b`).
5. Run `judge_validation.ipynb` to reproduce the judge validation (requires Ollama with `qwen3.5:9b`, `gpt-oss:20b`, `llama3.1:8b`; with the shipped `judge_audit_checkpoint.csv` it makes zero LLM calls).

Note: the LSTM is the only nondeterministic model in the benchmark (GPU/cuDNN run-to-run
variance). Its row varies across executions — see `LSTM_PROVENANCE.md`; all other models
are bit-reproducible.

---

## Data

The dataset comes from the [WHO GLASS surveillance system](https://www.who.int/initiatives/glass).
The audited panel covers 44 countries across **five** WHO regions (African, Eastern
Mediterranean, European, South-East Asia, Western Pacific; no Region-of-the-Americas
country met the contiguity requirement) and contains **6,276 observations** for 2021–2023
(train 1,940 / validation 2,114 / test 2,222), with 2020 records retained only to supply
prior-year values. 2020 was otherwise excluded, and the rebuilt extraction recovered 367
genuine observations that the legacy panel had dropped (legacy: 5,909).

---

## Results

One-year-ahead forecast error on the 2023 test set of the audited panel (n = 2,222;
95% country-cluster bootstrap CIs):

| Model | Test MAE | 95% CI | Test R² |
|-------|----------|--------|---------|
| **Hybrid selector** | **5.74** | [4.73, 6.85] | — |
| **Persistence (lag-1)** | **5.79** | [4.74, 6.93] | 0.859 |
| Linear Regression | 6.39 | [5.45, 7.47] | 0.871 |
| Ridge | 6.41 | [5.47, 7.49] | 0.871 |
| LightGBM | 6.53 | [5.63, 7.51] | 0.873 |
| XGBoost | 6.55 | [5.62, 7.50] | 0.876 |
| LSTM † | 7.06 | [6.26, 8.02] | 0.861 |
| Series train mean | 7.23 | [5.96, 8.49] | 0.810 |
| MLP | 9.32 | [7.92, 11.01] | 0.783 |
| Global train mean | 24.68 | [23.10, 26.35] | 0.000 |

† Nondeterministic across runs; the manuscript quotes an earlier execution
(7.89 [6.99, 8.91]) — see `LSTM_PROVENANCE.md`.

Regional MAE (hybrid selector) ranges from 3.03 pp (European Region) to 7.91 pp
(South-East Asia Region), tracking surveillance data availability. South-East Asia is the
only region where XGBoost beats persistence.

**Legacy (defective) results**, reproduced exactly by the Stage A replay and superseded by
the above: XGBoost 6.13, LightGBM 6.30, LSTM 7.16, Linear 8.13, Ridge 8.15, naive
41.79 — the basis of the withdrawn 85.3% claim (`legacy/model_comparison_results_legacy.csv`).

## Citation

If you use this code or data in your research, please cite the preprint:

**Plain text:**
> Turja, M. T. H. (2026). Forecasting Antimicrobial Resistance Trends Using Machine Learning on WHO GLASS Surveillance Data: A Retrieval-Augmented Generation Approach for Policy Decision Support. *arXiv preprint arXiv:2602.22673* (v3 documents the corrected analysis).

**BibTeX:**
```bibtex
@misc{turja2026forecastingantimicrobialresistancetrends,
      title={Forecasting Antimicrobial Resistance Trends Using Machine Learning on WHO GLASS Surveillance Data: A Retrieval-Augmented Generation Approach for Policy Decision Support}, 
      author={Md Tanvir Hasan Turja},
      year={2026},
      eprint={2602.22673},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.22673}, 
      note={v3 documents the leakage audit that corrected these results}
}
```

---

## Author

Md Tanvir Hasan Turja  
Independent Researcher  
[github.com/TanvirTurja](https://github.com/TanvirTurja)
