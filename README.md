# AMR Trend Forecasting with Policy Q&A

This project forecasts antimicrobial resistance (AMR) rates using WHO GLASS surveillance data and pairs those forecasts with a local RAG-based Q&A system that can answer policy questions grounded in WHO documents.

The paper is on arXiv: **[arxiv.org/abs/2602.22673](https://arxiv.org/abs/2602.22673)**

---

## What this does

**Part 1 — Forecasting**

Six models are trained on WHO GLASS data (2021–2023) to predict resistance percentages at the country-pathogen-antibiotic level. To prevent data leakage, a **strict temporal split** is enforced (Train: 2021-2022, Test: 2023).

- Naive baseline
- Linear Regression
- Ridge Regression
- XGBoost
- LightGBM
- LSTM

XGBoost achieved the highest performance with a test MAE of 6.13% (95% CI: 5.83--6.44) and an $R^2$ of 0.885, representing an **85.3% error reduction** over the Naive Baseline. We also implemented **SHAP** analysis to provide mathematically consistent, direction-aware interpretations of how prior-year resistance and antibiotic consumption drive forecasts.

**Part 2 — Policy Q&A**

A Retrieval-Augmented Generation (RAG) pipeline built with ChromaDB and Gemma 4 translates the computational forecasts into actionable policy. We deliberately constrained the knowledge base to **six foundational WHO policy documents** to minimize retrieval noise and guarantee 100% citation faithfulness. It runs fully locally — no API keys, no cloud services.

---

## Files

```
AMR_Training_Dataset_Final.csv        # Processed WHO GLASS dataset (2021–2023)
training.ipynb                        # Full training pipeline (models, CIs, SHAP)
rag_pipeline.ipynb                    # RAG system for policy Q&A
```

---

## Running the code

**Requirements**

```bash
pip install pandas numpy scikit-learn xgboost lightgbm torch matplotlib seaborn category_encoders shap chromadb sentence-transformers
```

**Steps**

1. Clone the repo
2. Ensure `AMR_Training_Dataset_Final.csv` is in the same folder as the notebooks
3. Open `training.ipynb` and run all cells to reproduce the forecasts and SHAP values.
4. Open `rag_pipeline.ipynb` to query the WHO policy system.

For the RAG/Q&A part, you also need [Ollama](https://ollama.com) installed with the Gemma 4 model pulled:

```bash
ollama pull gemma4:e4b
```

---

## Data

The dataset comes from the [WHO GLASS surveillance system](https://www.who.int/initiatives/glass), covering resistance rates across 44 countries and 6 WHO regions. The dataset used here spans 2021–2023 and contains 5,909 observations after preprocessing. Data prior to 2021 was excluded due to inconsistent lag-1 variables caused by pandemic reporting disruptions.

---

## Results

Model evaluation on the 2023 held-out test set with 95% Bootstrap Confidence Intervals:

| Model | Test MAE | 95% CI | Test R² |
|-------|----------|--------|---------|
| Naive Baseline | 41.79% | [40.77, 42.79] | — |
| Linear Regression | 8.13% | [7.78, 8.49] | 0.830 |
| Ridge | 8.15% | [7.79, 8.50] | 0.830 |
| LSTM | 7.15% | [6.79, 7.52] | 0.853 |
| LightGBM | 6.30% | [6.00, 6.63] | 0.881 |
| **XGBoost** | **6.13%** | **[5.83, 6.44]** | **0.885** |

Regional MAE ranged from 3.65% (European Region) to 8.61% (South-East Asia Region), closely tracking with surveillance data availability.

## Citation

If you use this code or data in your research, please cite the preprint:

**Plain text:**
> Turja, M. T. H. (2026). Forecasting Antimicrobial Resistance Trends Using Machine Learning on WHO GLASS Surveillance Data: A Retrieval-Augmented Generation Approach for Policy Decision Support. *arXiv preprint arXiv:2602.22673*.

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
}
```

---

## Author

Md Tanvir Hasan Turja  
Independent Researcher
[github.com/TanvirTurja](https://github.com/TanvirTurja)
