# AMR Trend Forecasting with Policy Q&A

This project forecasts antimicrobial resistance (AMR) rates using WHO GLASS surveillance data and pairs those forecasts with a local RAG-based Q&A system that can answer policy questions grounded in WHO documents.

The paper is on arXiv: **[arxiv.org/abs/2602.22673](https://arxiv.org/abs/2602.22673)**

---

## What this does

**Part 1 — Forecasting**

Six models are trained on WHO GLASS data (2021–2023) to predict resistance percentages at the country-pathogen-antibiotic level:

- Naive baseline
- Linear Regression
- Ridge Regression
- XGBoost
- LightGBM
- LSTM

XGBoost came out on top with a test MAE of 7.07% and R² of 0.854.

**Part 2 — Policy Q&A**

A RAG pipeline built with ChromaDB and Phi-3 Mini answers questions like *"Which antibiotics should Southeast Asia prioritize?"* by pulling from WHO policy documents and injecting the forecast results as context. It runs fully locally — no API keys, no cloud services.

---

## Files

```
AMR_Training_Dataset_Final.csv   # Processed WHO GLASS dataset (2021–2023)
training.ipynb                   # Full training pipeline
model_comparison_results.csv     # Model evaluation results
model_comparison.png             # Figure 1
xgb_feature_importance.png       # Figure 2
mae_by_region.png                # Figure 3
error_analysis.png               # Figure 4
paper_arxiv.pdf                  # The paper
paper_arxiv.tex                  # LaTeX source
references.bib                   # Bibliography
```

---

## Running the notebook

**Requirements**

```bash
pip install pandas numpy scikit-learn xgboost lightgbm torch matplotlib seaborn category_encoders
```

**Steps**

1. Clone the repo
2. Make sure `AMR_Training_Dataset_Final.csv` is in the same folder as the notebook
3. Open `training.ipynb` and run all cells

For the RAG/Q&A part, you also need [Ollama](https://ollama.com) with Phi-3 Mini pulled:

```bash
ollama pull phi3:mini
```

---

## Data

The dataset comes from the [WHO GLASS surveillance system](https://www.who.int/initiatives/glass), covering resistance rates across 44 countries and 6 WHO regions. The version used here spans 2021–2023 and has 5,909 observations after preprocessing.

---

## Results

| Model | Test MAE | Test R² |
|-------|----------|---------|
| Naive | 41.83% | — |
| Linear Regression | 8.23% | 0.821 |
| Ridge | 8.25% | 0.821 |
| LightGBM | 7.17% | 0.853 |
| LSTM | 7.16% | 0.872 |
| **XGBoost** | **7.07%** | **0.854** |

Regional MAE ranged from 4.16% (European Region) to 10.14% (South-East Asia Region).

---

## Author

Md Tanvir Hasan Turja
MSc Data Science, Middlesex University London
[github.com/TanvirTurja](https://github.com/TanvirTurja)
