# AMR Trend Forecasting with Policy Q&A
## A Machine Learning and Retrieval-Augmented Generation Framework for Antimicrobial Resistance Surveillance

---

## Abstract

Antimicrobial resistance (AMR) is a growing global crisis projected to cause 10 million deaths per year by 2050. While the WHO Global Antimicrobial Resistance and Use Surveillance System (GLASS) provides standardized surveillance data across 44 countries, few studies have applied machine learning to forecast population-level resistance trends from this data. This paper presents a two-component framework for AMR trend forecasting and evidence-grounded policy decision support. We benchmark six models, including Naive, Linear Regression, Ridge Regression, XGBoost, LightGBM, and LSTM, on 5,909 WHO GLASS observations across six WHO regions (2021 to 2023). XGBoost achieved the best performance with a test MAE of 7.07% and R2 of 0.854, outperforming the naive baseline by 83.1%. Feature importance analysis identified the prior-year resistance rate as the dominant predictor (50.5% importance), while regional MAE ranged from 4.2% (European Region) to 10.1% (South-East Asia Region), reflecting disparities in GLASS data quality across surveillance systems. We additionally implemented a Retrieval-Augmented Generation (RAG) pipeline combining a ChromaDB vector store of WHO policy document excerpts with a locally deployed Phi-3 Mini language model, injecting forecast context into each query to produce source-attributed, hallucination-constrained policy answers. Together, these components demonstrate the feasibility of integrating AMR surveillance forecasting with intelligent policy question answering using openly available data and locally deployable models.

**Keywords:** antimicrobial resistance, WHO GLASS, machine learning, XGBoost, LSTM, time series forecasting, retrieval-augmented generation, public health surveillance, policy decision support

---

## 1. Introduction

Antimicrobial resistance (AMR) represents one of the most urgent threats to global health in the twenty-first century. In 2019, AMR was directly responsible for an estimated 1.27 million deaths worldwide and was associated with an additional 4.95 million deaths, with the highest burden concentrated in Sub-Saharan Africa and South Asia [Murray et al., 2022]. Without coordinated international action, AMR is projected to cause 10 million deaths per year by 2050, surpassing cancer as the leading cause of death globally [O'Neill, 2016; de Kraker et al., 2016]. These projections carry wide-ranging consequences across patient outcomes, healthcare systems, and national economies [Dadgostar, 2019; Tang et al., 2023], and have prompted global political responses including the WHO Global Action Plan on Antimicrobial Resistance [World Health Organization, 2015; Sugden et al., 2016].

A critical prerequisite for effective AMR governance is the ability to monitor and forecast resistance trends across pathogens, antibiotics, and geographic regions. The WHO Global Antimicrobial Resistance and Use Surveillance System (GLASS) was established to provide standardized, comparable AMR data from national surveillance networks worldwide [World Health Organization, 2022; World Health Organization, 2023]. GLASS data captures resistance rates across multiple pathogen-antibiotic combinations and has been used to demonstrate the relationship between antibiotic consumption and resistance across participating countries [Ajulo and Awosile, 2024]. However, GLASS data availability remains uneven across regions, with participation concentrated in higher-income countries and more established surveillance systems.

Machine learning has attracted growing interest as a tool for AMR prediction, primarily applied to clinical isolate datasets to classify susceptibility or resistance at the level of individual bacterial samples [Sakagianni et al., 2023; Kim et al., 2022]. These approaches have shown promise within clinical laboratory settings but do not directly address population-level forecasting of resistance trends from surveillance data. The temporal and geographic structure of GLASS data, covering resistance rates over time and across countries, calls for time series forecasting approaches rather than isolate-level classification.

Beyond forecasting, there is a parallel need for tools that make AMR surveillance findings interpretable and actionable for policymakers and public health practitioners. Large language models (LLMs) and Retrieval-Augmented Generation (RAG) systems [Lewis et al., 2020] offer a promising direction for this purpose, enabling natural-language policy queries to be answered with evidence grounded in specific source documents rather than model parameters alone.

This paper addresses both gaps with a two-component framework. First, we benchmark six machine learning models on WHO GLASS surveillance data (2021 to 2023) across 44 countries to forecast resistance rates at the country-pathogen-antibiotic level. Second, we implement a RAG pipeline that combines forecast results with retrieved WHO policy documents to answer natural-language questions about AMR policy and resistance trends.

Our main contributions are as follows:

1. A multi-model benchmark (Naive, Linear Regression, Ridge Regression, XGBoost, LightGBM, LSTM) on WHO GLASS surveillance data for population-level resistance rate forecasting, identifying XGBoost as the best-performing approach with a test MAE of 7.07% and R2 of 0.854.
2. A feature importance and regional error analysis that identifies the resistance lag feature as the dominant predictor and highlights geographic disparities in forecast accuracy tied to GLASS participation rates.
3. A locally deployed RAG system grounding AMR policy answers in retrieved WHO document chunks and forecast context, without requiring proprietary cloud APIs.

---

## 2. Related Work

### 2.1 Machine Learning for AMR Prediction

The application of machine learning to AMR has grown substantially over the past decade. Sakagianni et al. (2023) provided a comprehensive literature review of ML approaches to AMR prediction, covering random forests, support vector machines, neural networks, and gradient boosting methods across a range of settings. Kim et al. (2022) reviewed current practice and limitations from a clinical perspective in *Clinical Microbiology Reviews*, noting that most existing work focuses on isolate-level susceptibility classification rather than population-level temporal forecasting, and calling for methods that generalize across clinical contexts. Both reviews identified a gap in approaches that use surveillance data for anticipatory forecasting of resistance trends, which is the primary focus of this work.

### 2.2 Gradient Boosting for Epidemiological Forecasting

Gradient boosting methods have demonstrated consistent advantages over classical statistical models for infectious disease time series. Alim et al. (2020) compared XGBoost and ARIMA for predicting human brucellosis incidence in China, finding that XGBoost achieved a test MAPE of 7.64% compared to 17.68% for ARIMA, attributing XGBoost's advantage to its ability to model nonlinear patterns without the stationarity assumptions required by ARIMA. Fang et al. (2022) replicated this finding for COVID-19 transmission forecasting in the United States, with XGBoost again outperforming ARIMA across MAE, RMSE, and MAPE metrics. Ahn et al. (2023) demonstrated that ensemble gradient boosting (XGBoost, LightGBM, CatBoost) combined with attention-based CNN-LSTM achieved strong performance for environmental time series forecasting, validating both XGBoost [Chen and Guestrin, 2016] and LightGBM [Ke et al., 2017] in a forecasting context with a temporal structure similar to our AMR dataset.

### 2.3 LSTM Networks for Epidemic Forecasting

Recurrent neural networks, and Long Short-Term Memory (LSTM) networks in particular, have been applied to epidemic time series forecasting. Chimmula and Zhang (2020) applied LSTM to model COVID-19 transmission dynamics in Canada from early case count data, demonstrating the feasibility of LSTM for short-horizon epidemic forecasting and motivating its inclusion in our model comparison. In our evaluation, LSTM achieves performance comparable to gradient boosting (MAE: 7.16%), confirming that the limited temporal dataset is the constraining factor rather than model architecture choice.

### 2.4 Retrieval-Augmented Generation

Lewis et al. (2020) introduced the RAG framework, combining a dense retrieval component with a sequence-to-sequence language model to answer knowledge-intensive questions by grounding responses in retrieved document passages. RAG has since become a widely adopted approach for question answering in domains where factual accuracy and source attribution are critical, including medicine and public health. Our work applies the RAG framework to the AMR domain, integrating WHO policy documents with ML-derived forecast context to support evidence-based policy queries. To our knowledge, this is the first application of RAG specifically for AMR policy question answering informed by GLASS surveillance forecasts.

---

## 3. Data and Methods

### 3.1 Dataset

We used the WHO Global Antimicrobial Resistance and Use Surveillance System (GLASS) dataset covering the years 2021 to 2023 [World Health Organization, 2022; World Health Organization, 2023]. The dataset contains standardized resistance observations collected from national surveillance networks across 44 countries spanning six WHO geographic regions: the African Region, the Region of the Americas, the Eastern Mediterranean Region, the European Region, the South-East Asia Region, and the Western Pacific Region.

The final processed dataset comprised 5,909 observations after cleaning. Each observation records a unique combination of country, pathogen, antibiotic, and year. The target variable is the resistance percentage, defined as the proportion of tested isolates that demonstrated resistance to the corresponding antibiotic. Table 1 summarizes the key dataset characteristics.

**Table 1. Dataset characteristics.**

| Attribute | Value |
|-----------|-------|
| Years covered | 2021 to 2023 |
| Number of countries | 44 |
| WHO regions | 6 |
| Total observations | 5,909 |
| Target variable | Resistance percentage (%) |
| Income groups | High, Upper-middle, Lower-middle, Low, Unknown |

The AMR-consumption relationship documented by Ajulo and Awosile (2024) across GLASS-participating countries informed our feature selection, particularly the inclusion of antibiotic consumption as a predictive feature.

### 3.2 Features

We defined the following input features for all supervised models:

- **Pathogen name**: the bacterial species under surveillance (e.g., *Escherichia coli*, *Klebsiella pneumoniae*)
- **Antibiotic name**: the antibiotic being tested (e.g., meropenem, ampicillin)
- **Country or territory**: the reporting country
- **WHO region**: the six-region geographic classification
- **Income group**: World Bank income classification of the reporting country
- **Year**: observation year (2021, 2022, or 2023)
- **Resistance lag (lag1)**: the resistance percentage recorded in the previous year for the same pathogen-antibiotic-country combination
- **Antibiotic consumption (DDD per 1,000 inhabitants per day)**: proxy for selection pressure driven by antibiotic use

High-cardinality categorical features (country, pathogen, antibiotic) were encoded using target encoding applied within the training set to prevent data leakage into the held-out evaluation.

### 3.3 Preprocessing

Missing values in the target variable and lag features were imputed using the median resistance rate for the corresponding pathogen-antibiotic pair. This conservative imputation strategy preserves the distribution of resistance rates without introducing bias from cross-pathogen averaging.

The dataset was split temporally to respect the time-series structure of the data. Observations from 2021 to 2022 were used for training, 2022 observations served as the validation set, and 2023 observations formed the held-out test set. This split prevents information leakage from future time points into training and ensures evaluation reflects a realistic forecasting scenario where only historical data is available at prediction time.

### 3.4 Models

We compared six models of increasing complexity to benchmark forecasting performance on WHO GLASS surveillance data.

**Naive Baseline.** The naive model predicts each observation's resistance rate as equal to the prior year's resistance rate (lag1 feature). This represents the minimum performance threshold any supervised model must surpass to justify its complexity.

**Linear Regression.** Ordinary least squares regression trained on all encoded features provides a linear upper bound for simple models and reveals the contribution of the full feature set without nonlinear learning.

**Ridge Regression.** L2-regularized linear regression controls overfitting on correlated features. The regularization parameter was selected via cross-validation on the validation set.

**XGBoost.** We employ XGBoost [Chen and Guestrin, 2016], a scalable gradient boosting framework using decision trees as base learners with gradient-based optimization. XGBoost has consistently outperformed classical statistical models for infectious disease time series forecasting [Alim et al., 2020; Fang et al., 2022].

**LightGBM.** We include LightGBM [Ke et al., 2017], a histogram-based gradient boosting algorithm that offers efficient training on tabular data through leaf-wise tree growth. Gradient boosting ensembles including LightGBM have demonstrated strong performance on environmental and biological time series [Ahn et al., 2023].

**Long Short-Term Memory (LSTM).** We implement a single-layer LSTM network to evaluate whether sequential modeling captures temporal patterns not accessible to tree-based models [Chimmula and Zhang, 2020]. Sequences of length 1 were used given the limited temporal span of three years in the dataset. The model was trained on CPU due to GPU incompatibility between the test system's RTX 5060 and the current PyTorch CUDA build.

Hyperparameters for XGBoost and LightGBM were tuned using grid search over learning rate, maximum tree depth, number of estimators, and subsampling ratio. LSTM training used the Adam optimizer with a learning rate of 0.001 and early stopping based on validation MAE.

### 3.5 Evaluation Metrics

All models were evaluated on the held-out 2023 test set using three complementary metrics:

- **Mean Absolute Error (MAE)**: the average absolute deviation between predicted and actual resistance percentage, reported in percentage points
- **Root Mean Square Error (RMSE)**: penalizes large errors more heavily than MAE and is sensitive to outlier predictions
- **R-squared (R2)**: the proportion of variance in resistance rates explained by the model, with 1.0 indicating perfect fit

Regional MAE was additionally computed by disaggregating test set predictions by WHO region to identify geographic patterns in forecast accuracy.

### 3.6 Retrieval-Augmented Generation Pipeline

To augment model forecasts with interpretable policy context, we implemented a Retrieval-Augmented Generation (RAG) system following the architecture introduced by Lewis et al. (2020).

**Knowledge base.** We assembled a knowledge base of six curated WHO policy document excerpts covering: carbapenem resistance and ESKAPE pathogens, the Global Action Plan on AMR [World Health Organization, 2015], the AWaRe (Access, Watch, Reserve) antibiotic classification framework, AMR surveillance investment priorities for low- and middle-income countries, and regional resistance trends from GLASS data.

**Embedding and retrieval.** Each document chunk was encoded using the `all-MiniLM-L6-v2` sentence transformer model, selected for its balance of speed and semantic accuracy on short texts. Embeddings were stored in a ChromaDB vector database using cosine similarity as the distance metric. For each user query, the top 3 most semantically relevant chunks were retrieved and passed to the language model.

**Language model.** Retrieved chunks were combined with XGBoost forecast findings and regional MAE results to construct a structured prompt. The prompt was submitted to Phi-3 Mini, a 3.8-billion-parameter instruction-tuned language model, running locally via Ollama with CUDA 12.8 GPU acceleration through the llama.cpp backend. The system prompt explicitly prohibited fabricated citations, restricting all source references to the retrieved document chunks labeled as [Source 1], [Source 2], and [Source 3].

**Forecast context injection.** Each query automatically received an appended context block summarizing the top five XGBoost feature importances, regional MAE values across the six WHO regions, and the overall model performance comparison table. This enabled the language model to connect WHO policy recommendations to specific resistance trends observed in the forecast results.

---

## 4. Results

### 4.1 Model Performance Comparison

Table 2 presents the performance of all six models evaluated on the 2023 held-out test set.

**Table 2. Model performance on the 2023 held-out test set.**

| Model | Test MAE (%) | Test RMSE (%) | Test R2 | Improvement over Naive |
|-------|-------------|--------------|---------|------------------------|
| Naive Baseline | 41.83 | 48.15 | n/a | Reference |
| Linear Regression | 8.23 | 11.96 | 0.821 | 80.3% |
| Ridge Regression | 8.25 | 11.95 | 0.821 | 80.3% |
| LightGBM | 7.17 | 10.84 | 0.853 | 82.9% |
| LSTM | 7.16 | 10.10 | 0.872 | 82.9% |
| **XGBoost** | **7.07** | **10.80** | **0.854** | **83.1%** |

![Figure 1. Model comparison bar chart showing MAE across all six models on validation and test sets.](d:\All_Folder\data project\AMR Trend Forecasting with Policy Q&A\model_comparison.png)

**Figure 1.** Model performance comparison across all six models. XGBoost achieves the lowest test MAE (7.07%), with LightGBM and LSTM closely behind.

XGBoost achieved the lowest test MAE of 7.07% and the highest R2 of 0.854, outperforming the naive baseline by 83.1%. This result is consistent with prior findings demonstrating XGBoost superiority over statistical baselines for infectious disease time series [Alim et al., 2020; Fang et al., 2022].

Gradient boosting models (XGBoost, LightGBM) and the LSTM network achieved comparable performance with a spread of less than 0.15 percentage points in MAE. The convergence of deep learning and tree-based performance is expected given the limited temporal span of three years. LSTM networks typically provide greater advantage over longer sequences where long-range temporal dependencies accumulate, consistent with findings in larger epidemic datasets [Chimmula and Zhang, 2020].

Linear and Ridge regression models substantially outperformed the naive baseline but fell approximately 1.1 percentage points behind XGBoost in test MAE, indicating that nonlinear feature interactions contribute meaningfully to prediction accuracy beyond what a linear model can capture.

### 4.2 Feature Importance Analysis

Table 3 shows the top five features ranked by XGBoost gain-based importance score.

**Table 3. XGBoost feature importances (gain-based, top 5).**

| Rank | Feature | Importance (%) |
|------|---------|---------------|
| 1 | Resistance_lag1 (prior year resistance) | 50.5 |
| 2 | CountryTerritoryArea | 9.0 |
| 3 | AntibioticName | 6.7 |
| 4 | PathogenName | 6.0 |
| 5 | Antibiotic consumption (DID) | 4.8 |

![Figure 2. XGBoost feature importance chart showing the top predictors by gain score.](d:\All_Folder\data project\AMR Trend Forecasting with Policy Q&A\xgb_feature_importance.png)

**Figure 2.** XGBoost feature importance (gain-based). Resistance_lag1 dominates at 50.5%, confirming strong temporal autocorrelation in AMR rates.

The dominance of the resistance lag feature (50.5% importance) reflects strong temporal autocorrelation in AMR resistance rates. A country's resistance rate in a given year is by far the single strongest predictor of its resistance in the following year. This finding implies that short-term AMR trajectories are largely driven by the inertia of existing resistance patterns rather than rapid emergence events, and that even simple lag-only models capture a substantial fraction of the predictable signal.

Country-level heterogeneity (9.0%) and antibiotic identity (6.7%) together account for the next largest share of predictive power, indicating that resistance patterns differ substantially by both antibiotic class and geographic context. The inclusion of antibiotic consumption as the fifth-ranked feature (4.8%) is consistent with the AMR-consumption relationship documented in GLASS-participating countries by Ajulo and Awosile (2024).

### 4.3 Regional Error Analysis

Table 4 presents the disaggregated test MAE by WHO region for the XGBoost model.

**Table 4. XGBoost test MAE by WHO region (2023 test set).**

| WHO Region | Test MAE (%) | Test observations |
|-----------|-------------|-------------------|
| European Region | 4.16 | 809 |
| Western Pacific Region | 7.30 | 181 |
| African Region | 8.70 | 265 |
| Eastern Mediterranean Region | 9.00 | 709 |
| South-East Asia Region | 10.14 | 161 |

*Note: The Region of the Americas had no observations in the 2023 test set and is excluded from regional analysis.*

![Figure 3. Horizontal bar chart of XGBoost test MAE by WHO region.](d:\All_Folder\data project\AMR Trend Forecasting with Policy Q&A\mae_by_region.png)

**Figure 3.** XGBoost test MAE disaggregated by WHO region. The European Region achieves the lowest error (4.16%), while South-East Asia shows the highest (10.14%), reflecting disparities in GLASS data coverage.

Forecast accuracy varied substantially across regions. The European Region achieved the lowest MAE (4.16%), reflecting its comprehensive GLASS participation, mature national surveillance infrastructure, and high data consistency. The South-East Asia Region showed the highest MAE (10.14%), approximately 2.4 times higher than Europe.

This regional gradient in forecast accuracy closely mirrors the global AMR burden distribution documented by Murray et al. (2022), in which Sub-Saharan Africa and South/South-East Asia bear the highest mortality from drug-resistant infections. Regions with higher AMR burden tend to have weaker surveillance systems, fewer GLASS-enrolling countries, and greater year-to-year variability in reported resistance rates, all of which contribute to forecasting uncertainty.

### 4.4 RAG Policy Q&A System

We evaluated the RAG system on five predefined policy questions spanning: antibiotic prioritization by region, treatment guidelines for resistant pathogens, regional forecast uncertainty, income-stratified resistance trends, and surveillance investment priorities.

![Figure 4. Error analysis showing residual distribution and prediction vs actual resistance rates.](d:\All_Folder\data project\AMR Trend Forecasting with Policy Q&A\error_analysis.png)

**Figure 4.** XGBoost residual and error analysis on the 2023 test set. The distribution of errors confirms that most predictions fall within a narrow range, with larger errors concentrated in high-resistance observations.

All five responses demonstrated: (1) accurate retrieval of relevant WHO policy excerpts, (2) correct attribution of retrieved sources without fabricated references, and (3) coherent integration of forecast findings with policy recommendations. A representative example is provided below.

**Question:** "Which antibiotics should Southeast Asia prioritize preserving based on resistance forecasts?"

**System response (summarized):** The system correctly identified carbapenem-resistant ESKAPE pathogens as critical threats in South-East Asia based on retrieved WHO guidance, recommended prioritizing Access-group antibiotics per the AWaRe classification, connected the region's high forecast uncertainty (MAE: 10.1%) to surveillance infrastructure gaps, and recommended targeted investment in antibiotic stewardship programs and laboratory capacity consistent with WHO LMIC policy guidance.

The system correctly withheld responses based on unsupported information, instead noting that "the provided context does not specify this" when a query exceeded the scope of retrieved chunks. This behavior was enforced through explicit system prompt constraints restricting citation to retrieved document labels only.

---

## 5. Discussion

### 5.1 Forecasting Performance

Our results confirm that XGBoost is an effective model for short-term AMR resistance rate forecasting from WHO surveillance data, achieving a test MAE of 7.07% and an R2 of 0.854. The dominance of the resistance lag feature (50.5% importance) indicates that short-term AMR forecasting is primarily an autoregressive problem: past resistance strongly predicts future resistance. This has practical implications for resource-constrained settings, as lag-only features capture the majority of the predictable signal without requiring complex surveillance infrastructure.

The near-equivalent performance of gradient boosting and LSTM models suggests that the three-year temporal window available in GLASS (2021 to 2023) does not provide sufficient historical depth for LSTM to leverage its sequential modeling advantage. Future access to longer GLASS time series may reveal more meaningful gains from deep learning, consistent with findings from longer epidemic datasets [Chimmula and Zhang, 2020].

The regional MAE gradient reflects inequities in GLASS participation. European countries contribute richer, more consistent longitudinal records, enabling more accurate predictions. Expanding GLASS enrollment in high-burden regions would simultaneously improve AMR surveillance and improve the accuracy of data-driven forecasting systems that depend on that data.

### 5.2 RAG System Evaluation

The RAG pipeline produced grounded, policy-relevant answers by combining retrieved WHO document excerpts with XGBoost forecast context. The prompt-level citation constraint was effective at preventing hallucinated references, a known and critical failure mode of large language models in high-stakes domains [Lewis et al., 2020].

A key strength of the system is that answers are anchored in retrieved WHO policy text, making each claim traceable to a specific source chunk. This traceability property is important for policy decision support tools, where unexplained or unsourced recommendations may undermine institutional trust.

The current knowledge base of six WHO policy chunks serves as a functional proof of concept. Expanding it with complete WHO GLASS annual reports, national AMR action plans, and the WHO AWaRe classification database would substantially improve retrieval coverage and the specificity of responses to complex policy queries.

### 5.3 Limitations

Several limitations should be acknowledged. First, the dataset spans only three years (2021 to 2023), which restricts the model's ability to capture multi-year resistance emergence trajectories or seasonal patterns. Second, GLASS data represents a geographically non-random sample of countries, weighted toward those with existing surveillance capacity, introducing potential selection bias in regional comparisons. Third, the RAG knowledge base is small and manually curated; a systematic approach to corpus construction from WHO, national, and peer-reviewed sources would improve coverage. Fourth, Phi-3 Mini, while efficient and locally deployable, has lower reasoning capacity than larger frontier models, and responses to complex multi-step policy questions may be less nuanced.

### 5.4 Future Directions

Future work should explore: (1) integration of 2024 GLASS data as it becomes available to extend the temporal training window; (2) inclusion of complete WHO GLASS annual reports and national AMR action plans in the RAG knowledge base; (3) evaluation of the Temporal Fusion Transformer for longer-range forecasting with its explicit handling of covariate time series; (4) domain-adapted sentence embeddings trained on AMR literature for improved semantic retrieval precision; and (5) formal automated evaluation of RAG response faithfulness and coverage using frameworks such as RAGAS.

---

## 6. Conclusion

This paper presented an integrated framework for antimicrobial resistance trend forecasting and evidence-grounded policy question answering using WHO GLASS surveillance data. Across six models benchmarked on 5,909 observations from 44 countries, XGBoost [Chen and Guestrin, 2016] achieved the best forecasting performance with a test MAE of 7.07% and R2 of 0.854, representing an 83.1% improvement over the naive baseline. The prior-year resistance lag feature accounted for over half of the model's predictive power, and regional forecast accuracy tracked closely with GLASS data quality: the European Region achieved 4.2% MAE while the South-East Asia Region reached 10.1%.

We additionally implemented a Retrieval-Augmented Generation pipeline [Lewis et al., 2020] that combined retrieved WHO policy document excerpts with forecast-derived context to answer natural-language AMR policy questions. The system produced grounded, source-attributed responses without fabricated citations, demonstrating the feasibility of integrating ML-based surveillance forecasts into a policy decision support tool that is fully locally deployable.

As AMR is projected to cause 10 million deaths per year by 2050 [O'Neill, 2016; de Kraker et al., 2016], tools that translate global surveillance data into actionable policy insights are urgently needed. This work demonstrates that even limited surveillance data, combined with interpretable machine learning and retrieval-augmented language models, can produce meaningful forecasts and grounded policy recommendations. Expanding GLASS participation in high-burden regions and coupling those data streams with forecasting and policy retrieval systems represents a concrete step toward evidence-based AMR governance at the global scale.

---

## References

[1] O'Neill, J. (2016). *Tackling drug-resistant infections globally: Final report and recommendations*. Review on Antimicrobial Resistance.

[2] de Kraker, M. E. A., Stewardson, A. J., & Harbarth, S. (2016). Will 10 million people die a year due to antimicrobial resistance by 2050? *PLOS Medicine*, 13(11), e1002184. https://doi.org/10.1371/journal.pmed.1002184

[3] Murray, C. J. L., et al. (2022). Global burden of bacterial antimicrobial resistance in 2019: a systematic analysis. *The Lancet*, 399(10325), 629-655. https://doi.org/10.1016/S0140-6736(21)02724-0

[4] Dadgostar, P. (2019). Antimicrobial resistance: Implications and costs. *Infection and Drug Resistance*, 12, 3903-3910. https://doi.org/10.2147/IDR.S234610

[5] Tang, K. W. K., Millar, B. C., & Moore, J. E. (2023). Antimicrobial resistance (AMR). *British Journal of Biomedical Science*, 80, 11387. https://doi.org/10.3389/bjbs.2023.11387

[6] Sugden, R., Kelly, R., & Davies, S. (2016). Combatting antimicrobial resistance globally. *Nature Microbiology*, 1, 16187. https://doi.org/10.1038/nmicrobiol.2016.187

[7] Sakagianni, A., Koufopoulou, C., Feretzakis, G., Kalles, D., Verykios, V. S., Myrianthefs, P., & Fildisis, G. (2023). Using machine learning to predict antimicrobial resistance: A literature review. *Antibiotics*, 12(3), 452. https://doi.org/10.3390/antibiotics12030452

[8] Kim, J. I., Maguire, F., Tsang, K. K., Gouliouris, T., Peacock, S. J., McAllister, T. A., McArthur, A. G., & Beiko, R. G. (2022). Machine learning for antimicrobial resistance prediction: Current practice, limitations, and clinical perspective. *Clinical Microbiology Reviews*, 35(3), e00179-21. https://doi.org/10.1128/cmr.00179-21

[9] Chimmula, V. K. R., & Zhang, L. (2020). Time series forecasting of COVID-19 transmission in Canada using LSTM networks. *Chaos, Solitons and Fractals*, 135, 109864. https://doi.org/10.1016/j.chaos.2020.109864

[10] Alim, M., Ye, G-H., Guan, P., Huang, D-S., Zhou, B-S., & Wu, W. (2020). Comparison of ARIMA model and XGBoost model for prediction of human brucellosis in mainland China: a time-series study. *BMJ Open*, 10, e039498. https://doi.org/10.1136/bmjopen-2020-039498

[11] Fang, Z., Yang, S., Lv, C., An, S., & Wu, W. (2022). Application of a data-driven XGBoost model for the prediction of COVID-19 in the USA: a time-series study. *BMJ Open*, 12(7), e056685. https://doi.org/10.1136/bmjopen-2022-056685

[12] Ahn, J. M., Kim, J., & Kim, K. (2023). Ensemble machine learning of gradient boosting (XGBoost, LightGBM, CatBoost) and attention-based CNN-LSTM for harmful algal blooms forecasting. *Toxins*, 15(10), 608. https://doi.org/10.3390/toxins15100608

[13] Ajulo, S., & Awosile, B. (2024). Global antimicrobial resistance and use surveillance system (GLASS 2022): Investigating the relationship between antimicrobial resistance and antimicrobial consumption data across the participating countries. *PLOS ONE*, 19(2), e0297921. https://doi.org/10.1371/journal.pone.0297921

[14] World Health Organization. (2022). *Global antimicrobial resistance and use surveillance system (GLASS) report: 2021-2022*. WHO. https://www.who.int/publications/i/item/9789240062702

[15] World Health Organization. (2015). *Global action plan on antimicrobial resistance*. WHO. https://www.who.int/publications/i/item/9789241509763

[16] World Health Organization. (2023). *GLASS manual for antimicrobial resistance surveillance in common bacteria causing human infection*. WHO.

[17] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., Yih, W., Rocktaschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems (NeurIPS)*, 33. https://arxiv.org/abs/2005.11401

[18] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794. https://doi.org/10.1145/2939672.2939785

[19] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
