# Census Income Analysis
### Fair Income Prediction for Financial Access

A responsible machine learning pipeline for binary income classification using the UCI Adult Census dataset. The project predicts whether an individual's annual income exceeds USD 50,000, with an explicit focus on predictive accuracy, fairness auditing, and model explainability.

---

## Features

- **Multi-model comparison** — Logistic Regression, Random Forest, and XGBoost trained and benchmarked with 5-fold stratified cross-validation
- **Threshold optimisation** — F1-maximising decision threshold sweep to improve minority-class recall beyond the default 0.5 cutoff
- **Fairness auditing** — Demographic Parity Difference and Equalized Odds Difference measured across sex and race using `fairlearn`
- **Bias mitigation** — Post-processing via `ThresholdOptimizer` (equalized odds constraint), achieving a 43% reduction in DPD and 71% in EOD for sex
- **SHAP explainability** — Global beeswarm/bar plots and local waterfall/dependence plots for per-prediction audit trails
- **Interactive dashboard** — 8-page Streamlit application that loads data directly from disk; no file upload required

---

## Setup Instructions

**Prerequisites:** Python 3.9+

1. **Clone the repository**
   ```bash
   git clone https://github.com/akritisshetty/census-income-analysis.git
   cd census-income-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   Place `adult.csv` (UCI Adult Census dataset) in the project root directory.

4. **Run the Jupyter notebook** (training & evaluation)
   ```bash
   jupyter notebook census_income_analysis.ipynb
   ```

5. **Launch the Streamlit dashboard**
   ```bash
   streamlit run app.py
   ```

> **Requirements:** `pandas==3.0.2`, `numpy==2.4.4`, `scikit-learn==1.8.0`, `xgboost`, `fairlearn==0.13.0`, `shap==0.51.0`, `matplotlib==3.10.8`, `seaborn==0.13.2`, `streamlit`

---

## Project Structure

```
census-income-analysis/
│
├── adult.csv                        # UCI Adult Census dataset (32,561 records)
├── census_income_analysis.ipynb     # End-to-end ML pipeline notebook (34 cells)
├── app.py                           # Streamlit 8-page dashboard
├── requirements.txt
└── README.md
```

---

## How It Works

The pipeline runs end-to-end through the following stages:

1. **Load & Sanitise** — Replace `"?"` entries with `NaN`; drop the `fnlwgt` sampling weight column
2. **EDA** — Explore feature distributions, missing values, and fairness baselines
3. **Preprocessing** — Engineer `capital.net` and `work_intensity` features; impute missing values; apply One-Hot Encoding and StandardScaler (for Logistic Regression only); perform an 80/20 stratified train-test split
4. **Model Training** — Train Logistic Regression, Random Forest, and XGBoost with 5-fold cross-validation; XGBoost achieves the best CV ROC-AUC of **0.9275** and test ROC-AUC of **0.9231**
5. **Threshold Optimisation** — Sweep thresholds from 0.1 to 0.9; optimal threshold of **0.43** improves minority-class F1 from 0.7124 → **0.7282**
6. **Fairness Audit** — Compute Demographic Parity Difference and Equalized Odds Difference via `fairlearn.MetricFrame` across sex and race groups
7. **Bias Mitigation** — Apply `ThresholdOptimizer` with equalized odds constraint; reduces sex-based DPD by **43%** and EOD by **71%**
8. **SHAP Analysis** — Identify top predictive features globally (age, marital status, capital gain, education, hours/week) and generate local per-prediction explanations
9. **Dashboard** — Serve all results through an interactive Streamlit application

---

## Technologies Used

| Purpose | Library |
|---|---|
| Data manipulation | pandas 3.0.2, numpy 2.4.4 |
| Visualisation | matplotlib 3.10.8, seaborn 0.13.2 |
| Machine learning | scikit-learn 1.8.0 |
| Gradient boosting | XGBoost |
| Fairness auditing & mitigation | fairlearn 0.13.0 |
| Explainability | SHAP 0.51.0 |
| Interactive dashboard | Streamlit |

---
