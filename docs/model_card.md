# Model Card — Adult Income Classifier

## Model Description

| Field | Value |
|---|---|
| **Model name** | adult_income_classifier |
| **Version** | MLflow Model Registry version 3, Production |
| **Task** | Binary classification |
| **Output** | `>50K` or `<=50K` (annual income) |
| **Algorithm** | XGBoost, selected by highest F1 among Logistic Regression, Random Forest, and XGBoost |
| **Framework** | scikit-learn / XGBoost |
| **Training date** | 2026-05-03 10:28:50 UTC |

---

## Intended Use

**Primary use case:** Predict whether an individual's annual income exceeds $50,000 based on demographic and employment attributes. Intended for educational and research demonstrations of MLOps pipelines.

**Out-of-scope uses:**
- Real hiring, credit, or lending decisions
- Any decision with legal or financial consequences for individuals
- Deployment without bias auditing on the target population

---

## Training Data

| Property | Value |
|---|---|
| **Dataset** | UCI Adult Income (Census Income) |
| **Source** | https://archive.ics.uci.edu/dataset/2/adult |
| **License** | CC BY 4.0 |
| **Training split** | `adult.data` — 32,561 rows |
| **Reference period** | 1994 US Census data |

**Features used:** age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country.

---

## Evaluation Metrics

### Overall (on adult.test, 16,281 rows)

| Metric | Value |
|---|---|
| Accuracy | 0.8515 |
| F1 Score | 0.7117 |
| Precision | 0.6573 |
| Recall | 0.7759 |
| ROC-AUC | 0.9204 |

### Per-Subgroup Metrics

Metrics below are computed with `classification_report` on `adult.test` filtered separately by `sex` and `race`. Precision, recall, and F1 are for the positive class (`>50K`).

| Attribute | Group | Rows | Positive Rate | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sex | Female | 5,421 | 0.1088 | 0.9290 | 0.6740 | 0.6729 | 0.6735 |
| sex | Male | 10,860 | 0.2998 | 0.8128 | 0.6548 | 0.7945 | 0.7179 |
| race | Amer-Indian-Eskimo | 159 | 0.1195 | 0.9119 | 0.6923 | 0.4737 | 0.5625 |
| race | Asian-Pac-Islander | 480 | 0.2771 | 0.8458 | 0.7323 | 0.6992 | 0.7154 |
| race | Black | 1,561 | 0.1147 | 0.9321 | 0.7355 | 0.6369 | 0.6826 |
| race | Other | 135 | 0.1852 | 0.8889 | 0.8125 | 0.5200 | 0.6341 |
| race | White | 13,946 | 0.2503 | 0.8416 | 0.6515 | 0.7894 | 0.7138 |

> **Note:** The model inherits the historical biases present in 1994 Census data. Income disparity by sex and race is a data artefact, not a model design choice.

---

## Limitations

- **Temporal:** Trained on 1994 census data; income dynamics have changed substantially.
- **Geographic:** US-centric; not valid for other countries.
- **Feature bias:** `sex` and `race` are correlated with the target due to systemic inequality, not because they are causal predictors.
- **Class imbalance:** 75.9190% of original training samples are `<=50K`; minority class performance is lower.

---

## Ethical Considerations

This model **must not** be used for any real-world decisions about individuals. It contains known demographic biases (gender pay gap, racial income disparity) encoded from historical census data. Using it for hiring, lending, or benefits eligibility would constitute unlawful discrimination in most jurisdictions.

Teams using this model are responsible for conducting fairness audits (e.g., Equalized Odds, Demographic Parity) before any production deployment beyond the educational scope of this project.

---

## Caveats

- The model is part of an MLOps course project; accuracy was not the primary optimisation goal.
- Hyperparameter search was limited to `n_iter=10` for time constraints.
- Drift monitoring is simulated via subgroup filtering, not real temporal drift.
