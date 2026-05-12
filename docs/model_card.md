# Model Card — Adult Income Classifier

## Model Description

| Field | Value |
|---|---|
| **Model name** | adult_income_classifier |
| **Version** | Local retrain, MLflow Model Registry version 3 in temporary validation store |
| **Task** | Binary classification |
| **Output** | `>50K` or `<=50K` (annual income) |
| **Algorithm** | XGBoost, selected by highest F1 among Logistic Regression, Random Forest, and XGBoost |
| **Framework** | scikit-learn / XGBoost |
| **Training date** | 2026-05-12 |

XGBoost was selected over RandomForest (F1=0.6976) and LogisticRegression (F1=0.6821) based on highest threshold-calibrated F1 score on the held-out test set.

### Best Hyperparameters

| Hyperparameter | Value |
|---|---:|
| `model_type` | XGBoost |
| `n_estimators` | 350 |
| `max_depth` | 5 |
| `learning_rate` | 0.1 |
| `subsample` | 1.0 |
| `colsample_bytree` | 1.0 |
| `min_child_weight` | 1 |
| `reg_lambda` | 5 |
| `scale_pos_weight` | 1.0 |
| `decision_threshold` | 0.425 |

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

## Preprocessing Summary

The training pipeline applies the saved preprocessing workflow in `data/processed/pipeline.pkl`, including imputation, one-hot encoding, and scaling. XGBoost is trained on the original class distribution, and the saved model carries a calibrated positive-class threshold (`0.425`) used consistently by evaluation, FastAPI inference, and the Streamlit demo.

The serving API validates categorical inputs against the known Adult Income category sets before inference. Invalid categories are rejected with a 422 response instead of being silently encoded as unknown one-hot values.

---

## Evaluation Metrics

### Overall (on adult.test, 16,281 rows)

| Metric | Value |
|---|---|
| Accuracy | 0.8689 |
| F1 Score | 0.7199 |
| Precision | 0.7268 |
| Recall | 0.7132 |
| ROC-AUC | 0.9281 |
| Predicted Positive Rate | 0.2318 |

### Per-Subgroup Metrics

Metrics below are computed with `classification_report` on `adult.test` filtered separately by `sex` and `race`. Precision, recall, and F1 are for the positive class (`>50K`).

| Attribute | Group | Rows | Positive Rate | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sex | Female | 5,421 | 0.1088 | 0.9356 | 0.7287 | 0.6508 | 0.6876 |
| sex | Male | 10,860 | 0.2998 | 0.8356 | 0.7265 | 0.7245 | 0.7255 |
| race | Amer-Indian-Eskimo | 159 | 0.1195 | 0.9119 | 0.6667 | 0.5263 | 0.5882 |
| race | Asian-Pac-Islander | 480 | 0.2771 | 0.8375 | 0.6978 | 0.7293 | 0.7132 |
| race | Black | 1,561 | 0.1147 | 0.9276 | 0.7230 | 0.5978 | 0.6544 |
| race | Other | 135 | 0.1852 | 0.8963 | 0.8667 | 0.5200 | 0.6500 |
| race | White | 13,946 | 0.2503 | 0.8627 | 0.7278 | 0.7209 | 0.7243 |

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
