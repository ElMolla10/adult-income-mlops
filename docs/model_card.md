# Model Card — Adult Income Classifier

## Model Description

| Field | Value |
|---|---|
| **Model name** | adult_income_classifier |
| **Version** | Current MLflow Model Registry Production version |
| **Task** | Binary classification |
| **Output** | `>50K` or `<=50K` (annual income) |
| **Algorithm** | XGBoost, selected by highest F1 among Logistic Regression, Random Forest, and XGBoost |
| **Framework** | scikit-learn / XGBoost |
| **Training date** | 2026-05-08 13:01:25 UTC |

XGBoost was selected over RandomForest (F1=0.6892) and LogisticRegression (F1=0.6698) based on highest F1 score on the held-out test set.

### Best Hyperparameters

| Hyperparameter | Value |
|---|---:|
| `model_type` | XGBoost |
| `n_estimators` | 200 |
| `max_depth` | 3 |
| `learning_rate` | 0.3 |
| `subsample` | 0.8 |

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

The training pipeline applies the saved preprocessing workflow in `data/processed/pipeline.pkl`, including imputation, one-hot encoding, and scaling. SMOTE is applied inside the model-selection pipeline during `RandomizedSearchCV`, so synthetic samples are generated separately within each cross-validation fold. The same `pipeline.pkl` artifact is loaded during serving, eliminating train/serve skew between model training, FastAPI inference, and the Streamlit demo.

The serving API validates categorical inputs against the known Adult Income category sets before inference. Invalid categories are rejected with a 422 response instead of being silently encoded as unknown one-hot values.

---

## Evaluation Metrics

### Overall (on adult.test, 16,281 rows)

| Metric | Value |
|---|---|
| Accuracy | 0.8541 |
| F1 Score | 0.7097 |
| Precision | 0.6697 |
| Recall | 0.7548 |
| ROC-AUC | 0.9196 |

### Per-Subgroup Metrics

Metrics below are computed with `classification_report` on `adult.test` filtered separately by `sex` and `race`. Precision, recall, and F1 are for the positive class (`>50K`).

| Attribute | Group | Rows | Positive Rate | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sex | Female | 5,421 | 0.1088 | 0.9299 | 0.6895 | 0.6475 | 0.6678 |
| sex | Male | 10,860 | 0.2998 | 0.8163 | 0.6668 | 0.7743 | 0.7165 |
| race | Amer-Indian-Eskimo | 159 | 0.1195 | 0.9119 | 0.7778 | 0.3684 | 0.5000 |
| race | Asian-Pac-Islander | 480 | 0.2771 | 0.8354 | 0.7177 | 0.6692 | 0.6926 |
| race | Black | 1,561 | 0.1147 | 0.9263 | 0.7105 | 0.6034 | 0.6526 |
| race | Other | 135 | 0.1852 | 0.8963 | 0.9231 | 0.4800 | 0.6316 |
| race | White | 13,946 | 0.2503 | 0.8456 | 0.6656 | 0.7699 | 0.7140 |

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
