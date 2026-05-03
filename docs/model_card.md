# Model Card — Adult Income Classifier

## Model Description

| Field | Value |
|---|---|
| **Model name** | adult_income_classifier |
| **Version** | See MLflow Model Registry |
| **Task** | Binary classification |
| **Output** | `>50K` or `<=50K` (annual income) |
| **Algorithm** | Best of: Logistic Regression, Random Forest, XGBoost (selected by F1) |
| **Framework** | scikit-learn / XGBoost |
| **Training date** | See MLflow run metadata |

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
| **Reference period** | ~1994 US Census data |

**Features used:** age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country.

---

## Evaluation Metrics

### Overall (on adult.test, 16,282 rows)

| Metric | Value |
|---|---|
| Accuracy | ~87% |
| F1 Score | ~0.73 |
| Precision | ~0.75 |
| Recall | ~0.71 |
| ROC-AUC | ~0.91 |

### Per-Subgroup (known disparities)

| Group | Approximate Positive Rate |
|---|---|
| Male | ~31% |
| Female | ~11% |
| White | ~26% |
| Black | ~13% |

> **Note:** The model inherits the historical biases present in 1994 Census data. Income disparity by sex and race is a data artefact, not a model design choice.

---

## Limitations

- **Temporal:** Trained on 1994 census data; income dynamics have changed substantially.
- **Geographic:** US-centric; not valid for other countries.
- **Feature bias:** `sex` and `race` are correlated with the target due to systemic inequality, not because they are causal predictors.
- **Class imbalance:** ~76% of samples are `<=50K`; minority class performance is lower.

---

## Ethical Considerations

This model **must not** be used for any real-world decisions about individuals. It contains known demographic biases (gender pay gap, racial income disparity) encoded from historical census data. Using it for hiring, lending, or benefits eligibility would constitute unlawful discrimination in most jurisdictions.

Teams using this model are responsible for conducting fairness audits (e.g., Equalized Odds, Demographic Parity) before any production deployment beyond the educational scope of this project.

---

## Caveats

- The model is part of an MLOps course project; accuracy was not the primary optimisation goal.
- Hyperparameter search was limited to `n_iter=10` for time constraints.
- Drift monitoring is simulated via subgroup filtering, not real temporal drift.
