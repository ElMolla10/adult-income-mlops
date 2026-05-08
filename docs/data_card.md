# Data Card — Adult Income Dataset

## Dataset Overview

| Field | Value |
|---|---|
| **Name** | Adult Income / Census Income |
| **Source** | UCI Machine Learning Repository |
| **URL** | https://archive.ics.uci.edu/dataset/2/adult |
| **License** | CC BY 4.0 |
| **Original donors** | Ronny Kohavi & Barry Becker, Data Mining and Visualization, Silicon Graphics |
| **Creation date** | 1996 (based on 1994 US Census Bureau data) |

---

## Schema

| Feature | Type | Description | Missing? |
|---|---|---|---|
| age | Integer | Age of individual | No |
| workclass | Categorical | Employment type (Private, Self-emp, Gov, etc.) | Yes (~6%) |
| fnlwgt | Integer | Census sampling weight | No |
| education | Categorical | Highest education level achieved | No |
| education-num | Integer | Education encoded as years (1–16) | No |
| marital-status | Categorical | Marital status | No |
| occupation | Categorical | Job category | Yes (~6%) |
| relationship | Categorical | Relationship to household head | No |
| race | Categorical | Race (White, Black, Asian-Pac-Islander, etc.) | No |
| sex | Categorical | Binary sex (Male / Female) | No |
| capital-gain | Integer | Capital gains income | No |
| capital-loss | Integer | Capital losses | No |
| hours-per-week | Integer | Average weekly work hours | No |
| native-country | Categorical | Country of origin | Yes (~2%) |
| **income** | **Target** | `<=50K` or `>50K` annual income | No |

---

## Splits

| Split | File | Rows | Purpose |
|---|---|---|---|
| Train (Reference) | `adult.data` | 32,561 | Model training + drift reference |
| Test | `adult.test` | 16,281 | Model evaluation + production simulation |
| Production (drifted) | `data/splits/production.csv` | 9,561 | Drift monitoring (Male + White subgroup filter) |

## Class Distribution

| Split | `<=50K` Count | `<=50K` Share | `>50K` Count | `>50K` Share |
|---|---:|---:|---:|---:|
| Train | 24,720 | 75.9190% | 7,841 | 24.0810% |
| Test | 12,435 | 76.3774% | 3,846 | 23.6226% |

---

## Preprocessing Decisions

1. **Missing values:** The original dataset encodes missing values as `"?"`. These are replaced with `NaN` on load and imputed using:
   - Numeric features: median imputation
   - Categorical features: most-frequent imputation
2. **Scaling:** All numeric features are standardised with `StandardScaler` (zero mean, unit variance).
3. **Encoding:** All categorical features are one-hot encoded with `handle_unknown='ignore'` to gracefully handle unseen categories at inference time.
4. **Label fix:** The `adult.test` file adds a trailing `.` to income labels (e.g. `<=50K.`). This is stripped on load.
5. **Target encoding:** `>50K` → 1, `<=50K` → 0 for model training.
6. **Class imbalance:** SMOTE is applied inside the model-selection pipeline during cross-validation, not before the CV split, so synthetic samples do not leak across folds.

---

## Drift Simulation

To demonstrate data drift for the monitoring component, the production set is created by filtering `adult.test` to only rows where `sex == "Male"` AND `race == "White"`. This creates measurable distribution shift on:

- `sex` — Female class disappears entirely
- `race` — Non-white classes disappear entirely
- Correlated features (`relationship`, `marital-status`, `occupation`) shift as well

This is documented and visible in `monitoring/evidently_reports/drift_report.html`.

---

## Known Biases

- **Gender bias:** Female individuals earn `>50K` at approximately one-third the rate of males in this dataset, reflecting real 1994 wage inequality.
- **Racial bias:** White individuals have a higher positive income rate than Black, Hispanic, and Asian-Pacific Islander individuals.
- **Survivorship / sampling:** `fnlwgt` represents census sampling weights, not uniformly sampled individuals.
- **Binary sex:** The dataset only includes binary Male/Female; non-binary individuals are not represented.

---

## Privacy & Licensing

- Data is fully anonymised; no individual can be re-identified from the published features.
- License: CC BY 4.0 — free to use for research and education with attribution.
- No GDPR or HIPAA considerations apply to this public dataset.
