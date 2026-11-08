# Heart Disease Prediction using Machine Learning

## a. Problem Statement

Heart disease is a leading cause of death worldwide, and spotting high-risk patients early allows timely treatment. This project is a **binary classification** problem: given a patient's demographic details, lifestyle habits, medical history and clinical measurements, predict whether the patient **has heart disease (1) or not (0)**.

Five machine learning models are trained and compared using Accuracy, AUC, Precision, Recall, F1 and MCC. A Streamlit app lets the user choose a model, upload test data (CSV), and view the evaluation metrics, confusion matrix and classification report.

## b. Dataset Description

| Item | Detail |
|---|---|
| Type | Synthetic tabular data (no real patients) |
| Size | 50,000 records, 20 input features + 1 target |
| Target | `Heart_Disease` (0 = no disease, 1 = disease) |
| Class balance | 53.65% no disease, 46.35% disease |
| Missing values | None (`None` in `Alcohol_Intake` is a valid category) |

**Features**

- **Numeric (9):** Age, Weight, Height, BMI, Systolic_BP, Diastolic_BP, Heart_Rate, Blood_Sugar_Fasting, Cholesterol_Total
- **Binary, 0/1 (5):** Hypertension, Diabetes, Hyperlipidemia, Family_History, Previous_Heart_Attack
- **Categorical (6):** Gender, Smoking, Alcohol_Intake, Physical_Activity, Diet, Stress_Level

**Preprocessing:** numeric columns scaled with `StandardScaler` (not needed for the Decision Tree), categorical columns one-hot encoded, 80/20 stratified train-test split (`random_state=42`).

## c. GitHub Repository

https://github.com/Abhinav128/Models-Using-Heart-Disease

```
app.py              Streamlit app
requirements.txt    Dependencies
test_data.csv       Data file
model/              Saved models (.joblib)
src/                Training scripts and evaluate.py
```

## d. Models Used

All metrics are on the 20% held-out test split.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9238 | 0.9822 | 0.9178 | 0.9178 | 0.9178 | 0.8468 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| KNN (k=21) | 0.9271 | 0.9830 | 0.9352 | 0.9055 | 0.9201 | 0.8535 |
| Naive Bayes (Gaussian) | 0.8359 | 0.9264 | 0.8472 | 0.7881 | 0.8166 | 0.6699 |
| Support Vector Machine | 0.9476 | 0.9913 | 0.9446 | 0.9422 | 0.9434 | 0.8946 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Solid baseline (92.4% accuracy, AUC 0.982) with balanced precision and recall. It draws a straight-line boundary, so it cannot capture the dataset's "combination of risk factors" pattern exactly. Fast and easy to interpret. |
| Decision Tree | Perfect scores on every metric. The labels in this synthetic dataset follow a threshold rule (at least 2 of 5 risk factors: Age > 55, Cholesterol > 240, Hypertension, Diabetes, Previous heart attack), which a tree learns exactly while ignoring the other columns. |
| KNN | Slightly better than Logistic Regression (92.7%), with higher precision (0.935) but lower recall (0.906), so it misses more sick patients. The irrelevant columns add noise to the distance calculation, and prediction is slower. |
| Naive Bayes | Weakest model (83.6% accuracy, MCC 0.670) and lowest recall (0.788). It assumes features are independent, but the label depends on combinations of features. Very fast to train. |
| Support Vector Machine | Best of the non-tree models (94.8% accuracy, AUC 0.991, MCC 0.895). The RBF kernel's curved boundary approximates the rule well. It is the slowest model to train. |

### Overall Winner

**Decision Tree** is the winner on this dataset, with 100% on every metric. This is a result of the synthetic, rule-based labels, so it should not be expected on real, noisy patient data. **SVM** is the strongest of the remaining models.