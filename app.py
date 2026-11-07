from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

MODEL_LOCATION = Path(__file__).parent / "model"

MODELS = {
    "Logistic Regression": {
        "file": "heart_logReg_model.joblib",
        "script": "src/heart_disease_logreg.py",
    },
    "Decision Tree": {
        "file": "heart_tree_model.joblib",
        "script": "src/heart_disease_DecTree.py",
    },
    "K-Nearest Neighbors": {
        "file": "heart_knn_model.joblib",
        "script": "src/heart_disease_knn.py",
    },
    "Gaussian Naive Bayes": {
        "file": "heart_gnb_model.joblib",
        "script": "src/heart_diesease_NVg.py",
    },
    "Support Vector Machine": {
        "file": "heart_svm_model.joblib",
        "script": "src/heart_disease_svm.py",
    },
}

FEATURE_ORDER = [
    "Age", "Gender", "Weight", "Height", "BMI", "Smoking", "Alcohol_Intake",
    "Physical_Activity", "Diet", "Stress_Level", "Hypertension", "Diabetes",
    "Hyperlipidemia", "Family_History", "Previous_Heart_Attack", "Systolic_BP",
    "Diastolic_BP", "Heart_Rate", "Blood_Sugar_Fasting", "Cholesterol_Total",
]

TARGET = "Heart_Disease"
CLASS_NAMES = ["No disease", "Disease"]


@st.cache_resource
def load_model(file_name):
    return joblib.load(MODEL_LOCATION / file_name)


def model_exists(model_name):
    return (MODEL_LOCATION / MODELS[model_name]["file"]).exists()


def read_test_data(uploaded_file):
    df = pd.read_csv(uploaded_file, keep_default_na=False)

    missing = [c for c in FEATURE_ORDER + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError("The CSV is missing these columns: " + ", ".join(missing))
    if df.empty:
        raise ValueError("The CSV has no rows.")

    labels = pd.to_numeric(df[TARGET], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(f"{TARGET} must contain only 0 and 1.")
    df[TARGET] = labels.astype(int)
    return df


@st.cache_data(show_spinner=False)
def run_evaluation(model_file, df):
    model = load_model(model_file)
    X = df[FEATURE_ORDER]
    y = df[TARGET]

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC Score": roc_auc_score(y, y_prob) if y.nunique() == 2 else None,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1 Score": f1_score(y, y_pred, zero_division=0),
        "MCC Score": matthews_corrcoef(y, y_pred),
    }
    matrix = confusion_matrix(y, y_pred, labels=[0, 1])
    report = classification_report(
        y, y_pred, labels=[0, 1], target_names=CLASS_NAMES, zero_division=0
    )
    return metrics, matrix, report


def go_to(page):
    st.session_state.page = page


def show_welcome():
    st.markdown(
        "<h1 style='text-align:center;'>Heart Disease Risk Predictor</h1>",
        unsafe_allow_html=True,
    )
    st.write("")

    st.subheader("Choose a model")

    names = list(MODELS.keys())
    previous = st.session_state.get("model_name")
    st.selectbox(
        "Model", names,
        index=names.index(previous) if previous in names else None,
        placeholder="Select a model",
        key="chosen_model",
        label_visibility="collapsed",
    )

    if st.button("Get started", type="primary"):
        choice = st.session_state.get("chosen_model")
        if not choice:
            st.warning("Please select a model first.")
        elif not model_exists(choice):
            st.warning(
                f"The file **{MODELS[choice]['file']}** was not found. Run "
                f"`python {MODELS[choice]['script']}` once to create it, "
                f"then try again."
            )
        else:
            st.session_state.model_name = choice
            go_to("evaluate")
            st.rerun()


def show_evaluation(model_name, info):
    st.caption(
        "Upload a CSV of test data with the same columns as the training data, "
        f"including the {TARGET} column."
    )
    uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
    if uploaded is None:
        return

    try:
        df = read_test_data(uploaded)
        with st.spinner("Evaluating..."):
            metrics, matrix, report = run_evaluation(info["file"], df)
    except Exception as error:
        st.error(f"Could not evaluate this file: {error}")
        return

    st.success(f"Evaluated {len(df)} rows with {model_name}.")

    with st.expander("Preview of uploaded data"):
        st.dataframe(df.head(10))

    st.subheader("Evaluation metrics")
    items = list(metrics.items())
    for start in (0, 3):
        columns = st.columns(3)
        for column, (label, value) in zip(columns, items[start:start + 3]):
            column.metric(label, "N/A" if value is None else f"{value:.4f}")

    st.subheader("Confusion matrix")
    st.dataframe(pd.DataFrame(
        matrix,
        index=[f"Actual: {name}" for name in CLASS_NAMES],
        columns=[f"Predicted: {name}" for name in CLASS_NAMES],
    ))

    st.subheader("Classification report")
    st.code(report)


def show_model_page():
    model_name = st.session_state.get("model_name")
    if model_name is None:
        go_to("welcome")
        st.rerun()
        return

    info = MODELS[model_name]

    with st.sidebar:
        st.button("Change model", on_click=go_to, args=("welcome",))

    st.title("Evaluate model")
    st.caption(f"Model: **{model_name}**")

    if not model_exists(model_name):
        st.warning(
            f"The file **{info['file']}** was not found. Run "
            f"`python {info['script']}` once to create it, then reload this page."
        )
        return

    show_evaluation(model_name, info)


def main():
    st.set_page_config(page_title="Heart Disease Predictor")
    if "page" not in st.session_state:
        st.session_state.page = "welcome"

    if st.session_state.page == "welcome":
        show_welcome()
    else:
        show_model_page()


if __name__ == "__main__":
    main()