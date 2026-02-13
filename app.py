"""
Build the streamlit app for Breast Cancer Prediction
"""

# import required libraries
import logging
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay, classification_report)


# Read the Fit & Transformed Scaler
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# Read the trained models and create a model dictionary
model_map = {
    "Logistic Regression": pickle.load(open("model/Logistic Regression.pkl", "rb")),
    "Decision Tree Classifier": pickle.load(open("model/Decision Tree.pkl", "rb")),
    "K-Nearest Neighbor": pickle.load(open("model/K-Nearest Neighbor.pkl", "rb")),
    "Naive Bayes (Gaussian)": pickle.load(open("model/Gaussian Naive Bayes.pkl", "rb")),
    "Random Forest (Ensemble)": pickle.load(open("model/Random Forest.pkl", "rb")),
    "XGBoost (Ensemble)": pickle.load(open("model/XG Boost.pkl", "rb")),
}

# Page Config
st.set_page_config(page_title="Breast Cancer Prediction App", layout="wide")
st.title("Breast Cancer Prediction Application")
st.write("Using the numerical features extracted from cell nucleus image, \
We can predict whether the tumor is benign or malignant.")

st.markdown("---")

# Sidebar - Model Selection
st.sidebar.header("⚙️ Model Selection")
model_option = st.sidebar.selectbox(
    "Choose Classification Model",
    [
        "Logistic Regression",
        "Decision Tree Classifier",
        "K-Nearest Neighbor",
        "Naive Bayes (Gaussian)",
        "Random Forest (Ensemble)",
        "XGBoost (Ensemble)"
    ]
)

# Dataset Upload
st.subheader("📁 Upload Test Dataset (CSV Only)")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded test csv file
        data = pd.read_csv(uploaded_file)

        # Basic validation
        if data.shape[-1] < 2:
            st.error("Dataset must contain at least 1 feature column and 1 target column.")
            st.stop()

        # Extract the features and target attribute
        X_test = data.iloc[:, :-1]
        y_test = data.iloc[:, -1]

        # Scale the features
        X_test_scaled = ""
        try:
            X_test_scaled = scaler.transform(X_test)
        except ValueError as val_error:
            logging.error(f"{val_error} for Data with Shape:{data.shape}")
            st.error(f"{val_error.args[0]}")
            st.stop()

        # Make Prediction using selected Model
        selected_models = model_map[str(model_option)]
        if model_option in ["Logistic Regression", "K-Nearest Neighbor"]:
            y_pred = selected_models.predict(X_test_scaled)
        else:
            y_pred = selected_models.predict(X_test)

        st.markdown("---")
        st.subheader(f"📈 Selected Model: {model_option}")

        # Evaluation Metrics
        st.markdown("---")
        st.subheader("📈 Evaluation Metrics:")

        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        mcc_score = matthews_corrcoef(y_test, y_pred)

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC Score", f"{auc_score:.4f}")
        col3.metric("Precision", f"{precision:.4f}")
        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1-Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc_score:.4f}")

        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1], gap="large")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        with col1:
            st.subheader("🔎 Confusion Matrix:")
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
            disp.plot(ax=ax, cmap='Blues', colorbar=False)
            plt.tight_layout()
            st.pyplot(fig)

        # Classification Report
        with col2:
            st.subheader("📄 Classification Report:")
            report = classification_report(y_test, y_pred)
            st.text(report)

    except Exception as error:
        logging.error(error)
        st.error("Sorry! Something went wrong!")
else:
    st.info("Please upload a CSV dataset to begin.")
