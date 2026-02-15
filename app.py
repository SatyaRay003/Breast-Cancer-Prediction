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
st.title("Breast Cancer Prediction Application", text_alignment="center")
st.markdown("<center>Using the numerical features extracted from cell nucleus image, \
We can predict whether the tumor is benign or malignant.</center>", unsafe_allow_html=True)

st.markdown("---")

# Dataset Upload
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Upload Test Dataset:")
with col2:
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])


# Model Selection
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Choose Classification Model:")
with col2:
    model_option = st.selectbox(
        "Models",
        [
            "Logistic Regression",
            "Decision Tree Classifier",
            "K-Nearest Neighbor",
            "Naive Bayes (Gaussian)",
            "Random Forest (Ensemble)",
            "XGBoost (Ensemble)"
        ]
    )

# Process uploaded file
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
        col1, col2, col3 = st.columns([1, 1, 1], gap="large")

        # Evaluation Metrics
        with col1:
            st.header("Evaluation Metrics:")
            st.subheader(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
            st.subheader(f"AUC Score: {roc_auc_score(y_test, y_pred):.4f}")
            st.subheader(f"Precision: {precision_score(y_test, y_pred, average="weighted"):.4f}")
            st.subheader(f"Recall   : {recall_score(y_test, y_pred, average="weighted"):.4f}")
            st.subheader(f"F1-Score : {f1_score(y_test, y_pred, average="weighted"):.4f}")
            st.subheader(f"MCC Score: {matthews_corrcoef(y_test, y_pred):.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        with col2:
            st.header("Confusion Matrix:")
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
            disp.plot(ax=ax, cmap='Blues', colorbar=False)
            plt.tight_layout()
            st.pyplot(fig)

        # Classification Report
        with col3:
            st.header("Classification Report:")
            report = classification_report(y_test, y_pred)
            st.text(report)

    except Exception as error:
        logging.error(error)
        st.error("Sorry! Something went wrong!")
else:
    st.info("Please upload a CSV dataset to begin.")
