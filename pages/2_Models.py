import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Load dataset based on selection
def load_dataset(name):
    if name == "Iris (classification)":
        data = datasets.load_iris()
        X, y = data.data, data.target
    elif name == "Diabetes (regression)":
        data = datasets.load_diabetes()
        X, y = data.data, data.target
    elif name == "Wine (classification)":
        data = datasets.load_wine()
        X, y = data.data, data.target
    elif name == "Boston Housing (regression)":
        data = datasets.load_boston()
        X, y = data.data, data.target
    else:
        X, y = None, None
    return X, y

st.title("Model Evaluation App")

# User selections
model_type = st.radio("Model type", ["Classification", "Regression"])

if model_type == "Classification":
    model_option = st.selectbox("Select Model", ["Logistic Regression"])
else:
    model_option = st.selectbox("Select Model", ["Linear Regression"])
dataset_name = st.selectbox("Select Dataset", ["Iris (classification)", "Wine (classification)", "Diabetes (regression)", "Boston Housing (regression)"])


if st.button("Run Model"):
    X, y = load_dataset(dataset_name)
    if X is None:
        st.error("Failed to load dataset.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Initialize model
        if model_option == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_option == "Linear Regression":
            model = LinearRegression()
        else:
            st.error("Unknown model selected.")
            model = None
        
        if model:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Plotting
            fig, ax = plt.subplots()
            if "regression" in dataset_name.lower():
                ax.scatter(y_test, y_pred)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Actual vs. Predicted")
            else:
                # For classification, show scatter and confusion-like info
                ax.scatter(y_test, y_pred)
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Actual vs. Predicted")
            st.pyplot(fig)
            
            # Show metrics
            if "regression" in dataset_name.lower():
                mse = mean_squared_error(y_test, y_pred)
                st.write(f"Mean Squared Error: {mse:.2f}")
            else:
                accuracy = accuracy_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred, average='weighted')
                precision = precision_score(y_test, y_pred, average='weighted')
                st.write(f"Accuracy: {accuracy:.2f}")
                st.write(f"Recall: {recall:.2f}")
                st.write(f"Precision: {precision:.2f}")
