# app.py
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Aphrodate", page_icon="💘")
st.title("💘 Aphrodate")
st.markdown("Adjust the sliders in the sidebar to set input features, then see the closest matches.")

# ----------------------------
# Load model, scaler, and training data
# ----------------------------
@st.cache_resource
def load_model_and_data():
    knn_model = load("knn_model.joblib")
    scaler = load("scaler.joblib")
    X_train = load("X_train.joblib")
    y_train = load("y_train.joblib")
    return knn_model, scaler, X_train, y_train

knn_model, scaler, X_train, y_train = load_model_and_data()

# ----------------------------
# Feature columns (same as training)
# ----------------------------
feature_columns = X_train.columns.tolist()

# ----------------------------
# Sidebar: user inputs
# ----------------------------
st.sidebar.header("Set Input Features")

def user_input_features():
    inputs = {}
    # Gender first
    if "gender_male" in feature_columns:
        inputs["gender_male"] = st.sidebar.selectbox(
            "Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female"
        )
    # Age first
    if "age" in feature_columns:
        inputs["age"] = st.sidebar.slider("Age", 18, 50, 25)
    
    # Other features (0-10 scale)
    for col in feature_columns:
        if col in ["gender_male", "age"]:
            continue
        inputs[col] = st.sidebar.slider(col, 0, 10, 5)
    
    return pd.DataFrame([inputs])

input_df = user_input_features()

# ----------------------------
# Prediction button
# ----------------------------
if st.sidebar.button("Find Nearest Matches"):
    # Make sure input columns are in the correct order
    input_ordered = input_df[feature_columns]

    # Scale input
    input_scaled = scaler.transform(input_ordered)

    # ----------------------------
    # Find the 5 nearest neighbors
    # ----------------------------
    distances, indices = knn_model.kneighbors(input_scaled, n_neighbors=5)
    
    nearest_neighbors = X_train.iloc[indices[0]].copy()
    nearest_neighbors["match"] = y_train.iloc[indices[0]].values
    nearest_neighbors["distance"] = distances[0]

    st.subheader("5 Nearest Neighbors")
    st.dataframe(nearest_neighbors)

    # ----------------------------
    # Feature Visualization
    # ----------------------------
    st.subheader("Input Feature Values")
    fig, ax = plt.subplots(figsize=(8,4))
    input_ordered.T.plot(kind='bar', legend=False, ax=ax)
    ax.set_ylabel("Value")
    ax.set_xlabel("Feature")
    ax.set_title("Selected Feature Values")
    st.pyplot(fig)
