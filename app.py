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
st.markdown("Adjust the sliders in the sidebar to set input features, then see your predicted match probability.")

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
# Define features (same order as training)
# ----------------------------
feature_columns = [
    'age', 'attractive', 'intelligence', 'funny', 'ambition', 'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking',
    'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga', 'gender_male', 
]

# ----------------------------
# Sidebar: user inputs
# ----------------------------
st.sidebar.header("Set Input Features")

def user_input_features():
    inputs = {}
    # Gender first
    inputs["gender_male"] = st.sidebar.selectbox(
        "Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female"
    )

    # Feature ranges
    feature_ranges = {
        "age": (18, 50, 25),
        "d_age": (0, 30, 5),
        "samerace": (0, 1, 0)
    }

    # Remaining features
    for col in feature_columns:
        if col == "gender_male":
            continue
        if col in feature_ranges:
            min_val, max_val, default = feature_ranges[col]
        else:
            min_val, max_val, default = 0, 10, 5
        inputs[col] = st.sidebar.slider(col, min_val, max_val, default)

    return pd.DataFrame([inputs])

# Create sliders so they always show
# Always create sliders
input_df = user_input_features()

# Prediction happens only when button is pressed
if st.sidebar.button("Predict Match"):
    input_df_ordered = input_df[feature_columns]
    input_scaled = scaler.transform(input_df_ordered)
    
    # Nearest neighbors
    distances, indices = knn_model.kneighbors(input_scaled, n_neighbors=5)
    nearest_neighbors = X_train.iloc[indices[0]].copy()
    nearest_neighbors["match"] = y_train.iloc[indices[0]].values
    nearest_neighbors["distance"] = distances[0]

    st.subheader("5 Nearest Neighbors")
    st.dataframe(nearest_neighbors)

    # Feature visualization
    st.subheader("Feature Values")
    fig, ax = plt.subplots(figsize=(8,4))
    input_df_ordered.T.plot(kind='bar', legend=False, ax=ax)
    ax.set_ylabel("Value")
    ax.set_xlabel("Feature")
    ax.set_title("Selected Feature Values")
    st.pyplot(fig)
