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
st.markdown(
    "Adjust the sliders in the sidebar to set input features, then see your nearest matches."
)

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
feature_columns = X_train.columns.tolist()

# ----------------------------
# Sidebar: user inputs
# ----------------------------
st.sidebar.header("Set Input Features")

def user_input_features():
    inputs = {}

    # Gender
    inputs["gender_male"] = st.sidebar.selectbox(
        "Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female"
    )

    # Age
    inputs["age"] = st.sidebar.slider("Desired age", 18, 50, 25)

    # Other numeric features (0-10)
    numeric_features = [col for col in feature_columns if col not in ["age","gender_male"] and "race_" not in col]
    for col in numeric_features:
        inputs[col] = st.sidebar.slider(col, 0, 10, 5)

    # Race (one-hot encoding)
    race_cols = [col for col in feature_columns if col.startswith("race_")]
    race_options = [col.replace("race_", "") for col in race_cols]
    selected_race = st.sidebar.selectbox("Desired race", race_options)
    for race in race_options:
        inputs[f"race_{race}"] = 1 if race == selected_race else 0

    return pd.DataFrame([inputs])

input_df = user_input_features()

# ----------------------------
# Prediction button
# ----------------------------
if st.sidebar.button("Predict Match"):
    # Match input columns
    input_df_ordered = input_df[feature_columns]

    # Scale input
    input_scaled = scaler.transform(input_df_ordered)

    # Compute nearest neighbors
    distances, indices = knn_model.kneighbors(input_scaled, n_neighbors=5)
    nearest_neighbors = X_train.iloc[indices[0]].copy()
    nearest_neighbors["match"] = y_train.iloc[indices[0]].values
    nearest_neighbors["distance"] = distances[0]

    # Display nearest neighbors
    st.subheader("💘 Your 5 best matches")
    st.dataframe(nearest_neighbors)


    # Feature comparison chart
    st.subheader("🎨 Feature Values")
    fig, ax = plt.subplots(figsize=(8,4))
    input_df_ordered.T.plot(kind='bar', legend=False, ax=ax, color='lightcoral')
    ax.set_ylabel("Value")
    ax.set_xlabel("Feature")
    ax.set_title("Selected Feature Values")
    st.pyplot(fig)
