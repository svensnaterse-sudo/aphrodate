# app.py
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

# Page config
st.set_page_config(page_title="Aphrodate", page_icon="💕")
st.title("❤️ Aphrodate")
st.markdown(
    "Adjust the sliders in the sidebar to set input features, then see your nearest matches."
)

# Load model, scaler, and training data
@st.cache_resource
def load_model_and_data():
    knn_model = load("knn_model.joblib")
    scaler = load("scaler.joblib")
    X_train = load("X_train.joblib")
    y_train = load("y_train.joblib")
    return knn_model, scaler, X_train, y_train

knn_model, scaler, X_train, y_train = load_model_and_data()

# Define features (same order as training)
feature_columns = X_train.columns.tolist()

# Sidebar: user inputs
st.sidebar.header("Set Input Features")

def user_input_features():
    inputs = {}

    # Gender
    inputs["gender_male"] = st.sidebar.selectbox(
        "Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female"
    )

    # Race (one-hot encoding)
    race_cols = [col for col in feature_columns if col.startswith("race_")]
    race_options = [col.replace("race_", "") for col in race_cols]
    selected_race = st.sidebar.selectbox("Desired race", race_options)
    for race in race_options:
        inputs[f"race_{race}"] = 1 if race == selected_race else 0

    # Age
    inputs["age"] = st.sidebar.slider("Desired age", 18, 50, 25)

    # Other numeric features (0-10)
    numeric_features = [col for col in feature_columns if col not in ["age","gender_male"] and "race_" not in col]
    for col in numeric_features:
        inputs[col] = st.sidebar.slider(col, 0, 10, 5)

    inputs["standards"] = st.sidebar.slider("How high are your standards?", 0, 10, 6)

    return pd.DataFrame([inputs])

input_df = user_input_features()

# Prediction button
if st.sidebar.button("Predict Match"):
    # Match input columns
    input_df_ordered = input_df[feature_columns]

    # Scale input
    input_scaled = scaler.transform(input_df_ordered)

    # Filter training data to only include opposite gender
    user_gender = int(input_df["gender_male"].iloc[0])
    if "gender_male" in X_train.columns:
        opposite_gender_mask = X_train["gender_male"] != user_gender
        X_train_filtered = X_train[opposite_gender_mask]
        y_train_filtered = y_train[opposite_gender_mask]
    else:
        X_train_filtered = X_train
        y_train_filtered = y_train

    # Scale filtered training data
    X_train_scaled = scaler.transform(X_train_filtered)

    # Compute distances manually to avoid re-fitting model
    distances = pairwise_distances(input_scaled, X_train_scaled)[0]

    # Attach distances and match info
    X_train_filtered = X_train_filtered.copy()
    X_train_filtered["distance"] = distances
    X_train_filtered["match"] = y_train_filtered.values

    # Remove duplicate feature rows (keep closest)
    feature_only_cols = [c for c in X_train_filtered.columns if c not in ["match", "distance"]]
    X_train_filtered = X_train_filtered.sort_values("distance").drop_duplicates(
        subset=feature_only_cols, keep="first"
    )

    # Determine number of neighbors to show
    num_neighbors = min(5, X_train_filtered.shape[0])
    nearest_neighbors = X_train_filtered.nsmallest(num_neighbors, "distance").copy()

    # Get the user's standards slider value
    standards_value = input_df["standards"].iloc[0]

    # Add Match Status based on standards threshold
    nearest_neighbors["Match Status"] = nearest_neighbors["distance"].apply(
        lambda d: "❤️ Match" if d < standards_value else "💔 Not a match"
    )

    # Drop the original match column from display
    final_nearest_neighbors = nearest_neighbors.drop(columns=["match"])

    # Display nearest neighbors
    st.subheader(f"💘 Your {num_neighbors} best matches")
    st.dataframe(final_nearest_neighbors)

