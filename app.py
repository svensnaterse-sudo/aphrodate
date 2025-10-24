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
    'age', 'attractive', 'intelligence', 'funny', 'ambition','exercise','art',
    'reading', 'movies', 'music', 'shopping', 'gender_male', 'race_Black/African American',
    'race_European/Caucasian-American', 'race_Latino/Hispanic American', 'race_Other'
]

# ----------------------------
# Sidebar: user inputs
# ----------------------------
st.sidebar.header("Set Input Features")

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
    # Race as single selectbox
    race_options = [
        "Black/African American",
        "European/Caucasian-American",
        "Latino/Hispanic American",
        "Other"
    ]
    selected_race = st.sidebar.selectbox("Desired race", race_options)
    # Age slider
    inputs["age"] = st.sidebar.slider("Desired age", 18, 50, 25)
        



    # Other numeric features (example range 0-10)
    numeric_features = ['attractive', 'intelligence', 'funny', 'ambition', 'sports', 'tvsports', 'exercise',
                        'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv',
                        'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga']
    for col in numeric_features:
        inputs[col] = st.sidebar.slider(col, 0, 10, 5)


    
    # Convert race to one-hot encoding for your model
    for race in race_options:
        inputs[f"race_{race}"] = 1 if race == selected_race else 0

    return pd.DataFrame([inputs])

# Create sliders and selectbox
input_df = user_input_features()


# Prediction happens only when button is pressed
if st.sidebar.button("Predict Match"):
    input_df_ordered = input_df[feature_columns]
    input_scaled = scaler.transform(input_df_ordered)
    
    selected_gender = input_df_ordered["gender_male"].iloc[0]
    X_train_filtered = X_train[X_train["gender_male"] != selected_gender]
    y_train_filtered = y_train[X_train["gender_male"] != selected_gender]
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
