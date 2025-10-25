# app.py
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt


st.set_page_config(page_title="Aphrodate", page_icon="💘")
st.title("💘 Aphrodate")
st.markdown("Adjust the sliders in the sidebar to set input features, then see your perfect match")


@st.cache_resource
def load_model_and_data():
    knn_model = load("knn_model.joblib")
    scaler = load("scaler.joblib")
    X_train = load("X_train.joblib")
    y_train = load("y_train.joblib")
    return knn_model, scaler, X_train, y_train

knn_model, scaler, X_train, y_train = load_model_and_data()


st.sidebar.header("Set Input Features")

def user_input_features():
    inputs = {}

    # Gender first
    inputs["gender_male"] = st.sidebar.selectbox(
        "Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female"
    )
    # Race as single selectbox
    race_options = [
        "European/Caucasian-American",
        "Black/African American",
        "Latino/Hispanic American",
        "Other"
    ]
    selected_race = st.sidebar.selectbox("Desired race", race_options)
    # Age slider
    inputs["age"] = st.sidebar.slider("Desired age", 18, 50, 25)

    # Other numeric features
    numeric_features = ['attractive', 'intelligence', 'funny', 'ambition', 'exercise',
                        'art', 'reading', 'movies','music', 'shopping']
    for col in numeric_features:
        inputs[col] = st.sidebar.slider(col, 0, 10, 5)

    for race in race_options:
        inputs[f"race_{race}"] = 1 if race == selected_race else 0

    return pd.DataFrame([inputs])

feature_columns = [
    'age', 'attractive', 'intelligence', 'funny', 'ambition','exercise','art',
    'reading', 'movies', 'music', 'shopping', 'gender_male', 'race_Black/African American',
    'race_European/Caucasian-American', 'race_Latino/Hispanic American', 'race_Other'
]



# Create sliders and selectbox
input_df = user_input_features()


# ----------------------------
# Prediction button
# ----------------------------
# Inside the Predict Match button block
# ----------------------------
# Prediction button
# ----------------------------
if st.sidebar.button("Predict Match"):
    # Ensure the input columns match the trained model
    input_df_ordered = input_df[feature_columns]

    # Scale input
    input_scaled = scaler.transform(input_df_ordered)

    # Nearest neighbors
    distances, indices = knn_model.kneighbors(input_scaled, n_neighbors=5)
    nearest_neighbors = X_train.iloc[indices[0]].copy()
    nearest_neighbors["match"] = y_train.iloc[indices[0]].values
    nearest_neighbors["distance"] = distances[0]

    st.subheader("Nearest Neighbors")
    st.write("These are the 5 closest matches to your input:")
    st.dataframe(nearest_neighbors)

    # Feature visualization
    st.subheader("Feature Values")
    fig, ax = plt.subplots(figsize=(8,4))
    input_df_ordered.T.plot(kind='bar', legend=False, ax=ax)
    ax.set_ylabel("Value")
    ax.set_xlabel("Feature")
    ax.set_title("Selected Feature Values")
    st.pyplot(fig)


