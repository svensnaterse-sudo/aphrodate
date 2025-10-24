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
st.markdown("""
Adjust the sliders in the sidebar to set input features, then see your predicted match probability.
""")

# ----------------------------
# Load model and scaler
# ----------------------------
@st.cache_resource
def load_model_and_data():
    knn_model = load("knn_model.joblib (1)")
    scaler = load("scaler.joblib (1)")
    X_train = load("X_train.joblib")
    y_train = load("y_train.joblib")
    return knn_model, scaler, X_train, y_train

knn_model, scaler, X_train, y_train = load_model_and_data()

knn_model, scaler = load_model()

# ----------------------------
# Define features
# ----------------------------
feature_columns = [
    'age', 'd_age', 'samerace', 'importance_same_race',
    'attractive_important', 'intellicence_important', 'ambtition_important',
    'attractive', 'intelligence', 'funny', 'ambition',
    'shared_interests_partner', 'sports', 'tvsports', 'exercise', 'dining',
    'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv',
    'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga',
    'interests_correlate', 'gender_male',
    'race_Black/African American', 'race_European/Caucasian-American',
    'race_Latino/Hispanic American', 'race_Other',
    'race_o_Asian/Pacific Islander/Asian-American',
    'race_o_Black/African American', 'race_o_European/Caucasian-American',
    'race_o_Latino/Hispanic American', 'race_o_Other'

    # Add more features if your dataset has them
]

# ----------------------------
# Sidebar: user inputs
# ----------------------------
st.sidebar.header("Set Input Features")
def user_input_features():
    inputs = {}
    for col in feature_columns:
        # Use realistic ranges based on your dataset
        min_val = 0
        max_val = 10
        default = 5
        inputs[col] = st.sidebar.slider(col, min_val, max_val, default)
    return pd.DataFrame([inputs])

input_df = user_input_features()

# ----------------------------
# Display input data
# ----------------------------
st.subheader("Input Data")
st.write(input_df)

# ----------------------------
# Prediction
# ----------------------------
input_scaled = scaler.transform(input_df)
prediction = knn_model.predict(input_scaled)

st.subheader("Predicted Match Probability")
# ----------------------------
# Find the 3 nearest neighbors
# ----------------------------
# Get distances and indices of nearest neighbors
distances, indices = knn_model.kneighbors(input_scaled, n_neighbors=3)

# Convert indices to DataFrame rows
nearest_neighbors = X_train.iloc[indices[0]].copy()
nearest_neighbors["match"] = y_train.iloc[indices[0]].values
nearest_neighbors["distance"] = distances[0]

# Display
st.subheader("Nearest Neighbors")
st.write("These are the 3 closest matches to your input:")
st.dataframe(nearest_neighbors)

st.write("💖", np.round(prediction[0], 3))

# ----------------------------
# Feature Visualization
# ----------------------------
st.subheader("Feature Values")
fig, ax = plt.subplots(figsize=(8,4))
input_df.T.plot(kind='bar', legend=False, ax=ax)
ax.set_ylabel("Value")
ax.set_xlabel("Feature")
ax.set_title("Selected Feature Values")
st.pyplot(fig)
