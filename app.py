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
def load_model():
    knn_model = load("knn_model.joblib")
    scaler = load("scaler.joblib")
    return knn_model, scaler

knn_model, scaler = load_model()

# ----------------------------
# Define features
# ----------------------------
feature_columns = [
    "age", "attractive", "intelligence", "funny", "ambition", "ambtition_important", "art", "attractive_important", "clubbing", "concerts", "d_age", "dining", "exercising", 
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
