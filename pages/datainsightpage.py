import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Profile Insights", page_icon="📊")

st.title("📊 Profile Insights")
st.markdown("""
Explore general patterns and trends in the Aphrodate dataset.
""")

# Load model & data
@st.cache_resource
def load_data():
    X_train = load("X_train.joblib")
    y_train = load("y_train.joblib")
    return X_train, y_train

X_train, y_train = load_data()


st.write(f"**Total of people:** {len(X_train)}")
st.write(f"**Total features:** {X_train.shape[1]}")



st.subheader("📈 Summary Statistics")
st.dataframe(X_train.describe().T)



feature_cols = X_train.columns.tolist()

gender_col = "gender_male" if "gender_male" in X_train.columns else None
race_cols = [c for c in X_train.columns if c.startswith("race_")]
numeric_cols = [
    c for c in feature_cols
    if c not in race_cols and c != gender_col and X_train[c].dtype in [float, int]
]

st.subheader("Explore feature distributions using the sliders on your left")
if st.sidebar.button("Show query"):
    st.subheader("test")

# Gender & Race distribution combined
st.subheader("🚻🌍 Showcase statistics from either race or gender")

# Options to choose
race_or_gender = []
if race_cols:
    race_or_gender.append("Race")
if gender_col:
    race_or_gender.append("Gender")


selected_demo = st.selectbox("Select feature to visualize", race_or_gender)

fig, ax = plt.subplots()

if selected_demo == "Gender":
    gender_counts = X_train[gender_col].value_counts().rename({1: "Male", 0: "Female"})
    gender_counts.plot(kind="bar", color=["skyblue", "lightpink"], ax=ax)
    ax.set_ylabel("Count")
    ax.set_title("Gender Distribution")
elif selected_demo == "Race":
    race_counts = X_train[race_cols].sum().sort_values(ascending=False)
    race_counts.plot(kind="bar", ax=ax, color="lightgreen")
    ax.set_ylabel("Count")
    ax.set_title("Race Distribution")

st.pyplot(fig)

# Numeric feature distributions
st.subheader("📊 Trait Distributions")
selected_trait = st.selectbox("Select a trait to visualize", numeric_cols)

fig, ax = plt.subplots()
sns.histplot(X_train[selected_trait], color="blue", ax=ax)
ax.set_title(f"Distribution of {selected_trait}")
ax.set_xlabel(selected_trait)
st.pyplot(fig)


