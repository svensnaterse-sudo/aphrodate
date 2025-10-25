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

# ----------------------------
# Load model & data
# ----------------------------
@st.cache_resource
def load_data():
    X_train = load("X_train.joblib")
    y_train = load("y_train.joblib")
    return X_train, y_train

X_train, y_train = load_data()

# ----------------------------
# Basic dataset info
# ----------------------------
st.subheader("📁 Dataset Overview")
st.write(f"**Total of people:** {len(X_train)}")
st.write(f"**Total features:** {X_train.shape[1]}")


# ----------------------------
# Summary stats
# ----------------------------
st.subheader("📈 Summary Statistics")
st.dataframe(X_train.describe().T)

# ----------------------------
# Feature exploration controls
# ----------------------------
st.sidebar.header("🔍 Explore Feature Distributions")

feature_cols = X_train.columns.tolist()

# Identify common feature types
gender_col = "gender_male" if "gender_male" in X_train.columns else None
race_cols = [c for c in X_train.columns if c.startswith("race_")]
numeric_cols = [
    c for c in feature_cols
    if c not in race_cols and c != gender_col and X_train[c].dtype in [float, int]
]

# ----------------------------
# Gender & Race distribution combined
# ----------------------------
st.subheader("🚻🌍 Demographic Distribution")

# Options to choose
race_or_gender = []
if gender_col:
    race_or_gender.append("Gender")
if race_cols:
    race_or_gender.append("Race")

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


# ----------------------------
# Numeric feature distributions
# ----------------------------
st.subheader("📊 Trait Distributions")
selected_trait = st.selectbox("Select a trait to visualize", numeric_cols)

fig, ax = plt.subplots()
sns.histplot(X_train[selected_trait], kde=True, color="orchid", ax=ax)
ax.set_title(f"Distribution of {selected_trait}")
ax.set_xlabel(selected_trait)
st.pyplot(fig)

# ----------------------------
# Correlation heatmap
# ----------------------------
st.subheader("🔥 Feature Correlations")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(X_train[numeric_cols].corr(), cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Heatmap of Traits")
st.pyplot(fig)

# ----------------------------
# Match outcome analysis
# ----------------------------
if y_train is not None and isinstance(y_train, pd.Series):
    st.subheader("💞 Match Outcome Analysis")
    Xy = X_train.copy()
    Xy["match"] = y_train

    # Average traits for successful vs unsuccessful matches
    avg_traits = Xy.groupby("match")[numeric_cols].mean().T
    fig, ax = plt.subplots(figsize=(8, 4))
    avg_traits.plot(kind="bar", ax=ax)
    ax.set_title("Average Trait Values by Match Outcome")
    ax.set_ylabel("Average Score")
    ax.legend(title="Match", labels=["No Match", "Match"])
    st.pyplot(fig)

st.markdown("---")
st.caption("💘 Aphrodate — Exploring the science of attraction, one profile at a time.")
