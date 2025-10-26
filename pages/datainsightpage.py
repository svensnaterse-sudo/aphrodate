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


st.sidebar.header("Explore feature distributions")

feature_cols = X_train.columns.tolist()
numeric_features = [col for col in feature_cols if col not in ["age","gender_male"] and "race_" not in col]
def user_input_features():
    inputs = {}

    # Gender
    inputs["gender_male"] = st.sidebar.selectbox(
        "Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female"
    )
    # Age
    with st.sidebar:
        show_age = st.checkbox("Age", value=True, help="Exlude from the query")
        if show_age:
            age = st.slider("", 18, 50, 25, key="age_slider")
        else:
            age = None

    # Other numeric features (0-10)
    

    for col in numeric_features:
        show_trait = st.sidebar.checkbox(col, value=True, help="Exclude from the query")
        if show_trait:
            inputs[col] = st.sidebar.slider(col, 0, 10, 5)
        else:
            inputs[col] = None
    


    return pd.DataFrame([inputs])

input_df = user_input_features()

st.subheader("📈 Summary Statistics")
st.dataframe(X_train.describe().T)


gender_col = "gender_male" if "gender_male" in X_train.columns else None
race_cols = [c for c in X_train.columns if c.startswith("race_")]
numeric_cols = [
    c for c in feature_cols
    if c not in race_cols and c != gender_col and X_train[c].dtype in [float, int]
]



st.subheader("Explore feature distributions using the sliders")
if st.button("Show query"):
    trait_counts= {}
    trait_value = input_df[col].iloc[0]
    if trait_value is not None:
        for col in numeric_features:
            count=(X_train[col]>=5).sum()
    else:
        count = None
    trait_counts[col] = count
    st.dataframe(pd.DataFrame.from_dict(trait_counts, orient="index", columns=["Count"]))
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


