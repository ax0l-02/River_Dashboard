import streamlit as st
import pandas as pd

st.title("Model Training")

# ---------------------------------------------------------
# Load Model Results
# ---------------------------------------------------------

df = pd.read_excel("model_comparison.xlsx", usecols="A:K")

# Remove blank/header rows
df = df[df["Model"].notna()]
df = df[df["Model"] != "Model"]

# ---------------------------------------------------------
# Introduction
# ---------------------------------------------------------

st.write("""
Four machine learning regression models were developed to predict river discharge.
Each model was trained using the three preprocessed datasets (Hampel Filter,
First Difference + Reversal, and Rolling Median) to evaluate the effect of
different preprocessing techniques on prediction performance.

For the tree-based models (Random Forest, Gradient Boosting, and XGBoost),
multiple hyperparameter configurations were tested prior to optimization.
""")

st.divider()

# ---------------------------------------------------------
# Training Workflow
# ---------------------------------------------------------

st.subheader("Training Workflow")

st.code(
"""
Preprocessed Dataset
        │
        ▼
Model Training
        │
        ▼
Prediction
        │
        ▼
Performance Evaluation
""",
language="text"
)

st.divider()

# =========================================================
# Linear Regression
# =========================================================

with st.expander("Linear Regression", expanded=True):

    st.write("""
Linear Regression was used as the baseline model. Since it does not require
hyperparameter tuning, the default implementation was trained on each of the
three processed datasets.
""")

    st.dataframe(
        df[df["Model"] == "Linear Regression"],
        use_container_width=True
    )

# =========================================================
# Random Forest
# =========================================================

with st.expander("Random Forest"):

    st.write("""
Random Forest is an ensemble learning algorithm that combines multiple decision
trees to improve prediction accuracy and reduce overfitting. Six different
parameter configurations were evaluated across the three processed datasets.
""")

    st.subheader("Parameter Configurations")

    params = (
        df[df["Model"] == "Random Forest"]
        [["Run", "n_estimators", "max_depth",
          "min_samples_split", "min_samples_leaf"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    st.table(params)

    st.subheader("Training Results")

    st.dataframe(
        df[df["Model"] == "Random Forest"],
        use_container_width=True
    )

# =========================================================
# Gradient Boost
# =========================================================

with st.expander("Gradient Boost"):

    st.write("""
Gradient Boosting builds decision trees sequentially, with each new tree
attempting to correct the errors made by previous trees. Six parameter
configurations were evaluated on each processed dataset.
""")

    st.subheader("Parameter Configurations")

    params = (
        df[df["Model"] == "Gradient Boost"]
        [["Run", "n_estimators",
          "max_depth", "learning_rate"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    st.table(params)

    st.subheader("Training Results")

    st.dataframe(
        df[df["Model"] == "Gradient Boost"],
        use_container_width=True
    )

# =========================================================
# XGBoost
# =========================================================

with st.expander("XGBoost"):

    st.write("""
XGBoost is an optimized gradient boosting algorithm designed for high predictive
performance and computational efficiency. Six parameter configurations were
tested using each of the three processed datasets.
""")

    st.subheader("Parameter Configurations")

    params = (
        df[df["Model"] == "XGBoost"]
        [["Run", "n_estimators",
          "max_depth", "learning_rate"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    st.table(params)

    st.subheader("Training Results")

    st.dataframe(
        df[df["Model"] == "XGBoost"],
        use_container_width=True
    )

st.divider()

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

st.subheader("Training Summary")

summary = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Gradient Boost",
        "XGBoost"
    ],
    "Parameter Sets": [
        1,
        6,
        6,
        6
    ],
    "Datasets": [
        3,
        3,
        3,
        3
    ],
    "Total Models": [
        3,
        18,
        18,
        18
    ]
})

st.table(summary)

st.write(f"**Total model configurations evaluated:** {summary['Total Models'].sum()}")