import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Model Comparison")

# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------

results = pd.read_excel("model_comparison.xlsx", usecols="A:K")
results = results[results["Model"].notna()]
results = results[results["Model"] != "Model"]

best = pd.read_excel("model_comparison.xlsx", usecols="N:S")
best.columns = [
    "Model",
    "Best Dataset",
    "Best Parameters",
    "MAE",
    "RMSE",
    "R²"
]
best = best.dropna(how="all")

# ---------------------------------------------------------
# Introduction
# ---------------------------------------------------------

st.write("""
The performance of each machine learning model was evaluated using three
standard regression metrics:

- **MAE (Mean Absolute Error):** Lower values indicate better prediction accuracy.
- **RMSE (Root Mean Squared Error):** Lower values indicate fewer large prediction errors.
- **R² Score:** Higher values indicate better model fit.
""")

st.divider()

# ---------------------------------------------------------
# Linear Regression
# ---------------------------------------------------------

st.header("Linear Regression")

linear = results[results["Model"] == "Linear Regression"][
    ["Dataset", "MAE", "RMSE", "R²"]
]

st.dataframe(linear, use_container_width=True)

st.info(
    "Rolling Median produced the best performance for Linear Regression "
    "with the lowest MAE and RMSE, and the highest R² score."
)

st.divider()

# ---------------------------------------------------------
# Best Model From Each Algorithm
# ---------------------------------------------------------

st.header("Best Configuration for Each Model")

st.dataframe(best, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# Visual Comparison
# ---------------------------------------------------------

st.header("Performance Comparison")

fig, ax = plt.subplots(figsize=(8,5))

ax.bar(best["Model"], best["MAE"])

ax.set_title("Best MAE Obtained by Each Model")
ax.set_ylabel("MAE")

st.pyplot(fig)

fig, ax = plt.subplots(figsize=(8,5))

ax.bar(best["Model"], best["RMSE"])

ax.set_title("Best RMSE Obtained by Each Model")
ax.set_ylabel("RMSE")

st.pyplot(fig)

fig, ax = plt.subplots(figsize=(8,5))

ax.bar(best["Model"], best["R²"])

ax.set_title("Best R² Score Obtained by Each Model")
ax.set_ylabel("R²")

st.pyplot(fig)

st.divider()

# ---------------------------------------------------------
# Overall Ranking
# ---------------------------------------------------------

st.header("Overall Ranking")

ranking = best.sort_values(
    by=["R²", "RMSE", "MAE"],
    ascending=[False, True, True]
).reset_index(drop=True)

ranking.index = ranking.index + 1

st.dataframe(ranking, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# Conclusion
# ---------------------------------------------------------

st.header("Conclusion")

st.write("""
The Rolling Median preprocessing technique consistently produced the best
results across all machine learning models. Among the evaluated algorithms,
Gradient Boosting achieved the highest prediction accuracy, recording the
lowest MAE and RMSE together with the highest R² score before
hyperparameter optimization.

Based on these observations, the Rolling Median dataset was selected for
further optimization using RandomizedSearchCV.
""")