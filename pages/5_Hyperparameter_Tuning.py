import streamlit as st
import pandas as pd

st.title("Hyperparameter Tuning")

# ---------------------------------------------------------
# Introduction
# ---------------------------------------------------------

st.write("""
After evaluating all preprocessing techniques, the **Rolling Median** dataset
consistently produced the best prediction performance across all machine learning
models. Therefore, this dataset was selected for hyperparameter optimization
using **RandomizedSearchCV**.

RandomizedSearchCV performs an efficient search over a predefined parameter
space by evaluating randomly selected parameter combinations through
cross-validation, reducing computation time while identifying high-performing
configurations.
""")

st.divider()

# ---------------------------------------------------------
# Random Forest
# ---------------------------------------------------------

st.header("Random Forest")

st.write("""
A total of **250** randomly selected parameter combinations were evaluated
using **5-fold cross-validation**, resulting in **1250 model fits**.
""")

st.subheader("Best Hyperparameters")

rf_params = pd.DataFrame({
    "Parameter": [
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "bootstrap"
    ],
    "Value": [
        500,
        "None",
        10,
        1,
        "log2",
        True
    ]
})

st.table(rf_params)

st.subheader("Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric("CV R²", "0.8376")
c2.metric("MAE", "361.34")
c3.metric("RMSE", "637.60")
c4.metric("R²", "0.8231")

st.divider()

# ---------------------------------------------------------
# Gradient Boosting
# ---------------------------------------------------------

st.header("Gradient Boosting")

st.write("""
A total of **200** randomly selected parameter combinations were evaluated
using **5-fold cross-validation**, resulting in **1000 model fits**.
""")

st.subheader("Best Hyperparameters")

gb_params = pd.DataFrame({
    "Parameter": [
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "subsample"
    ],
    "Value": [
        200,
        0.1,
        6,
        10,
        2,
        "sqrt",
        0.8
    ]
})

st.table(gb_params)

st.subheader("Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric("CV R²", "0.8431")
c2.metric("MAE", "368.58")
c3.metric("RMSE", "627.56")
c4.metric("R²", "0.8286")

st.divider()

# ---------------------------------------------------------
# XGBoost
# ---------------------------------------------------------

st.header("XGBoost")

st.write("""
A total of **200** randomly selected parameter combinations were evaluated
using **5-fold cross-validation**, resulting in **1000 model fits**.
""")

st.subheader("Best Hyperparameters")

xgb_params = pd.DataFrame({
    "Parameter": [
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "gamma",
        "reg_alpha",
        "reg_lambda",
        "subsample",
        "colsample_bytree"
    ],
    "Value": [
        200,
        0.2,
        5,
        5,
        0.3,
        0.1,
        10,
        1.0,
        1.0
    ]
})

st.table(xgb_params)

st.subheader("Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric("CV R²", "0.8280")
c2.metric("MAE", "381.12")
c3.metric("RMSE", "667.23")
c4.metric("R²", "0.8062")

st.divider()

# ---------------------------------------------------------
# Final Comparison
# ---------------------------------------------------------

st.header("Optimized Model Comparison")

comparison = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Gradient Boosting",
        "XGBoost"
    ],
    "Cross Validation R²": [
        0.8376,
        0.8431,
        0.8280
    ],
    "MAE": [
        361.34,
        368.58,
        381.12
    ],
    "RMSE": [
        637.60,
        627.56,
        667.23
    ],
    "R²": [
        0.8231,
        0.8286,
        0.8062
    ]
})

st.dataframe(comparison, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# Conclusion
# ---------------------------------------------------------

st.header("Conclusion")

st.write("""
Hyperparameter optimization further improved the performance of the tree-based
models by identifying parameter combinations that generalized well through
cross-validation.

Among the optimized models, **Gradient Boosting** achieved the highest overall
performance, recording the **highest R² score (0.8286)** and the **lowest RMSE
(627.56)**. **Random Forest** produced the **lowest MAE (361.34)**, while
**XGBoost** also demonstrated competitive performance after optimization.

These optimized models form the final stage of the predictive modeling
workflow before deployment for river discharge prediction.
""")