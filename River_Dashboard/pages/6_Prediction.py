import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediction", page_icon="🔮")

st.title("🔮 River Flow Prediction")
st.markdown("Predict the **Gerukamukh** river flow using the trained machine learning models.")

st.divider()

# -----------------------------
# Load Models
# -----------------------------
rf_model = joblib.load("models/rf_rolling.pkl")
xgb_model = joblib.load("models/xgb_rolling.pkl")
gb_model = joblib.load("models/gradient_boosting_rolling.pkl")

# -----------------------------
# Model Selection
# -----------------------------
model_choice = st.selectbox(
    "Select Model",
    [
        "Random Forest",
        "XGBoost",
        "Gradient Boosting"
    ]
)

st.divider()

# -----------------------------
# Input Values
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    tamen = st.number_input(
        "Tamen Flow",
        min_value=0.0,
        value=1000.0,
        step=10.0
    )

with col2:
    daporizo = st.number_input(
        "Daporizo Flow",
        min_value=0.0,
        value=1000.0,
        step=10.0
    )

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict River Flow", use_container_width=True):

    input_data = pd.DataFrame(
        [[tamen, daporizo]],
        columns=["Tamen", "Daporizo"]
    )

    if model_choice == "Random Forest":
        prediction = rf_model.predict(input_data)[0]

    elif model_choice == "XGBoost":
        prediction = xgb_model.predict(input_data)[0]

    else:
        prediction = gb_model.predict(input_data)[0]

    st.success("Prediction Completed!")

    st.metric(
        label="Predicted Gerukamukh Flow",
        value=f"{prediction:.2f}"
    )

    st.info(
        "The prediction is generated using the selected optimized machine learning model trained on the Rolling Median preprocessed dataset."
    )