import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediction", page_icon="🔮")

st.title("🔮 River Flow Prediction")
st.write("Predict the Gerukamukh river flow using the trained machine learning models.")

# ---------------------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("Final_Dataset.xlsx")

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce"
    )

    # Remove invalid dates
    df = df.dropna(subset=["Date"])

    return df


# ---------------------------------------------------------------------
# Load Model
# ---------------------------------------------------------------------
@st.cache_resource
def load_model(path):
    saved = joblib.load(path)
    return saved["model"], saved["features"]


df = load_data()

# ---------------------------------------------------------------------
# Model Selection
# ---------------------------------------------------------------------
model_choice = st.selectbox(
    "Select Model",
    [
        "Random Forest",
        "Gradient Boosting",
        "XGBoost"
    ]
)

model_paths = {
    "Random Forest": "models/randomforest_Gerukamukh.pkl",
    "Gradient Boosting": "models/gradientboost_Gerukamukh.pkl",
    "XGBoost": "models/xgboost_Gerukamukh.pkl"
}

model, feature_names = load_model(model_paths[model_choice])

# ---------------------------------------------------------------------
# Date Selection
# ---------------------------------------------------------------------
available_dates = sorted(
    df["Date"].dt.normalize().unique()
)

selected_date = st.date_input(
    "Select Date",
    value=available_dates[0].date(),
    min_value=available_dates[0].date(),
    max_value=available_dates[-1].date()
)

# ---------------------------------------------------------------------
# Flow Inputs
# ---------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    tamen = st.number_input(
        "Tamen Flow",
        min_value=0.0,
        value=1000.0
    )

with col2:
    daporijo = st.number_input(
        "Daporizo Flow",
        min_value=0.0,
        value=1000.0
    )

# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------
if st.button("Predict", use_container_width=True):

    row = df[
        df["Date"].dt.normalize() == pd.Timestamp(selected_date)
    ]

    if row.empty:
        st.error("No data available for this date.")
        st.stop()

    # Take the first row for that day
    row = row.iloc[[0]].copy()

    # Replace user inputs
    row["Tamen"] = tamen
    row["Daporizo"] = daporijo

    # Keep only the features used during training
    X = row[feature_names]

    prediction = model.predict(X)[0]

    st.success("Prediction Complete!")

    st.metric(
        "Predicted Gerukamukh Flow",
        f"{prediction:.2f} cumecs"
    )