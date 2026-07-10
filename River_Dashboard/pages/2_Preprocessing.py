import streamlit as st

st.title("Data Preprocessing")

st.write("""
The original river discharge dataset contained anomalous observations that could
affect the performance of machine learning models. Three preprocessing techniques
were implemented and evaluated to identify and correct these anomalies before
model training.
""")

# ---------------------------------------------------------------------
# Hampel Filter
# ---------------------------------------------------------------------

st.header("1. Hampel Filter")

st.write("""
The Hampel Filter identifies outliers by comparing each observation with the
median of neighboring values within a moving window. Values exceeding a specified
threshold are treated as anomalies and replaced using the local median, reducing
the impact of isolated spikes while preserving the overall trend.
""")

st.subheader("Anomalies Detected")

st.markdown("""
- Tamen : **334**
- Daporizo : **192**
""")

col1, col2 = st.columns(2)

with col1:
    st.image(
        "images/hampel.png",
        caption="Detected anomalies in Tamen",
        use_container_width=True
    )

with col2:
    st.image(
        "images/hampel_final.png",
        caption="Original vs Cleaned Tamen",
        use_container_width=True
    )

st.divider()

# ---------------------------------------------------------------------
# First Difference + Reversal
# ---------------------------------------------------------------------

st.header("2. First Difference + Reversal")

st.write("""
This method detects abrupt changes by analyzing the first difference between
consecutive observations. Significant spikes followed by sudden reversals are
identified as anomalies and corrected to improve the continuity of the time
series.
""")

st.subheader("Anomalies Detected")

st.markdown("""
- Tamen : **12**
- Daporizo : **54**
- Gerukamukh : **20**
""")

col1, col2 = st.columns(2)

with col1:
    st.image(
        "images/reversal.png",
        caption="Detected anomalies in Tamen",
        use_container_width=True
    )

with col2:
    st.image(
        "images/reversal_final.png",
        caption="Original vs Cleaned Tamen",
        use_container_width=True
    )

st.divider()

# ---------------------------------------------------------------------
# Rolling Median
# ---------------------------------------------------------------------

st.header("3. Rolling Median")

st.write("""
The Rolling Median method replaces anomalous observations using the median value
computed over a moving window. This approach effectively smooths isolated spikes
while preserving the natural behavior of the river discharge data.
""")

st.subheader("Anomalies Detected")

st.markdown("""
- Tamen : **608**
- Daporizo : **514**
- Gerukamukh : **688**
""")

col1, col2 = st.columns(2)

with col1:
    st.image(
        "images/rolling.png",
        caption="Detected anomalies in Tamen",
        use_container_width=True
    )

with col2:
    st.image(
        "images/rolling_final.png",
        caption="Original vs Cleaned Tamen",
        use_container_width=True
    )

st.divider()

# ---------------------------------------------------------------------
# Selection of the Best Method
# ---------------------------------------------------------------------

st.header("Selection of the Best Preprocessing Method")

st.write("""
All three preprocessing techniques were evaluated by training four machine
learning models—Linear Regression, Random Forest, Gradient Boosting, and
XGBoost—on each processed dataset.

Among the three methods, the Rolling Median dataset consistently achieved the
lowest prediction errors (MAE and RMSE) together with the highest R² values
across most models. Based on these results, the Rolling Median dataset was
selected for hyperparameter optimization using RandomizedSearchCV.
""")