import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📂 Original Dataset")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_excel(
    "river_data.xlsx",
    header=2
)

# Remove catchment area row
df = df.iloc[1:].reset_index(drop=True)

# Keep only first four columns
df = df.iloc[:, :4]

# Rename columns
df.columns = [
    "datetime",
    "tributary_A",
    "tributary_B",
    "combined_river"
]

# Convert data types
df["datetime"] = pd.to_datetime(df["datetime"])
df["tributary_A"] = pd.to_numeric(df["tributary_A"])
df["tributary_B"] = pd.to_numeric(df["tributary_B"])
df["combined_river"] = pd.to_numeric(df["combined_river"])

# -----------------------------
# About
# -----------------------------
st.header("About the Dataset")

st.write("""
This dataset contains hourly river discharge measurements collected from the
Subansiri Basin between 2002 and 2006. It was used for anomaly detection,
preprocessing, machine learning model development, and river flow prediction.
""")

# -----------------------------
# Dataset Information
# -----------------------------
st.header("Dataset Information")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Records", len(df))
col2.metric("Features", len(df.columns))
col3.metric("Time Interval", "Hourly")
col4.metric("Study Period", "2002–2006")

# -----------------------------
# Preview
# -----------------------------
st.header("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

# -----------------------------
# Visualization
# -----------------------------
st.header("River Flow Visualization")

fig, ax = plt.subplots(figsize=(15,5))

ax.plot(df["datetime"], df["tributary_A"], label="Tamen")
ax.plot(df["datetime"], df["tributary_B"], label="Daporizo")
ax.plot(df["datetime"], df["combined_river"], label="Gerukamukh")

ax.set_xlabel("Date")
ax.set_ylabel("Discharge (cumec)")
ax.set_title("River Flow Time Series")
ax.legend()

st.pyplot(fig)