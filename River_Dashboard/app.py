import streamlit as st

st.set_page_config(
    page_title="River Flow Prediction Dashboard",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 River Flow Prediction Dashboard")

st.markdown("""
## Overview

This dashboard presents the complete workflow of a river flow prediction project.

The project includes:
- Data preprocessing using three different methods
- Training four machine learning models
- Hyperparameter experimentation
- Model comparison
- Hyperparameter optimization using RandomizedSearchCV
- River flow prediction
""")