
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Advanced Model Enhancement", layout="wide")

st.title("Advanced Model Enhancement")
st.markdown("""
This page presents the second phase of the project, where additional hydrological
features were engineered to improve river discharge prediction beyond the original
two-feature models.
""")

st.header("1. Original Models (2 Features)")
st.write("Original models were trained using only **Tamen** and **Daporizo**.")

old = pd.DataFrame({
    "Model":["Random Forest","Gradient Boosting","XGBoost"],
    "Dataset":["Rolling","Rolling","Rolling"],
    "MAE":[365.818106,384.878217,394.471565],
    "RMSE":[650.003693,653.311067,680.769289],
    "R²":[0.816101,0.814225,0.798281]
})
st.dataframe(old, use_container_width=True)

st.header("2. Feature Engineering")

features = [
"Tamen","Daporizo",
"Rain_Gerukamukh","Rain_Tamen","Rain_Daporijo",
"Rain_Confluence","Rain_Point5","Rain_Point6",
"Rain_Gerukamukh_3day","Rain_Gerukamukh_7day",
"Rain_Tamen_3day","Rain_Tamen_7day",
"Rain_Daporijo_3day","Rain_Daporijo_7day",
"Rain_Confluence_3day","Rain_Confluence_7day",
"Rain_Point5_3day","Rain_Point5_7day",
"Rain_Point6_3day","Rain_Point6_7day",
"MeanRain","MeanRain_3day","MeanRain_7day"
]

st.success(f"Number of predictor features increased from **2** to **{len(features)}**.")

with st.expander("View engineered features"):
    st.write(features)

st.header("3. Enhanced Model Performance")

new = pd.DataFrame({
"Model":["Random Forest","Gradient Boosting","XGBoost"],
"MAE":[133.361653,169.676433,164.895873],
"RMSE":[214.189964,245.062632,246.996409],
"R²":[0.981007,0.975137,0.974744]
})

c1,c2,c3=st.columns(3)
best=new.sort_values("R²",ascending=False).iloc[0]
c1.metric("Best Model",best["Model"])
c2.metric("Best R²",f"{best['R²']:.4f}")
c3.metric("Features",len(features))

st.dataframe(new,use_container_width=True)

st.subheader("Performance Comparison")
fig=px.bar(new,x="Model",y="R²",text="R²",color="Model")
st.plotly_chart(fig,use_container_width=True)

st.header("4. Improvement Over Original Models")

comparison=old.merge(new,on="Model",suffixes=("_Old","_New"))
comparison["R² Improvement"]=comparison["R²_New"]-comparison["R²_Old"]
comparison["MAE Reduction"]=comparison["MAE_Old"]-comparison["MAE_New"]
comparison["RMSE Reduction"]=comparison["RMSE_Old"]-comparison["RMSE_New"]

st.dataframe(comparison[[
"Model","R²_Old","R²_New","R² Improvement",
"MAE_Old","MAE_New","RMSE_Old","RMSE_New"
]],use_container_width=True)

fig=px.bar(
comparison,
x="Model",
y="R² Improvement",
text="R² Improvement",
color="Model",
title="Increase in R² after Feature Engineering"
)
st.plotly_chart(fig,use_container_width=True)

st.header("5. Feature Importance")

rf=pd.DataFrame({
"Feature":["Daporizo","Tamen","Rain_Tamen_7day","MeanRain_7day","Rain_Tamen_3day",
"Rain_Point6_7day","Rain_Point5_7day","Rain_Daporijo_7day","Rain_Confluence_7day","Rain_Gerukamukh_7day"],
"Importance":[0.251599,0.174879,0.071413,0.053518,0.050981,0.047243,0.042820,0.036751,0.035928,0.033820]
})

gb=pd.DataFrame({
"Feature":["Daporizo","Tamen","Rain_Confluence_7day","Rain_Point5_7day","Rain_Tamen_7day",
"Rain_Tamen_3day","MeanRain_7day","Rain_Gerukamukh_7day","MeanRain_3day","Rain_Daporijo_3day"],
"Importance":[0.323193,0.146216,0.118114,0.077113,0.047559,0.034618,0.031736,0.028968,0.026944,0.021484]
})

xgb=pd.DataFrame({
"Feature":["Daporizo","MeanRain_3day","Rain_Tamen_7day","Tamen","Rain_Confluence_7day",
"Rain_Confluence_3day","MeanRain_7day","Rain_Tamen_3day","Rain_Gerukamukh_7day","Rain_Daporijo_7day"],
"Importance":[0.311266,0.094345,0.068196,0.064702,0.064232,0.059994,0.049347,0.036030,0.035399,0.034567]
})

tabs=st.tabs(["Random Forest","Gradient Boosting","XGBoost"])

datasets=[rf,gb,xgb]
notes=[
"Random Forest achieved the best predictive performance after feature engineering.",
"Gradient Boosting distributed importance across several rainfall-derived features.",
"XGBoost relied heavily on Daporizo and ignored several rainfall station variables."
]

for tab,df,note in zip(tabs,datasets,notes):
    with tab:
        st.dataframe(df,use_container_width=True)
        fig=px.bar(df.sort_values("Importance"),
                   x="Importance",
                   y="Feature",
                   orientation="h",
                   title="Feature Importance")
        st.plotly_chart(fig,use_container_width=True)
        st.info(note)

st.header("6. Key Findings")

st.markdown("""
- Feature engineering increased the predictor set from **2 to 23** features.
- Random Forest achieved the highest accuracy (**R² = 0.9810**).
- All three models showed substantial improvement compared to the original models.
- Daporizo remained the most influential feature across all models.
- Seven-day accumulated rainfall features consistently ranked among the most important predictors.
- Rainfall-derived temporal features significantly improved prediction accuracy.
""")

st.header("7. Conclusion")

st.success("""
The introduction of rainfall-based, temporal and aggregated hydrological features
substantially improved model performance. Compared to the original two-feature
models, all three machine learning algorithms achieved much higher prediction
accuracy, with Random Forest emerging as the best-performing model.
""")
