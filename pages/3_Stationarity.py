
import streamlit as st
from statsmodels.tsa.stattools import adfuller
from utils.validator import load_data

df = load_data()

def adf(series):
    result = adfuller(series.dropna())
    return result[0], result[1]

stat, pval = adf(df['Returns'])

st.header("📐 Stationarity Test – ADF")

st.metric("ADF Statistic", round(stat,4))
st.metric("P-value", round(pval,4))

if pval < 0.05:
    st.success("✅ Returns are stationary")
else:
    st.error("❌ Non-stationary series")
