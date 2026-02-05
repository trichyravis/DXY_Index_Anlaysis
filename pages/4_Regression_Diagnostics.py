
import streamlit as st
import yfinance as yf
import pandas as pd
import statsmodels.api as sm
from utils.validator import load_data

if not st.session_state.get("data_ready", False):
    st.warning("🔒 Complete Data Status first")
    st.stop()

st.header("📊 Regression Diagnostics: DXY vs Gold")

# Load DXY
dxy = load_data()["Returns"]

# Load Gold
gold = yf.download("GC=F", progress=False)["Close"].pct_change()

df = pd.concat([dxy, gold], axis=1)
df.columns = ["DXY_Returns", "Gold_Returns"]
df.dropna(inplace=True)

X = sm.add_constant(df["Gold_Returns"])
y = df["DXY_Returns"]

model = sm.OLS(y, X).fit()

st.subheader("OLS Regression Summary")
st.text(model.summary())

st.subheader("Residual Diagnostics")
st.line_chart(model.resid)
