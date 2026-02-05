
import streamlit as st
import yfinance as yf
import pandas as pd

st.header("🌍 Correlation Analysis")

dxy = yf.download("DX-Y.NYB")['Close']
gold = yf.download("GC=F")['Close']
eurusd = yf.download("EURUSD=X")['Close']

df = pd.concat([dxy, gold, eurusd], axis=1)
df.columns = ['DXY','Gold','EURUSD']
df.dropna(inplace=True)

st.subheader("Returns Correlation Matrix")
st.write(df.pct_change().corr())

st.subheader("Rolling Correlation: DXY vs Gold")
st.line_chart(
    df['DXY'].pct_change().rolling(60).corr(df['Gold'].pct_change())
)
