import streamlit as st
from utils.validator import load_data

df = load_data()

st.header("📊 Exploratory Data Analysis – DXY")

st.subheader("DXY Level")
st.line_chart(df['DXY'])

st.subheader("Returns")
st.line_chart(df['Returns'])

st.subheader("Moving Averages")
st.line_chart(df[['DXY','MA_50','MA_200']])

