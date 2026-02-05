import streamlit as st
from utils.validator import data_exists, load_data, data_health

st.header("📁 Data Status & Readiness Check")

st.markdown("""
This tab ensures that **DXY data is available and clean**
before any analysis is performed.
""")

# 1. Check existence
if not data_exists():
    st.error("❌ DXY data file not found. Download & clean data first.")
    st.stop()

st.success("✅ Data file detected")

# 2. Load & inspect
df = load_data()
health = data_health(df)

st.subheader("📅 Data Coverage")
c1, c2, c3 = st.columns(3)
c1.metric("Start Date", health["start"])
c2.metric("End Date", health["end"])
c3.metric("Observations", health["rows"])

st.subheader("🧪 Data Quality")
if health["missing"] == 0:
    st.success("✅ No missing values detected")
else:
    st.warning(f"⚠️ Missing values found: {health['missing']}")

st.subheader("🚦 Final Status")
if health["missing"] == 0 and health["rows"] > 0:
    st.success("🚀 Data READY for analysis")
else:
    st.error("⛔ Data NOT ready")

