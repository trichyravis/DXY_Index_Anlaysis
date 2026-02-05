
import streamlit as st
import yfinance as yf
import pandas as pd
import os

from utils.validator import (
    data_exists,
    load_data,
    validate_columns,
    data_health,
    last_updated
)

st.header("📁 Data Status & Readiness (Gatekeeper)")

st.markdown("""
This tab **must pass all checks** before any analysis is performed.
It acts as a **data-quality firewall** for the entire project.
""")

# =================================================
# 1️⃣ Data availability
# =================================================
st.subheader("1️⃣ Data Availability")

if not data_exists():
    st.error("❌ dxy_clean.csv not found")
    st.info("Use the **Refresh Data** button below to generate it.")
else:
    st.success("✅ dxy_clean.csv detected")

# =================================================
# 2️⃣ Data refresh button
# =================================================
st.subheader("2️⃣ Refresh / Regenerate Data")

if st.button("🔄 Download & Regenerate DXY Data"):
    with st.spinner("Downloading DXY data..."):
        dxy = yf.download(
            "DX-Y.NYB",
            start="2000-01-01",
            auto_adjust=False,
            progress=False
        )[["Close"]]

        dxy.rename(columns={"Close": "DXY"}, inplace=True)
        dxy.index = pd.to_datetime(dxy.index)
        dxy.sort_index(inplace=True)

        dxy["Returns"] = dxy["DXY"].pct_change()
        dxy["MA_50"] = dxy["DXY"].rolling(50).mean()
        dxy["MA_200"] = dxy["DXY"].rolling(200).mean()

        dxy.dropna(inplace=True)

        os.makedirs("data", exist_ok=True)
        dxy.to_csv("data/dxy_clean.csv")

    st.success("✅ Data successfully refreshed")
    st.rerun()   # ✅ FIXED LINE

# =================================================
# Stop execution if data not available
# =================================================
if not data_exists():
    st.stop()

# =================================================
# Load data
# =================================================
df = load_data()

# =================================================
# 3️⃣ Column structure validation
# =================================================
st.subheader("3️⃣ Column Structure Validation")

col_check = validate_columns(df)

if col_check["valid"]:
    st.success("✅ Required columns present")
else:
    st.error("❌ Column validation failed")
    st.write("Missing columns:", col_check["missing"])
    st.write("Unexpected columns:", col_check["extra"])

# =================================================
# 4️⃣ Data health summary
# =================================================
st.subheader("4️⃣ Data Health Summary")

health = data_health(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Start Date", health["start"])
c2.metric("End Date", health["end"])
c3.metric("Rows", health["rows"])
c4.metric("Missing Values", health["missing_values"])

st.caption(f"📌 Last updated: {last_updated()}")

# =================================================
# 5️⃣ Unit-test–like checks
# =================================================
st.subheader("5️⃣ Data Quality Tests")

tests = {
    "File exists": data_exists(),
    "Datetime index": str(type(df.index)).endswith("DatetimeIndex'>"),
    "No missing values": health["missing_values"] == 0,
    "Required columns present": col_check["valid"],
    "Sufficient observations (>500)": health["rows"] > 500
}

test_results = pd.DataFrame({
    "Test": tests.keys(),
    "Status": ["PASS ✅" if v else "FAIL ❌" for v in tests.values()]
})

st.table(test_results)

# =================================================
# Final readiness flag
# =================================================
st.subheader("🚦 Final Readiness Status")

if all(tests.values()):
    st.success("🚀 DATA READY — You may proceed to analysis tabs")
else:
    st.error("⛔ DATA NOT READY — Fix issues above before continuing")
