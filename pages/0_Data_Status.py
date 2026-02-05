
import streamlit as st
import yfinance as yf
import pandas as pd
import os

from utils.validator import (
    data_exists,
    load_data,
    validate_columns,
    data_health,
    last_updated,
    file_checksum
)

# -------------------------------------------------
# Session state initialization
# -------------------------------------------------
if "data_ready" not in st.session_state:
    st.session_state["data_ready"] = False

# -------------------------------------------------
# Page header
# -------------------------------------------------
st.header("📁 Data Status & Readiness (Gatekeeper)")

st.markdown("""
This tab acts as a **mandatory data-quality gate**.  
All analysis tabs remain **locked** until the dataset passes every check below.
""")

# =================================================
# 1️⃣ Data availability
# =================================================
st.subheader("1️⃣ Data Availability")

if not data_exists():
    st.error("❌ dxy_clean.csv not found")
    st.info("Use the **Refresh / Regenerate Data** button below to create it.")
else:
    st.success("✅ dxy_clean.csv detected")

# =================================================
# 2️⃣ Data refresh / regenerate button
# =================================================
st.subheader("2️⃣ Refresh / Regenerate DXY Data")

if st.button("🔄 Download & Regenerate DXY Data"):
    with st.spinner("Downloading and processing DXY data..."):
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
    st.rerun()

# -------------------------------------------------
# Stop execution if data still not available
# -------------------------------------------------
if not data_exists():
    st.session_state["data_ready"] = False
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
# 5️⃣ File integrity (checksum)
# =================================================
st.subheader("5️⃣ File Integrity Check (Checksum)")
st.code(file_checksum(), language="text")

# =================================================
# 6️⃣ Unit-test–like data quality checks
# =================================================
st.subheader("6️⃣ Data Quality Tests")

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
# 7️⃣ Graded rubric (assessment-ready)
# =================================================
st.subheader("🎓 Graded Rubric – Data Engineering (10 Marks)")

rubric = {
    "Data file exists (2)": data_exists(),
    "Correct columns (2)": col_check["valid"],
    "No missing values (2)": health["missing_values"] == 0,
    "Datetime index (2)": str(type(df.index)).endswith("DatetimeIndex'>"),
    "Sufficient observations (2)": health["rows"] > 500
}

score = sum(2 for v in rubric.values() if v)

rubric_df = pd.DataFrame({
    "Criterion": rubric.keys(),
    "Status": ["✔" if v else "✘" for v in rubric.values()],
    "Marks Awarded": [2 if v else 0 for v in rubric.values()]
})

st.table(rubric_df)
st.metric("Total Score", f"{score} / 10")

# =================================================
# 8️⃣ Final readiness flag (locks other tabs)
# =================================================
st.subheader("🚦 Final Readiness Status")

if all(tests.values()):
    st.success("🚀 DATA READY — Analysis tabs are now unlocked")
    st.session_state["data_ready"] = True
else:
    st.error("⛔ DATA NOT READY — Fix issues above")
    st.session_state["data_ready"] = False
