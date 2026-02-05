import streamlit as st

# -------------------------------------------------
# Streamlit App Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="DXY Analytics Project",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Main Landing Page
# -------------------------------------------------
st.title("📊 US Dollar Index (DXY) – Analytics Project")

st.markdown("""
### Project Overview

This application analyzes the **US Dollar Index (DXY)** using a  
**data-governance-first approach**.

🔒 **Important Rule**  
All analysis tabs are valid **only after** the dataset passes the  
**Data Status & Readiness Check**.

---

### Workflow
1. **Data Status & Readiness**  
   - Data availability  
   - Time period  
   - Missing values check  

2. **Exploratory Data Analysis (EDA)**  
   - Trends, returns, volatility, regimes  

3. **Correlation Analysis**  
   - DXY vs EUR/USD, Gold, Crude  

4. **Econometric Tests**  
   - Stationarity (ADF test)  

Use the **sidebar** to navigate through tabs.
""")

st.info("⬅️ Start with **Data Status & Readiness** from the sidebar")

