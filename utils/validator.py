
import pandas as pd
import os

# -------------------------------------------------
# Check whether cleaned data file exists
# -------------------------------------------------
def data_exists(path="data/dxy_clean.csv"):
    return os.path.exists(path)

# -------------------------------------------------
# Load data safely with enforced datetime index
# -------------------------------------------------
def load_data(path="data/dxy_clean.csv"):
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # Force datetime index (important for Streamlit Cloud)
    df.index = pd.to_datetime(df.index, errors="coerce")

    # Drop rows where date parsing failed (safety)
    df = df[~df.index.isna()]

    # Ensure sorted time series
    df.sort_index(inplace=True)

    return df

# -------------------------------------------------
# Data health & readiness summary
# -------------------------------------------------
def data_health(df):
    return {
        "start": df.index.min().strftime("%Y-%m-%d"),
        "end": df.index.max().strftime("%Y-%m-%d"),
        "rows": int(df.shape[0]),
        "missing": int(df.isnull().sum().sum())
    }
