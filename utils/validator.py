
import pandas as pd
import os
from datetime import datetime

# -------------------------------------------------
# Configuration
# -------------------------------------------------
REQUIRED_COLUMNS = {"DXY", "Returns", "MA_50", "MA_200"}
DATA_PATH = "data/dxy_clean.csv"

# -------------------------------------------------
# File existence check
# -------------------------------------------------
def data_exists(path=DATA_PATH):
    return os.path.exists(path)

# -------------------------------------------------
# Load data safely
# -------------------------------------------------
def load_data(path=DATA_PATH):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df.sort_index(inplace=True)
    return df

# -------------------------------------------------
# Column structure validation
# -------------------------------------------------
def validate_columns(df):
    existing = set(df.columns)
    missing = REQUIRED_COLUMNS - existing
    extra = existing - REQUIRED_COLUMNS
    return {
        "valid": len(missing) == 0,
        "missing": list(missing),
        "extra": list(extra)
    }

# -------------------------------------------------
# Data health summary
# -------------------------------------------------
def data_health(df):
    return {
        "start": df.index.min().strftime("%Y-%m-%d"),
        "end": df.index.max().strftime("%Y-%m-%d"),
        "rows": int(df.shape[0]),
        "missing_values": int(df.isnull().sum().sum())
    }

# -------------------------------------------------
# Last updated timestamp
# -------------------------------------------------
def last_updated(path=DATA_PATH):
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
