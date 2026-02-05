import pandas as pd
import os

def data_exists(path="data/dxy_clean.csv"):
    return os.path.exists(path)

def load_data(path="data/dxy_clean.csv"):
    return pd.read_csv(path, index_col=0, parse_dates=True)

def data_health(df):
    return {
        "start": df.index.min().date(),
        "end": df.index.max().date(),
        "rows": df.shape[0],
        "missing": int(df.isnull().sum().sum())
    }

