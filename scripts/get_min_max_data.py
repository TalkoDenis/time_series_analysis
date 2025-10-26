import pandas as pd


def get_min_data(df) -> int:
    return pd.to_datetime(df["ds"].min())


def get_max_data(df) -> int:
    return pd.to_datetime(df["ds"].max())
