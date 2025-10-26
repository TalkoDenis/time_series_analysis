import pandas as pd

from scripts.split_df import split_df


def test_split_df(sample_prophet_df):
    split_date = "2025-01-02"

    train_df, future_df = split_df(sample_prophet_df, data=split_date)

    assert len(train_df) == 2
    assert len(future_df) == 1
    assert train_df["ds"].max() == pd.to_datetime(split_date)
    assert future_df["ds"].min() > pd.to_datetime(split_date)
