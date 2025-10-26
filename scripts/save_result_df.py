import pandas as pd


def save_result_df(future_df, forecast):
    results_df_org = pd.merge(
        future_df, forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds"
    )
    results_df_org.rename(
        columns={
            "ds": "week",
            "y": "fd_cnt",
            "yhat": "forecast",
            "yhat_lower": "min_forecast",
            "yhat_upper": "max_forecast",
        },
        inplace=True,
    )
    return results_df_org
