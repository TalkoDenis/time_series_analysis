import pandas as pd


def get_forecast(model, train_data, test_data):
    n_periods = len(test_data)
    forecast_exog = test_data[["ds"]]

    fitted, confint = model.predict(
        n_periods=n_periods, return_conf_int=True, exogenous=forecast_exog
    )

    fitted_series = pd.Series(fitted, index=test_data.index)
    lower_series = pd.Series(confint[:, 0], index=test_data.index)
    upper_series = pd.Series(confint[:, 1], index=test_data.index)

    result_df = pd.DataFrame(
        {
            "ds": test_data["ds"],
            "Real_data": test_data["y"],
            "Forecust_data": fitted_series,
            "min": lower_series,
            "max": upper_series,
        }
    )

    result_df["Diff"] = result_df["Real_data"] - result_df["Forecust_data"]

    return result_df.reset_index(drop=True)
