from scripts.generate_data import generate_dates
from scripts.forecast import forecast_xgb_days


param_grid = {
            "max_depth": [3, 5, 7, 9],
            "eta": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "n_estimators": [500, 1000, 1500]
        }

df = generate_dates()
forecast = forecast_xgb_days(df, "Spain", param_grid=param_grid, days_ahead=10, tune_params=True, n_lags=3)
print(forecast)