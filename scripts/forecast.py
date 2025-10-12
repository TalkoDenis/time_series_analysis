# time_series_analysis

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, RepeatedKFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor

# === ФУНКЦИЯ ПРОГНОЗА === #
def forecast_xgb_days(df, country, param_grid, days_ahead=10, model_params=None, tune_params=True, n_lags=3):
    """
    Прогноз XGBoost для дневных данных на N дней вперед.
    Возвращает прогноз и диапазон min/max с учётом ошибки модели.
    """
    df_country = df[df["country_name"] == country].sort_values("month").copy()
    df_country["month"] = pd.to_datetime(df_country["month"])

    # Генерация признаков
    df_country["day"] = df_country["month"].dt.day
    df_country["month_num"] = df_country["month"].dt.month
    df_country["year"] = df_country["month"].dt.year
    df_country["weekday"] = df_country["month"].dt.weekday

    for lag in range(1, n_lags + 1):
        df_country[f"lag_{lag}"] = df_country["fd_cnt"].shift(lag)
    df_country = df_country.dropna()

    # Разделение признаков и целевой
    lag_cols = [f"lag_{lag}" for lag in range(1, n_lags + 1)]
    feature_cols = ["day", "month_num", "year", "weekday"] + lag_cols
    X = df_country[feature_cols]
    y = df_country["fd_cnt"]

    # Подбор гиперпараметров
    if tune_params:
        param_grid = param_grid
        search = RandomizedSearchCV(
            XGBRegressor(random_state=42),
            param_distributions=param_grid,
            scoring='neg_mean_absolute_error',
            cv=3,
            n_iter=10,
            n_jobs=-1,
            random_state=42
        )
        search.fit(X, y)
        best_params = search.best_params_
        print("Лучшие параметры после RandomizedSearchCV:", best_params)
        model_params = best_params
    else:
        model_params = model_params or {}

    # Обучение модели
    model = XGBRegressor(**model_params, random_state=42)
    model.fit(X, y)

    # Оценка модели
    cv = RepeatedKFold(n_splits=min(5, len(y)), n_repeats=2, random_state=42)
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    mean_mae = np.abs(scores).mean()
    std_mae = np.abs(scores).std()
    y_pred = model.predict(X)
    mape = mean_absolute_percentage_error(y, y_pred) * 100

    print(f"Cross-validated MAE: {mean_mae:.3f} (±{std_mae:.3f})")
    print(f"MAPE на обучающих данных: {mape:.2f}%")

    # Прогноз на будущее
    last_date = df_country["month"].max()
    last_values = df_country["fd_cnt"].iloc[-n_lags:].tolist()
    future_preds = []

    for i in range(days_ahead):
        next_date = last_date + pd.Timedelta(days=1)
        X_next = pd.DataFrame([{
            "day": next_date.day,
            "month_num": next_date.month,
            "year": next_date.year,
            "weekday": next_date.weekday(),
            **{f"lag_{lag}": last_values[-lag] for lag in range(1, n_lags + 1)}
        }])
        next_pred = model.predict(X_next)[0]

        future_preds.append({
            "date": next_date,
            "forecast_min": next_pred - mean_mae,
            "forecast": next_pred,
            "forecast_max": next_pred + mean_mae
        })

        last_values.append(next_pred)
        last_values = last_values[-n_lags:]
        last_date = next_date

    forecast_df = pd.DataFrame(future_preds)
    return forecast_df

param_grid = {
            "max_depth": [3, 5, 7, 9],
            "eta": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "n_estimators": [500, 1000, 1500]
        }

# === Вызов функции для прогноза на 10 дней вперед === #
forecast = forecast_xgb_days(df, "Аргентина", param_grid=param_grid, days_ahead=10, tune_params=True, n_lags=3)
