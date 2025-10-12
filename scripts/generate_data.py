# === Генерация тестовых данных на 365 дней === #
np.random.seed(42)
days = 365
base_vals = [200, 200, 220, 230, 200, 230, 240, 210, 200, 200, 220, 230, 200, 230, 240, 210]

dates = pd.date_range(start="2025-01-01", periods=days)
fd_vals = [base_vals[i % len(base_vals)] + np.random.randint(-15, 16) for i in range(days)]

df = pd.DataFrame({
    "month": dates,
    "country_name": ["Аргентина"] * days,
    "fd_cnt": fd_vals
})

param_grid = {
            "max_depth": [3, 5, 7, 9],
            "eta": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "n_estimators": [500, 1000, 1500]
        }