import numpy as np
import pandas as pd

def generate_dates(days=365, start='2025-01-01', country_name='Spain'):
    np.random.seed(42)
    days = days
    base_vals = [200, 210, 220, 230, 240]

    dates = pd.date_range(start=start, periods=days)
    fd_vals = [base_vals[i % len(base_vals)] + np.random.randint(-15, 16) for i in range(days)]

    df = pd.DataFrame({
        "month": dates,
        "country_name": country_name * days,
        "fd_cnt": fd_vals
    })
    return df