import pandas as pd


def validate_content(df):
    try:
        df["ds"] = pd.to_datetime(df["ds"])
    except Exception as e:
        print("ERROR: Failed to convert 'ds' column to datetime.")
        raise Exception(f"Invalid data in date column ('ds'): {e}")

    try:
        df["y"] = pd.to_numeric(df["y"], downcast="unsigned")
    except Exception as e:
        print("ERROR: Failed to convert 'y' column to numeric.")
        raise Exception(f"Invalid data in value column ('y'): {e}")

    return df.sort_values(by="ds").reset_index(drop=True)
