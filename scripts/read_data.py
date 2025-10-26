import pandas as pd


def read_data(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{path}' is empty or malformed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    else:
        return df
