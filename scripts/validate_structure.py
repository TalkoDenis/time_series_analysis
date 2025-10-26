def validate_structure(df):
    if df.empty:
        raise Exception("DataFrame is empty! No data found.")

    if len(df.columns) < 2:
        raise Exception(
            f"""DataFrame is not valid!
            Expected at least 2 columns, got {len(df.columns)}"""
        )

    return df
