def rename_columns(df):
    return df.rename(columns={df.columns[0]: "ds", df.columns[1]: "y"})
