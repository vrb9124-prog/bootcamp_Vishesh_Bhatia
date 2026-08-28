import pandas as pd

def eda_summary(df):
    """
    Return a one-row-per-column summary: dtype, missing count,
    missing %, and (for numeric columns) mean/std/skew.
    """
    rows = []
    for col in df.columns:
        row = {
            'column': col,
            'dtype': str(df[col].dtype),
            'missing': df[col].isna().sum(),
            'missing_pct': round(df[col].isna().mean() * 100, 1),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            row['mean'] = df[col].mean()
            row['std'] = df[col].std()
            row['skew'] = df[col].skew()
        else:
            row['n_unique'] = df[col].nunique()
        rows.append(row)
    return pd.DataFrame(rows)
