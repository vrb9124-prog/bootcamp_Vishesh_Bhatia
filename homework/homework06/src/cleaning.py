import pandas as pd

def fill_missing_median(df, cols=None):
    """
    Fill missing values in numeric columns with that column's median.
    If cols is None, applies to all numeric columns.
    Assumes missingness is roughly random, not systematically tied
    to the value itself (otherwise median-fill can bias results).
    """
    df = df.copy()
    if cols is None:
        cols = df.select_dtypes(include='number').columns
    for col in cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    return df

def drop_missing(df, how='any', subset=None):
    """
    Drop rows with missing values.
    how='any' drops a row if ANY listed column is missing;
    how='all' drops a row only if ALL listed columns are missing.
    subset limits which columns are checked (None = check all columns).
    """
    return df.dropna(how=how, subset=subset)

def normalize_data(df, cols=None):
    """
    Min-max normalize numeric columns to a 0-1 range.
    Assumes no extreme outliers are present (a single huge outlier
    would compress everything else toward 0).
    """
    df = df.copy()
    if cols is None:
        cols = df.select_dtypes(include='number').columns
    for col in cols:
        min_val, max_val = df[col].min(), df[col].max()
        if max_val > min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
    return df
