import pandas as pd

def encode_category_onehot(df, col='category'):
    """One-hot encode a categorical column, appended as new columns."""
    dummies = pd.get_dummies(df[col], prefix=col)
    return pd.concat([df, dummies], axis=1)

def add_zscore(df, col='value'):
    """Add a z-score column for the given numeric column."""
    df = df.copy()
    df[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()
    return df

def add_rolling_mean(df, col='value', window=3):
    """Add a rolling mean feature, sorted by date first."""
    df = df.copy()
    if 'date' in df.columns:
        df = df.sort_values('date')
    df[f'{col}_rolling_mean'] = df[col].rolling(window=window, min_periods=1).mean()
    return df
