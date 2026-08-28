import os
import pandas as pd

def write_df(df, path):
    """Save a DataFrame to CSV or Parquet based on the file extension."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith('.csv'):
        df.to_csv(path, index=False)
    elif path.endswith('.parquet'):
        try:
            df.to_parquet(path, index=False)
        except ImportError:
            raise ImportError("Parquet engine missing — run: pip install pyarrow")
    else:
        raise ValueError(f"Unsupported file type: {path}")
    print(f"Saved: {path}")

def read_df(path):
    """Load a DataFrame from CSV or Parquet based on the file extension."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file: {path}")
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.parquet'):
        try:
            return pd.read_parquet(path)
        except ImportError:
            raise ImportError("Parquet engine missing — run: pip install pyarrow")
    else:
        raise ValueError(f"Unsupported file type: {path}")
