import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def bootstrap_mae(df, feature_col, target_col, n_boot=500, seed=42):
    """Bootstrap the MAE of a simple linear fit, returns (list_of_maes, ci_lower, ci_upper)."""
    rng = np.random.default_rng(seed)
    maes = []
    for _ in range(n_boot):
        sample = df.sample(n=len(df), replace=True, random_state=rng.integers(0, 1_000_000))
        X, y = sample[[feature_col]], sample[target_col]
        model = LinearRegression().fit(X, y)
        maes.append(mean_absolute_error(y, model.predict(X)))
    return maes, np.percentile(maes, 2.5), np.percentile(maes, 97.5)
