import numpy as np


def corr(x, y) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    x0 = x[m] - x[m].mean()
    y0 = y[m] - y[m].mean()
    denom = np.sqrt((x0 ** 2).sum()) * np.sqrt((y0 ** 2).sum())
    return float((x0 * y0).sum() / denom) if denom > 0 else np.nan
