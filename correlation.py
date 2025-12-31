import math
from typing import Literal, Optional, Union, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

Method = Literal["pearson", "spearman", "kendall"]

def _series_like(df: pd.DataFrame, a: Union[str, pd.Series, np.ndarray]):
    if isinstance(a, str):
        return df[a]
    if isinstance(a, pd.Series):
        return a
    arr = np.asarray(a)
    return pd.Series(arr)

def _to_numeric(s: pd.Series, name: str) -> pd.Series:
    """
    Coerce series to float with NaNs where conversion fails.
    - bool -> 0/1
    - category -> codes (with NaN preserved)
    - datetime -> seconds since epoch (float)
    - object -> to_numeric(errors='coerce')
    """
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_categorical_dtype(s):
        out = s.cat.codes.astype("float64")
        out[out < 0] = np.nan  # -1 for NaN in cat.codes
        return out
    if pd.api.types.is_datetime64_any_dtype(s):
        # convert to seconds since epoch
        return (s.view("int64") / 1e9).astype("float64")
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("float64")
    # object or mixed -> best-effort numeric
    return pd.to_numeric(s, errors="coerce").astype("float64")

def corr_stats(
    df: pd.DataFrame,
    x: Union[str, pd.Series, np.ndarray],
    y: Union[str, pd.Series, np.ndarray],
    *,
    method: Method = "pearson",
    alpha: float = 0.05,
    dropna: Literal["pairwise", "listwise"] = "pairwise",
    ci_bootstrap: bool = False,
    n_boot: int = 2000,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Correlate two variables and return coefficient, p-value, N, and CI.
    Inputs are coerced to numeric (see _to_numeric) to avoid dtype errors.
    """
    sx = _series_like(df, x)
    sy = _series_like(df, y)

    # Align indices if they are Series with different indexes
    if isinstance(sx, pd.Series) and isinstance(sy, pd.Series):
        sx, sy = sx.align(sy, join="inner")

    # Coerce to numeric, preserving NaNs
    sx = _to_numeric(pd.Series(sx), "x")
    sy = _to_numeric(pd.Series(sy), "y")

    # NA handling
    if dropna == "pairwise":
        mask = sx.notna() & sy.notna()
    elif dropna == "listwise" and isinstance(x, str) and isinstance(y, str):
        # stricter: same rows must be complete across df
        base_mask = df.notna().all(axis=1)
        # align mask to sx/sy index
        base_mask = base_mask.reindex(sx.index, fill_value=False)
        mask = base_mask & sx.notna() & sy.notna()
    else:
        # fallback to pairwise if listwise not applicable
        mask = sx.notna() & sy.notna()

    x_clean = sx[mask].to_numpy(dtype="float64", copy=False)
    y_clean = sy[mask].to_numpy(dtype="float64", copy=False)
    n = x_clean.size

    if n < 2:
        raise ValueError("Not enough non-missing data to compute correlation.")

    result: Dict[str, Any] = {
        "method": method,
        "x": x if isinstance(x, str) else "x",
        "y": y if isinstance(y, str) else "y",
        "n": int(n),
        "alpha": float(alpha),
    }

    # Guard against constant inputs (scipy raises/returns nan)
    x_const = np.nanstd(x_clean, ddof=1) == 0 or np.all(x_clean == x_clean[0])
    y_const = np.nanstd(y_clean, ddof=1) == 0 or np.all(y_clean == y_clean[0])
    if x_const or y_const:
        raise ValueError("Correlation undefined: one or both inputs are constant.")

    if method == "pearson":
        r, p = stats.pearsonr(x_clean, y_clean)
        dfree = n - 2
        t_stat = r * math.sqrt(dfree / (1 - r * r)) if abs(r) < 1 else np.inf * np.sign(r)

        # 95% CI via Fisher z-transform
        if n > 3 and abs(r) < 1:
            z = np.arctanh(r)
            se = 1 / math.sqrt(n - 3)
            zcrit = stats.norm.ppf(1 - alpha / 2)
            lo, hi = np.tanh(z - zcrit * se), np.tanh(z + zcrit * se)
        else:
            lo, hi = np.nan, np.nan

        # Regression-style descriptors
        x_mean, y_mean = np.mean(x_clean), np.mean(y_clean)
        x_std, y_std = np.std(x_clean, ddof=1), np.std(y_clean, ddof=1)
        slope = r * (y_std / x_std) if x_std > 0 else np.nan
        intercept = y_mean - slope * x_mean if np.isfinite(slope) else np.nan

        result.update({
            "coef": float(r),
            "p_value": float(p),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "t_stat": float(t_stat),
            "df": int(dfree),
            "r_squared": float(r * r),
            "slope": float(slope) if np.isfinite(slope) else np.nan,
            "intercept": float(intercept) if np.isfinite(intercept) else np.nan,
        })

    elif method == "spearman":
        rho, p = stats.spearmanr(x_clean, y_clean)
        lo, hi = np.nan, np.nan
        if ci_bootstrap:
            rng = np.random.default_rng(random_state)
            boots = np.empty(n_boot, dtype="float64")
            for b in range(n_boot):
                idx = rng.integers(0, n, size=n)
                boots[b] = stats.spearmanr(x_clean[idx], y_clean[idx]).correlation
            lo, hi = np.nanpercentile(boots, [100 * (alpha / 2), 100 * (1 - alpha / 2)])
        result.update({"coef": float(rho), "p_value": float(p), "ci_low": float(lo), "ci_high": float(hi)})

    elif method == "kendall":
        tau, p = stats.kendalltau(x_clean, y_clean)
        lo, hi = np.nan, np.nan
        if ci_bootstrap:
            rng = np.random.default_rng(random_state)
            boots = np.empty(n_boot, dtype="float64")
            for b in range(n_boot):
                idx = rng.integers(0, n, size=n)
                boots[b] = stats.kendalltau(x_clean[idx], y_clean[idx]).correlation
            lo, hi = np.nanpercentile(boots, [100 * (alpha / 2), 100 * (1 - alpha / 2)])
        result.update({"coef": float(tau), "p_value": float(p), "ci_low": float(lo), "ci_high": float(hi)})

    else:
        raise ValueError("method must be 'pearson', 'spearman', or 'kendall'.")

    return result
