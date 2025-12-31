import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ---- Models
def quad(N, a, b, c):
    return a + b * N + c * N ** 2


def quad_plateau(N, a, b, c, N_b):
    q = a + b * N + c * N ** 2
    plateau = a + b * N_b + c * N_b ** 2
    return np.where(N <= N_b, q, plateau)


# ---- IC helpers
def _ic_from_residuals(y_true, y_pred, k: int):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = y_true.size
    rss = float(np.sum((y_true - y_pred) ** 2))
    # guard against rss==0
    mse = rss / max(n, 1)
    aic = n * np.log(mse if mse > 0 else np.finfo(float).eps) + 2 * k
    bic = n * np.log(mse if mse > 0 else np.finfo(float).eps) + k * np.log(max(n, 1))
    aicc = np.inf
    if n - k - 1 > 0:
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    return {"RSS": rss, "AIC": aic, "AICc": aicc, "BIC": bic}


# ---- Fit helpers
def fit_quadratic(df, N_col="N", y_col="Yield"):
    x = df[N_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    # initial guess via polyfit (returns [c, b, a])
    c0, b0, a0 = np.polyfit(x, y, deg=2)
    p0 = [a0, b0, c0]

    popt, pcov = curve_fit(quad, x, y, p0=p0, maxfev=10000)
    a, b, c = popt

    pred_y = quad(x, a, b, c)
    ic = _ic_from_residuals(y, pred_y, k=3)

    aonr = -b / (2 * c) if c < 0 else np.nan
    aonr = float(np.clip(aonr, x.min(), x.max())) if np.isfinite(aonr) else np.nan

    return {
        "model": "quadratic",
        "params": (a, b, c),
        "AONR": aonr,
        "IC": ic,  # dict with RSS, AIC, AICc, BIC
        "pred": lambda N: quad(np.asarray(N, float), a, b, c),
    }


def fit_quadratic_plateau(df, N_col="N", y_col="Yield"):
    x = df[N_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    # initial guesses: quadratic + break at argmax y
    c0, b0, a0 = np.polyfit(x, y, deg=2)
    Nb0 = float(x[np.argmax(y)])
    p0 = [a0, b0, c0, Nb0]

    # bounds: c <= 0; Nb within data range
    lower = [-np.inf, -np.inf, -np.inf, x.min()]
    upper = [np.inf, np.inf, 0.0, x.max()]

    popt, pcov = curve_fit(quad_plateau, x, y, p0=p0, bounds=(lower, upper), maxfev=20000)
    a, b, c, Nb = popt

    pred_y = quad_plateau(x, a, b, c, Nb)
    ic = _ic_from_residuals(y, pred_y, k=4)

    # AONR is vertex up to the break
    vertex = -b / (2 * c) if c < 0 else Nb
    aonr = float(np.clip(vertex, x.min(), Nb))

    return {
        "model": "quadratic-plateau",
        "params": (a, b, c, float(Nb)),
        "AONR": aonr,
        "IC": ic,  # dict with RSS, AIC, AICc, BIC
        "pred": lambda N: quad_plateau(np.asarray(N, float), a, b, c, Nb),
    }


# ---- EONR helpers (unchanged)
def eonr_quadratic(params, price_N_per_kg, price_grain_per_unit, N_min, N_max):
    a, b, c = params
    if c >= 0:
        return float(N_min)
    ratio = price_N_per_kg / price_grain_per_unit
    N_star = (ratio - b) / (2 * c)
    return float(np.clip(N_star, N_min, N_max))


def eonr_quadratic_plateau(params, price_N_per_kg, price_grain_per_unit, N_min, N_max):
    a, b, c, Nb = params
    if c >= 0:
        return float(N_min)
    ratio = price_N_per_kg / price_grain_per_unit
    N_star = (ratio - b) / (2 * c)
    return float(np.clip(min(N_star, Nb), N_min, N_max))
