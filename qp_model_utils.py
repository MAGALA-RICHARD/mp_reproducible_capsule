"""
Provides the utilitiess to fits a quadratic model SOC, balancece and corn grain yield
Author: Richard Magala
Created on Monday Oct 27 2024:
"""
import math
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import inv
from scipy.optimize import least_squares
from scipy.stats import t as tdist
from labels import LABELS
from logger import get_logger
from settings import RESULTS
from utils import load_manifest

conFIG = load_manifest()
MODULE_NAME = Path(__file__).stem
script_name = Path(__file__).stem
logger = get_logger(name=f'{script_name}')
prices = conFIG['prices']
pl = conFIG['plots']

P_y = prices['corn_grain']  # Per Mg grain yield
P_N = prices['nitrogen'] # Per kg N


def fit_quadratic_plateau(x, y, weights=None):
    """
    Quadratic-Plateau model:
      y = a + b x + c x^2 for x <= x0,  with c < 0
      y = plateau = a + b x0 + c x0^2 for x >= x0,  where x0 = -b / (2c)

    Returns: dict with params, SEs, t, p, and metrics (R2, adjR2, AIC, BIC, RMSE, etc.)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    n = x.size
    if n < 6:
        raise ValueError("Need at least 6 points to fit a QP model reliably.")

    # ----- weights -----
    if weights is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(weights, dtype=float)[mask]
        if w.ndim != 1 or w.size != n:
            raise ValueError("weights must be same length as x/y")
        # normalize weights for numerical stability
        w = w / np.nanmean(w)

    # ----- initial guess via quadratic OLS -----
    X = np.c_[np.ones(n), x, x ** 2]
    beta_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    a0, b0, c0 = beta_ols
    # enforce concavity for plateau (c < 0); if not, nudge
    if c0 >= -1e-6:
        c0 = -abs(c0) - 1e-3
    p0 = np.array([a0, b0, c0])

    # piecewise model using a,b,c -> x0, plateau
    def qp_predict(p, x_):
        a, b, c = p
        x0 = -b / (2.0 * c)  # vertex
        plateau = a + b * x0 + c * x0 ** 2
        y_hat = a + b * x_ + c * x_ ** 2
        y_hat = np.where(x_ <= x0, y_hat, plateau)
        return y_hat

    # residuals with weights
    def resid(p):
        y_hat = qp_predict(p, x)
        return np.sqrt(w) * (y - y_hat)

    # bounds: only constrain c<0 (upper bound -1e-12); a,b free
    lb = np.array([-np.inf, -np.inf, -np.inf])
    ub = np.array([np.inf, np.inf, -1e-12])

    ls = least_squares(resid, p0, bounds=(lb, ub), jac='2-point')  # robust & fast

    a, b, c = ls.x
    x0 = -b / (2.0 * c)
    plateau = (a + b * x0 + c * x0 ** 2)

    y_hat = qp_predict(ls.x, x)
    resid_raw = y - y_hat
    sse = float(np.sum(w * resid_raw ** 2))
    mse = sse / (n - 3)  # k=3 params (a,b,c)
    rmse = np.sqrt(mse)
    mae = float(np.mean(np.abs(resid_raw)))

    # R^2 (nonlinear): 1 - SSE/SST (weighted SST)
    y_bar = np.average(y, weights=w)
    sst = float(np.sum(w * (y - y_bar) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    r2_adj = 1.0 - (1 - r2) * (n - 1) / (n - 3 - 1) if n > 4 and np.isfinite(r2) else np.nan

    # AIC/BIC (Gaussian, using SSE)
    k = 3
    aic = n * np.log(sse / n) + 2 * k
    bic = n * np.log(sse / n) + k * np.log(n)

    # Parameter covariance via (J'J)^-1 * sigma^2, where J is Jacobian of residuals
    # SciPy returns J of residuals; build approx Fisher info
    J = ls.jac  # shape (n, k)
    # Weighted least squares already embedded in residuals, so use unweighted sigma^2 on residuals of resid()
    # Effective sigma^2:
    sigma2 = sse / (n - k)
    try:
        cov = inv(J.T @ J) * sigma2
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        cov = np.full((k, k), np.nan)
        se = np.array([np.nan, np.nan, np.nan])

    # t-stats & p-values (approx, using t with df=n-k)
    df = max(n - k, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        tvals = ls.x / se
        pvals = 2 * (1 - tdist.cdf(np.abs(tvals), df=df))

    params = {
        "a": a, "b": b, "c": c,
        "x_break": x0,
        "plateau": plateau
    }
    stderr = {"a": se[0], "b": se[1], "c": se[2]}
    tstats = {"a": tvals[0], "b": tvals[1], "c": tvals[2]}
    pvalues = {"a": pvals[0], "b": pvals[1], "c": pvals[2]}

    metrics = {
        "n": n, "k": k,
        "SSE": sse, "MSE": mse, "RMSE": rmse, "MAE": mae,
        "R2": r2, "R2_adj": r2_adj,
        "AIC": aic, "BIC": bic
    }

    return {
        "params": params,
        "stderr": stderr,
        "tstats": tstats,
        "pvalues": pvalues,
        "cov": cov,
        "metrics": metrics,
        "success": ls.success,
        "message": ls.message,
        # a convenient predictor
        "predict": lambda x_new: qp_predict(ls.x, np.asarray(x_new, dtype=float))
    }


import numpy as np


def consistent_optimum(res, x, model="qp", grid_n=4000, tol=1e-6, inverse_x=None):
    x = np.asarray(x, float)
    x_min, x_max = np.nanmin(x), np.nanmax(x)

    if model == "qp":
        x0 = float(res['params']['x_break'])
        if inverse_x is not None:
            x0 = float(inverse_x(x0))

        # Plateau value computed via the same predict function (inclusive)
        y0 = float(res['predict'](np.array([x0 + 1e-12]))[0])

        # Build grid that contains x0 exactly
        xg = np.linspace(x_min, x_max, int(grid_n))
        if (xg[0] > x0) or (xg[-1] < x0) or (np.all(np.abs(xg - x0) > 1e-12)):
            xg = np.sort(np.unique(np.concatenate([xg, [x0]])))

        yg = np.asarray(res['predict'](xg), float)

        # First x that achieves the max (earliest plateau)
        y_max = np.nanmax(yg)
        mask = np.isfinite(yg) & (yg >= y_max - tol)
        x_first = float(xg[np.argmax(mask)])

        info = {
            "model": "qp",
            "x_break_model": float(x0),
            "plateau": float(y0),
            "x_first_plateau_grid": x_first,
            "x_star": float(x0),  # canonical optimum
            "y_star": float(y0),
            "grid_step": (x_max - x_min) / max(grid_n - 1, 1),
            "aligned": abs(x_first - x0) <= max(tol, 1e-8),
        }
        return info

    elif model == "quad":
        # unchanged from your logic, but we can still return the first max
        a, b, c = (float(res['params'][k]) for k in ('a', 'b', 'c'))
        xg = np.linspace(x_min, x_max, int(grid_n))
        yg = np.asarray(res['predict'](xg), float)
        y_max = np.nanmax(yg)
        x_first = float(xg[np.argmax(yg >= y_max - tol)])
        if c < 0:
            x_vertex = -b / (2.0 * c)
            x_star = x_vertex if x_min <= x_vertex <= x_max else x_first
            y_star = float(res['predict'](np.array([x_star]))[0])
        else:
            x_star, y_star = x_first, float(y_max)
        return {
            "model": "quad",
            "x_vertex_analytic": float((-b / (2 * c)) if c != 0 else np.inf),
            "x_first_grid_max": x_first,
            "x_star": float(x_star),
            "y_star": float(y_star),
            "grid_step": (x_max - x_min) / max(grid_n - 1, 1),
        }

    else:
        raise ValueError("model must be 'qp' or 'quad'")


def x_zero(a, b, c):
    D = b * b - 4 * a * c
    if D < 0: return float('nan')  # no real crossing
    r1 = (-b - math.sqrt(D)) / (2 * c)
    r2 = (-b + math.sqrt(D)) / (2 * c)
    # smallest non-negative root
    roots = sorted([r for r in (r1, r2) if r >= 0])
    return roots[0] if roots else float('nan')


def plot_fitted_data(data, ylabel, filename=None):
    po = data.copy()
    # === facet grid ===
    g = sns.relplot(
        data=po,
        x="x",
        y="Actual",
        kind="scatter",
        col="Residue incorporation",
        col_wrap=2,
        s=18,
        alpha=0.6,
        color="red",
        aspect=1.5,
        facet_kws={"sharex": True, "sharey": True},
    )

    # === overlay fitted line + annotations per facet ===
    for label, ax in g.axes_dict.items():
        sub = (
            po.loc[po["Residue incorporation"] == label]
            .sort_values("x")
        )

        if len(sub):
            sns.lineplot(
                data=sub,
                x="x",
                y="Predicted",
                linewidth=2.5,
                color="green",
                ax=ax,
                label="Fitted line",
            )

        # dummy handle so APSIM shows in legend
        ax.scatter([], [], s=18, alpha=0.6, color="red", label="APSIM")

        # legend styling
        ax.legend(loc="lower right", frameon=False, fontsize=8)

        # panel annotation using your 'txt' column
        bg = ax.get_facecolor()
        txt = sub["txt"].iloc[0] if "txt" in sub and len(sub) else ""
        ax.text(
            0.4, 0.5, txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(facecolor=bg, alpha=0.75, edgecolor="none", pad=6),
            fontsize=12,
            family="DejaVu Sans Mono"
        )
        ax.legend(loc="lower right", frameon=False, fontsize=pl['legend_fontsize'])
    # === shared labels ===
    # remove individual y labels from all facets
    #########################################################
    for ax in g.axes.flatten():
        ax.set_ylabel("")

    # add ONE global y label, vertically centered for the whole figure
    #############################################################
    y_lab = ylabel
    g.fig.text(
        0.015,  # This pushes it to the left so it doesn't overlap y-ticks
        0.5,
        y_lab,
        rotation="vertical",
        va="center",
        ha="center",
        fontsize=18,
        # Optional: uncomment for crisp outline against ticks/bg
        # path_effects=[pe.withStroke(linewidth=3, foreground="white")],
    )

    # shared x label at bottom
    x_lab = LABELS.get("Nitrogen", "Nitrogen fertilizer (kg ha⁻¹)")
    g.set_xlabels(x_lab, fontsize=20)

    # === (optional) de-emphasize tick label darkness if you want contrast
    # for ax in g.axes.flatten():
    #     ax.tick_params(axis="y", labelsize=10, colors="0.4")

    # ====================== save the figure=======================
    fn = RESULTS / f"_fitted_yield.svg"
    fname  = filename or fn
    plt.savefig(fname, dpi=600, bbox_inches="tight")
    plt.savefig(fname, bbox_inches="tight")  # Windows vector



