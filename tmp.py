import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import seaborn as sns
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from plotting import LABELS
from logger import get_logger

sns.set_style("darkgrid", rc={"font.family": "DejaVu Sans"})
# pd.plotting.register_matplotlib_converters()

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from numpy.linalg import inv
from scipy.stats import t as tdist

logger = get_logger()

logger.info("finding the optimum N fertilizer for SOC sequestration\n ==========================================")
def fit_quadratic_plateau(x, y, weights=None):
    """
    Quadratic-Plateau model:
      y = a + b x + c x^2               for x <= x0,  with c < 0
      y = plateau = a + b x0 + c x0^2   for x >= x0,  where x0 = -b / (2c)

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
            "x_star": float(x0),      # canonical optimum
            "y_star": float(y0),
            "grid_step": (x_max - x_min) / max(grid_n - 1, 1),
            "aligned": abs(x_first - x0) <= max(tol, 1e-8),
        }
        return info

    elif model == "quad":
        # unchanged from your logic, but we can still return the first max
        a, b, c = (float(res['params'][k]) for k in ('a','b','c'))
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
            "x_vertex_analytic": float((-b/(2*c)) if c != 0 else np.inf),
            "x_first_grid_max": x_first,
            "x_star": float(x_star),
            "y_star": float(y_star),
            "grid_step": (x_max - x_min) / max(grid_n - 1, 1),
        }

    else:
        raise ValueError("model must be 'qp' or 'quad'")
def x_zero(a,b,c):
    D = b*b - 4*a*c
    if D < 0: return float('nan')  # no real crossing
    r1 = (-b - math.sqrt(D)) / (2*c)
    r2 = (-b + math.sqrt(D)) / (2*c)
    # smallest non-negative root
    roots = sorted([r for r in (r1, r2) if r >= 0])
    return roots[0] if roots else float('nan')
if __name__ == '__main__':
    from utils import find_soc_equilibrium, calculate_soc_changes, RESULTS, merge_tables, open_file

    opt_params = []
    dfm = merge_tables(data=RESULTS / 'one_single_split_single.db', tables=['carbon', 'yield'])
    dfm['Residue'] = pd.Categorical(dfm['Residue'], ordered=True)
    y_min = dfm[dfm['year'] == dfm['year'].min()]['SOC1'].iloc[0]
    dfm['SOC1'] = dfm['SOC_0_15CM'] - float(y_min)
    # x = Nitrogen (kg/ha), y = SOC (e.g., Mg C/ha)
    for residue in sorted(dfm.Residue.unique()):
        df = dfm[dfm['Residue'] == residue].groupby('N')['SOC1'].mean().reset_index()
        res = fit_quadratic_plateau(x=df['N'], y=df['SOC1'])

        print(res['params'])  # a, b, c, x_break, plateau
        print(res['metrics'])  # R2, R2_adj, AIC, BIC, RMSE, etc.

        # Predict and (optionally) plot
        xg = np.linspace(df['N'].min(), df['N'].max(), 200)
        yg = res['predict'](xg)

        import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

        x_obs = df['N'].to_numpy(float)
        y_obs = df['SOC1'].to_numpy(float)
        y_hat = res['predict'](x_obs)
        # y_hat = df['SOC1'].to_numpy(float)

        d = pd.DataFrame({'x': x_obs, 'Actual': y_obs, 'Predicted': y_hat})

        ax = sns.scatterplot(data=d, x='x', y='Actual', s=18, alpha=0.6, label='APSIM', color='red')
        sns.lineplot(data=d.sort_values('x'), x='x', y='Predicted', linewidth=2, label='Fitted', ax=ax)

        ax.set_xlabel(LABELS.get('Nitrogen'))
        ax.set_ylabel(LABELS.get('SOC1'))
        sns.despine()
        # plt.title(residue)
        # --- Add QP formula & fit stats (ASCII-safe, no LaTeX needed) ---
        a = float(res["params"]["a"])
        b = float(res["params"]["b"])
        c = float(res["params"]["c"])
        x0 = float(res["params"]["x_break"])
        plat = float(res["params"]["plateau"])
        r2 = float(res["metrics"]["R2"])
        rmse = float(res["metrics"]["RMSE"])
        logger.info(f"Residue `{residue}`:{x_zero(a,b,c)}")


        # Snap tiny values to zero to avoid ugly -0.000 strings
        def z(v, eps=1e-12):
            return 0.0 if abs(v) < eps else v


        a, b, c = z(a), z(b), z(c)

        try:
            test = "x\u00b2"
            assert ax.figure.canvas.get_renderer()  # forces font load in some backends
            use_sup2 = True
        except Exception:
            use_sup2 = False

        line1 = (
                f"y = {a:.3g}{b:+.3g}N{c:+.3g}" + ("N\u00b2" if use_sup2 else "N^2")
        )
        rsq = ("R\u00b2" if use_sup2 else "R^2")
        use_sub = True
        xo = ("N\u2080" if use_sub else "N_0")
        line2 = f"{xo} = {x0:.3g}; plateau = {plat:.3g}; {rsq} = {r2:.3f}; RMSE = {rmse:.3g}"
        txt = line1 + "\n" + line2 + "\n" + f"for {float(residue) * 100}% residue incorporation"
        plt.legend(frameon=False)
        bg = ax.get_facecolor()
        ax.text(
            0.2, 0.45, txt,
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(facecolor=bg, alpha=0.75, edgecolor="none", pad=6),
            fontsize=9, family="DejaVu Sans Mono"  # mono keeps signs/spacing tidy
        )

        plt.tight_layout()

        fname = RESULTS / f'{residue}-qp.png'
        plt.savefig(fname, dpi=600)
        open_file(fname)
        plt.close()
        argmax = int(np.argmax(y_hat))
        print(f"{residue}: {x_obs[argmax]}")
        qop = consistent_optimum(res, x_obs, model='qp')
        qop['Residue'] = residue
        logger.info(qop)
        opt_params.append(qop)
    dp = pd.DataFrame(opt_params)
    logger.info("\n%s", dp.to_csv(index=False))
    print(dp)
