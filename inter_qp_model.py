import numpy as np
from numpy.linalg import inv
from scipy.optimize import least_squares
from scipy.stats import t as tdist
import math
from pathlib import Path
import pandas as pd
import seaborn as sns
from data_manager import read_db
from labels import LABELS
from logger import get_logger
from simulate_quadratic_fit_data import datastore, table_name
import os

sns.set_style("darkgrid", rc={"font.family": "DejaVu Sans"})
# pd.plotting.register_matplotlib_converters()


from scipy.optimize import least_squares
from numpy.linalg import inv
from scipy.stats import t as tdist

script_name = Path(__file__).stem
logger = get_logger(name=f'{script_name}')

logger.info("finding the optimum N fertilizer for SOC sequestration under interactive eqp model\n "
            "===========================================================================")


def fit_qp_interactive(N, R, y, weights=None):
    """
    Quadratic-Plateau with interaction:
        y = a + b1*N + b2*R + b3*N^2 + b4*(N*R)  for N <= N0(R)
        y = plateau(R)                           for N >= N0(R)
        N0(R) = -(b1 + b4*R)/(2*b3),  with b3 < 0
    Returns dict with params, SE, p, and metrics including MSE, MAE, RMSE, RRMSE, R2.
    """
    import numpy as np
    from numpy.linalg import inv
    from scipy.optimize import least_squares
    from scipy.stats import t as tdist

    N, R, y = map(lambda a: np.asarray(a, float), (N, R, y))
    m = np.isfinite(N) & np.isfinite(R) & np.isfinite(y)
    N, R, y = N[m], R[m], y[m]
    n = len(y)
    if n < 8:
        raise ValueError("Need >= 8 points for interactive QP fit")

    # weights (normalized for stability)
    w = np.ones_like(y) if weights is None else np.asarray(weights, float)[m]
    w = w / np.nanmean(w)

    # ---- OLS start (quadratic with interaction) ----
    X = np.c_[np.ones(n), N, R, N ** 2, N * R]
    a0, b10, b20, b30, b40 = np.linalg.lstsq(X, y, rcond=None)[0]
    if b30 >= -1e-6:
        b30 = -abs(b30) - 1e-3
    p0 = np.array([a0, b10, b20, b30, b40])

    def predict(p, N_, R_):
        a, b1, b2, b3, b4 = p
        N0 = -(b1 + b4 * R_) / (2 * b3)
        base = a + b1 * N_ + b2 * R_ + b3 * N_ ** 2 + b4 * (N_ * R_)
        plateau = a + b1 * N0 + b2 * R_ + b3 * N0 ** 2 + b4 * (N0 * R_)
        return np.where(N_ <= N0, base, plateau)

    def resid(p):
        return np.sqrt(w) * (y - predict(p, N, R))

    lb = np.array([-np.inf] * 5)
    ub = np.array([np.inf, np.inf, np.inf, -1e-12, np.inf])  # b3 < 0
    ls = least_squares(resid, p0, bounds=(lb, ub), jac="2-point")

    # --- post fit ---
    a, b1, b2, b3, b4 = ls.x
    y_hat = predict(ls.x, N, R)
    err = y - y_hat

    # Weighted sums (use sum(w)=n_eff for stable averages)
    wsum = float(np.sum(w))
    sse_w = float(np.sum(w * err ** 2))
    mae_w = float(np.sum(w * np.abs(err))) / wsum
    mse_w = sse_w / wsum
    rmse_w = np.sqrt(mse_w)

    ybar_w = float(np.sum(w * y)) / wsum
    sst_w = float(np.sum(w * (y - ybar_w) ** 2))
    r2 = 1.0 - (sse_w / sst_w) if sst_w > 0 else np.nan
    r2_adj = 1.0 - (1 - r2) * (n - 1) / max(n - 5 - 1, 1) if np.isfinite(r2) else np.nan
    rrmse_pct = (rmse_w / ybar_w) * 100.0 if ybar_w != 0 else np.nan

    # AIC/BIC (Gaussian; use weighted SSE with n as effective count)
    k = 5
    aic = n * np.log(sse_w / n) + 2 * k
    bic = n * np.log(sse_w / n) + k * np.log(n)

    # SE / p-values from Jacobian
    J = ls.jac
    sigma2 = sse_w / max(n - k, 1)
    try:
        cov = inv(J.T @ J) * sigma2
        se = np.sqrt(np.diag(cov))
    except Exception:
        cov = np.full((k, k), np.nan)
        se = np.full(k, np.nan)
    from scipy.stats import t as tdist
    tvals = ls.x / se
    pvals = 2 * (1 - tdist.cdf(np.abs(tvals), df=max(n - k, 1)))

    return {
        "params": {"a": a, "b1": b1, "b2": b2, "b3": b3, "b4": b4},
        "stderr": dict(zip(["a", "b1", "b2", "b3", "b4"], se)),
        "pvalues": dict(zip(["a", "b1", "b2", "b3", "b4"], pvals)),
        "metrics": {
            "MSE": mse_w,
            "MAE": mae_w,
            "RMSE": rmse_w,
            "RRMSE_%": rrmse_pct,
            "R2": r2,
            "R2_adj": r2_adj,
            "AIC": aic,
            "BIC": bic
        },
        "predict": lambda Nnew, Rnew: predict(ls.x, np.asarray(Nnew, float), np.asarray(Rnew, float)),
        "success": ls.success,
        "message": ls.message,
    }


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_soc_surface_with_optima(N, R, y, fit_results, resolution=50):
    """
    Plots the 3D surface of SOC balance with overlaid residue-specific optima and plateaus.

    Parameters:
        N (array-like): Nitrogen rates
        R (array-like): Residue incorporation levels
        y (array-like): Observed SOC balance
        fit_results (dict): The result from fit_qp_interactive with params and predict method
        resolution (int): Resolution for grid plotting
    """
    # Create a mesh grid for plotting
    N_grid, R_grid = np.meshgrid(
        np.linspace(np.min(N), np.max(N), resolution),
        np.linspace(np.min(R), np.max(R), resolution)
    )

    # Use the predict function to get the predicted SOC values over the grid
    predicted_soc = fit_results["predict"](N_grid.ravel(), R_grid.ravel())
    predicted_soc = predicted_soc.reshape(N_grid.shape)

    # Get parameters from the fit
    params = fit_results["params"]
    a, b1, b2, b3, b4 = params["a"], params["b1"], params["b2"], params["b3"], params["b4"]

    # Compute optima N0(R) for each R
    N_optima = -(b1 + b4 * R_grid) / (2 * b3)  # N0(R)
    print("N_optima: ", N_optima.mean())

    # Plotting the 3D surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(R_grid, N_grid, predicted_soc, cmap='viridis', alpha=0.7)

    # Overlay the residue-specific optima N0(R)
    ax.plot_surface(N_optima, R_grid, predicted_soc, color='r', alpha=0.5, label="Optima N0(R)", linewidth=0)

    # Add labels
    ax.set_xlabel('Nitrogen Rate (kg/ha)')
    ax.set_ylabel('Residue Incorporation')
    ax.set_zlabel('SOC Balance')
    ax.set_title('SOC Balance vs Nitrogen × Residue with Optima')

    # Optional: Plot the observed data points (scatter)
    ax.scatter(N, R, y, color='k', label='Observed Data', s=50, alpha=0.7)

    # Add a legend
    ax.legend()

    plt.tight_layout()
    plt.savefig('f.png', dpi=600)
    os.startfile('f.png')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_predicted_observed(
        df,
        model_results,
        x_col="Nitrogen",  # column to use on the x-axis
        y_col="SOC1",  # observed response column
        r_col="Residue",  # residue column (used if model needs it)
        title=None,
        save=None,  # None, path-like, or directory; if directory, auto-name file
        show=True
):
    """
    Plot observations as points and model predictions as a smooth line vs `x_col`.

    Works with models that expose model_results["predict"] as either:
        predict(N) # 1D predictor
        predict(N, R) # 2D predictor (interactive model)

    Parameters
    ----------
    df : pd.DataFrame
    model_results : dict        # return dict from your fit function (must have "predict", optionally "metrics")
    x_col : str
    y_col : str
    r_col : str
    title : str or None
    save : str|Path|None        # if a directory, auto-generates filename; if a file path, saves there
    show : bool
    """

    # ---- 1) Coerce columns to float (robust to strings) ----
    def _to_float(s):
        s = np.asarray(s)
        try:
            return s.astype(float)
        except (ValueError, TypeError):
            return np.array([np.nan if (str(v).strip() in {"", "nan", "None"}) else float(v) for v in s], dtype=float)

    if x_col in df.columns:
        N = _to_float(df[x_col])
    elif "N" in df.columns:
        N = _to_float(df["N"])
        x_col = "N"
    else:
        raise KeyError(f"Could not find x column '{x_col}' (or fallback 'N') in df")

    y_obs = _to_float(df[y_col])
    R = _to_float(df[r_col]) if r_col in df.columns else None

    m = np.isfinite(N) & np.isfinite(y_obs)
    if R is not None:
        m &= np.isfinite(R)
    N, y_obs = N[m], y_obs[m]
    if R is not None:
        R = R[m]

    # ---- 2) Get predictions (support both signatures) ----
    predict = model_results["predict"]
    try:
        # try interactive: predict(N, R)
        if R is None:
            raise TypeError("Residue column missing for interactive model.")
        y_hat = predict(N, R)
    except TypeError:
        # fallback: 1D predict(N)
        y_hat = predict(N)

    # ---- 3) Build plotting frame (sorted for smooth line) ----
    d = pd.DataFrame({"x": N, "R":R, "Actual": y_obs, "Predicted": y_hat}).sort_values("x")
    d['R'] = pd.Categorical(d['R'], ordered=True)

    # ---- 4) Compute quick metrics (fallback if not provided) ----
    if "metrics" in model_results and "R2" in model_results["metrics"]:
        r2 = model_results["metrics"]["R2"]
        rmse = model_results["metrics"].get("RMSE", float(np.sqrt(np.mean((y_obs - y_hat) ** 2))))
        rrmse = model_results["metrics"].get("RRMSE_%",
                                             (rmse / np.mean(y_obs)) * 100 if np.mean(y_obs) != 0 else np.nan)
    else:
        sse = float(np.sum((y_obs - y_hat) ** 2))
        sst = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
        r2 = 1.0 - sse / sst if sst > 0 else np.nan
        rmse = float(np.sqrt(np.mean((y_obs - y_hat) ** 2)))
        rrmse = (rmse / np.mean(y_obs)) * 100 if np.mean(y_obs) != 0 else np.nan

    # ---- 5) Plot ----
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sns.scatterplot(data=d, x="x", y="Actual", s=18, alpha=0.6,  color="tab:red", ax=ax, hue='R')
    sns.lineplot(data=d, x="x", y="Predicted", linewidth=2.2, label="Fitted", color="tab:blue", ax=ax)

    ax.grid(True, alpha=0.25)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    # if title is None:
    #     ax.set_title(f"Observed vs Fitted ({y_col} ~ {x_col})   "
    #                  f"R²={r2:.3f}, RMSE={rmse:.3g}, RRMSE={rrmse:.2f}%")
    # else:
    title = title or ''
    ax.set_title(title)

    ax.legend()

    plt.tight_layout()

    # ---- 6) Save if requested ----

    fname = f"obs_fit_{y_col}_by_{x_col}.png"

    fig.savefig(fname, dpi=300, bbox_inches="tight")

    if show:
        try:
            plt.show()
        except Exception as e:
            os.startfile(fname)
    else:
        plt.close(fig)

    return {"figure": fig, "ax": ax, "metrics": {"R2": r2, "RMSE": rmse, "RRMSE_%": rrmse}}


if __name__ == "__main__":
    from change_metrics import compute_last_minus_first_change

    rdata = read_db(datastore, table_name)
    data = compute_last_minus_first_change(rdata, grouping=['Nitrogen', 'Residue'])

    res = fit_qp_interactive(
        N=data["Nitrogen"],
        R=data["Residue"],
        y=data["ΔSOC_0_15cm_Mg"]
    )

    print(res["params"])

    # Example usage:
    # Assuming you have fitted the model and the `fit_results` dictionary is returned
    fit_results = fit_qp_interactive(N=data["Nitrogen"], R=data["Residue"], y=data["ΔSOC_0_15cm_Mg"])
    plot_soc_surface_with_optima(data["Nitrogen"].astype(float), data["Residue"].astype(float), data["ΔSOC_0_15cm_Mg"],
                                 fit_results)

    plot_predicted_observed(data,
                            fit_results,
                            x_col="Nitrogen",  # column to use on the x-axis
                            y_col="ΔSOC_0_15cm_Mg",  # observed response column
                            r_col="Residue",  # residue column (used if model needs it)
                            show=True)
