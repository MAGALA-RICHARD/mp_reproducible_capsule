import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils import merge_tables, read_db_table
from settings import RESULTS


def func(x, a, b, c):
    return a * np.exp(-b * x) + c


# Define the data to be fit with some noise:
xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
rng = np.random.default_rng()
y_noise = 0.2 * rng.normal(size=xdata.size)
ydata = y + y_noise

plt.plot(xdata, ydata, 'b-', label='data')
plt.savefig('dat.png')
os.startfile('dat.png')
# Fit for the parameters a, b, c of the function `func`:
popt, pcov = curve_fit(func, xdata, ydata)

popt

import numpy as np
from scipy.optimize import curve_fit


# ----- models -----
def quad(x, a, b, c):
    return a + b * x + c * x ** 2


def quad_plateau(x, a, b, c, xp):
    yq = quad(x, a, b, c)
    yp = a + b * xp + c * xp ** 2
    return np.where(x < xp, yq, yp)


# ----- helper: initial guess from plain quadratic -----
def quad_init(x, y):
    # polyfit returns [c, b, a] in descending powers
    c, b, a = np.polyfit(x, y, deg=2)
    x_star = -b / (2 * c) if c != 0 else np.median(x)
    return a, b, c, x_star


# ----- fit function -----
def fit_quadratic_plateau(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    a0, b0, c0, xp0 = quad_init(x, y)

    # bounds: c<0 for concavity; xp within observed x-range
    lo = [-np.inf, -np.inf, -np.inf, min(x)]
    hi = [np.inf, np.inf, 0.0, max(x)]
    p0 = [a0, b0, min(c0, -1e-9), np.clip(xp0, x.min(), x.max())]

    popt, pcov = curve_fit(quad_plateau, x, y, p0=p0, bounds=(lo, hi))
    a, b, c, xp = popt
    # simple diagnostics
    yhat = quad_plateau(x, *popt)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return popt, pcov, r2


# ----- example (replace with your data) -----
if __name__ == "__main__":
    df = merge_tables(str(RESULTS / 'single_model.db'), ['carbon', 'yield'])
    import numpy as np
    from scipy.optimize import least_squares

    import numpy as np
    from scipy.optimize import least_squares


    def _predict_qp(x, a, b, c):
        # concave quad up to xp, then flat
        xp = -b / (2 * c)
        yp = a + b * xp + c * xp ** 2
        yq = a + b * x + c * x ** 2
        yhat = np.where(x <= xp, yq, yp)
        return yhat, xp, yp


    def fit_qp_with_plateau_point(x, y, *, assume_sorted=False):
        """
        Fit quadratic-plateau y = a + b x + c x^2 (flat after vertex).
        Returns xp, yp, and the first OBSERVED x that sits on/after the plateau.
        """
        x = np.asarray(x, float);
        y = np.asarray(y, float)

        # Keep original indices but work sorted for robustness
        if assume_sorted:
            order = np.arange(x.size)
            xs, ys = x, y
        else:
            order = np.argsort(x)
            xs, ys = x[order], y[order]

        # initialise with plain quadratic
        a0, b0, c0 = np.polyfit(xs, ys, 2)
        if c0 >= 0:
            c0 = -1e-6  # enforce concavity

        def resid(p):
            a, b, c = p
            yhat, _, _ = _predict_qp(xs, a, b, c)
            return yhat - ys

        a, b, c = least_squares(resid, x0=[a0, b0, c0]).x
        yhat_s, xp, yp = _predict_qp(xs, a, b, c)

        # Metrics
        mse = float(np.mean((yhat_s - ys) ** 2))
        ss_res = float(np.sum((ys - yhat_s) ** 2))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # Where does the plateau START in the observed data?
        # (first x >= xp). If none, plateau not reached by observations.
        idx_plateau_sorted = np.searchsorted(xs, xp, side="left")
        plateau_reached = idx_plateau_sorted < xs.size
        if plateau_reached:
            x_obs_plateau = xs[idx_plateau_sorted]
            i_obs = order[idx_plateau_sorted]  # index in original order
            yhat_obs_plateau = yhat_s[idx_plateau_sorted]
        else:
            x_obs_plateau = np.nan
            i_obs = -1
            yhat_obs_plateau = np.nan

        return {
            # fitted parameters and metrics
            "a": a, "b": b, "c": c, "xp": xp, "yp": yp,
            "mse": mse, "r2": r2,
            # plateau info in observed data
            "plateau_reached": plateau_reached,
            "obs_index_at_plateau": int(i_obs),
            "x_obs_at_plateau": float(x_obs_plateau),
            "yhat_obs_at_plateau": float(yhat_obs_plateau),
            # useful masks/series if you want them
            "yhat_sorted": yhat_s,  # predictions in sorted-x order
            "x_sorted": xs,  # sorted x used for fit
            "order": order.tolist()  # mapping to original order
        }


    def optim(residue):
        data = df[df['Residue'] == residue].copy()
        out = fit_qp_with_plateau_point(data['N'], data['SOC_0_15CM'])

        print(f"Fitted plateau at xp={out['xp']:.1f}, yp={out['yp']:.2f}")
        print(f"MSE={out['mse']:.3f}, R^2={out['r2']:.3f}")

        if out["plateau_reached"]:
            print("Observed plateau begins at:")
            print(f"  x[{out['obs_index_at_plateau']}] = {out['x_obs_at_plateau']:.1f} "
                  f"(pred ≈ {out['yhat_obs_at_plateau']:.2f})")
        else:
            print("Plateau not reached within observed x-range.")
