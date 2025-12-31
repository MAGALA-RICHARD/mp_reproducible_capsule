import numpy as np
import pandas as pd
from scipy.stats import linregress
from simulate_scenario_data import datastore, table_name

import numpy as np
import pandas as pd
from scipy.stats import linregress
import numpy as np
import pandas as pd



def detect_equilibrium_local(
        df_treat,
        soc_col="SOC_0_15cm_Mg",
        year_col="year",
        window=10,

):
    """
    Detect the first durable SOC equilibrium for one treatment (Residue × N),
    using *two adjacent windows* of length `window`.

    Definition:
    - Window W1 (length `window` years) is considered "candidate equilibrium"
      if SOC is nearly flat:
        * slope(W1) between 0 and slope_tol
        * internal relative variation in W1 <= band_tol
    - Window W2 (the next `window` years immediately after W1) must:
        * exist (full length)
        * have mean SOC not higher than W1's mean
        * stay within roughly the same band relative to W1 mean
          (no breakout increase or drop > band_tol)

    If both are satisfied, we call equilibrium at the last year of W1.
    We return that year, plus the mean SOC of W1 as the equilibrium SOC.

    Returns dict:
        equilibrium_year
        equilibrium_SOC
        reached_equilibrium (bool)
        slope_window (slope in W1)
    """
    soc_at_start = df_treat[df_treat[year_col] == df_treat[year_col].min()][soc_col].iloc[0]
    df_treat[soc_col] = df_treat[soc_col] - soc_at_start

    d = df_treat.sort_values(year_col).copy()
    d[soc_col] = df_treat[soc_col].rolling(window=window, center=True, min_periods=1).mean()
    years = d[year_col].to_numpy()
    soc = d[soc_col].to_numpy()
    n = len(d)

    # Need at least two windows back-to-back: 2 * window years
    if n < 2 * window:
        return {
            "equilibrium_year": np.nan,
            "equilibrium_SOC": np.nan,
            "reached_equilibrium": False,
            "slope_window": np.nan,
        }

    # We'll slide an ending window W1 [start_idx : end_idx] of size `window`
    # and define W2 immediately after that
    for end_idx in range(window - 1, n - window):
        # n - window ensures W2 will fit after W1

        start_idx = end_idx - (window - 1)

        # --- window 1 (candidate equilibrium window)
        yr_w1 = years[start_idx:end_idx + 1]
        soc_w1 = soc[start_idx:end_idx + 1]
        max = df_treat[soc_col].max()
        if max in soc_w1:
            continue
        print(max, 'max')

        mva_soc1 = soc_w1.mean()

        # slope in W1
        slope, _, _, p_value1, _ = linregress(yr_w1, soc_w1)

        # internal variability in W1
        mean_w1 = soc_w1.mean()
        range_w1 = soc_w1.max() - soc_w1.min()
        cv = (soc_w1.std() / mean_w1) * 100

        print('cv', cv)

        rel_range_w1 = (range_w1 / mean_w1) if mean_w1 != 0 else np.inf

        # --- window 2 (immediately after W1)
        start_w2 = end_idx + 1
        end_w2 = end_idx + window
        yr_w2 = years[start_w2:end_w2 + 1]
        soc_w2 = soc[start_w2:end_w2 + 1]
        slope2, _, _, p_value2, _ = linregress(yr_w2, soc_w2)



        # must be full length
        if len(soc_w2) < window:
            continue

        mean_w2 = soc_w2.mean()
        range_w2 = soc_w2.max() - soc_w2.min()
        rel_range_w2 = (range_w2 / mean_w2) if mean_w2 != 0 else np.inf

        # Condition A: W2 is not on an upward trajectory relative to W1.
        # If W2 means >> W1 mean, that means a system is *still* accumulating.
        # You said equilibrium shouldn't still be building.


        # Condition B: W2 stays "near" the equilibrium band of W1.
        # We'll express both windows relative to W1's mean.

        # If we reach here, W1 is stable AND W2 confirms that stability persists.
        equil_year = yr_w1[-1]  # last year of W1
        equil_soc = mean_w1
        if p_value1 > 0.05 and cv < 5:
            ny = equil_year - df_treat[year_col].min()
            return {
                "equilibrium_year": equil_year,
                "equilibrium_SOC": equil_soc,
                'rate':equil_soc/ny,
                "reached_equilibrium": True,
                "slope_window": slope,
                'n_years':ny
            }

    # If we never satisfied criteria:
    return {
        "equilibrium_year": np.nan,
        "equilibrium_SOC": np.nan,
        "reached_equilibrium": False,
        "slope_window": np.nan,
    }


if __name__ == "__main__":
    from data_manager import read_db
    from itertools import product

    df = read_db(datastore, table_name)
    # pick one treatment, e.g. Residue = 37, Nrate = 180
    rn = list(product(df.R.unique(), df.N.unique()))
    res = []
    for r, n in rn:
        mask = (df["R"] == r) & (df["N"] == n)
        df_one = df.loc[mask].copy()

        eq_info = detect_equilibrium_local(
            df_one,
            soc_col="SOC_0_15cm_Mg",
            year_col="year",
            window=7,  #stability window

        )
        eq_info['N'] = n
        eq_info['R'] = r * 100
        res.append(eq_info)

    ans = pd.DataFrame(res)
    from xlwings import view

    view(ans)
    from utils import find_soc_equilibrium
    q= find_soc_equilibrium(df, group_cols=['N', 'R'], min_stable_years=5, eps_rel=0.002)
    #view(q)
