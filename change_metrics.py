import numpy as np
import pandas as pd
from settings import SEED
from stat_module import dynamic_paired_ttest


def mean_diff_between_windows(
        df: pd.DataFrame,
        grouping,
        first_window: tuple = None,  # (start_year, end_year) inclusive
        last_window: tuple = None,  # (start_year, end_year) inclusive
        col: str = 'corn_yield_Mg',
        min_obs: int = 1,
        sample=False,
        sample_window: int = 30,
        sample_size=5
        # require at least this many observations in EACH window
):
    # ensure grouping is a list/tuple
    if isinstance(grouping, (str,)):
        grouping = [grouping]
    if not sample:
        assert isinstance(first_window, tuple), f"first_window must be a tuple if not sample is specified"
        assert isinstance(last_window, tuple), f"last_window must be a tuple if not sample is specified"
        y1s, y1e = map(int, first_window)
        y2s, y2e = map(int, last_window)

        # FIRST window: mean and count per group (inclusive years)
        f = (df[df['year'].between(y1s, y1e)]
             .groupby(grouping, dropna=False)[col]
             .agg(first_mean='mean', first_n='count')
             .reset_index())

        # LAST window: mean and count per group (inclusive years)
        l = (df[df['year'].between(y2s, y2e)]
             .groupby(grouping, dropna=False)[col]
             .agg(last_mean='mean', last_n='count')
             .reset_index())

    else:
        year_min, year_max = df['year'].min(), df['year'].max()
        y1s, y1e = map(int, [year_min, sample_window + year_min])
        first_years_window = year_min + sample_window
        y2s, y2e = (year_max - sample_window), year_max
        f_pool = df[df['year'].between(year_min, first_years_window)]
        last_years_window = year_max - sample_window
        l_pool = df[df['year'].between(last_years_window, year_max)]
        l_sample = l_pool.groupby(grouping).sample(n=sample_size, random_state=SEED).reset_index()
        f_sample = f_pool.groupby(grouping).sample(n=sample_size, random_state=SEED).reset_index()
        f = f_sample.groupby(grouping, dropna=False)[col].agg(first_mean='mean', first_n='count').reset_index()
        l = l_sample.groupby(grouping, dropna=False)[col].agg(last_mean='mean', last_n='count').reset_index()

    # Inner join to avoid NaNs from groups missing in either window
    out = f.merge(l, on=grouping, how='inner')

    # Enforce minimum observations in BOTH windows
    mask = (out['first_n'] >= min_obs) & (out['last_n'] >= min_obs)
    out = out.loc[mask].copy()

    # Compute difference: LAST − FIRST
    out[f'{col}@{y1s}-{y1e}'] = out.pop('first_mean')
    out[f'{col}@{y2s}-{y2e}'] = out.pop('last_mean')
    out[f'Δ{col}'] = out[f'{col}@{y2s}-{y2e}'] - out[f'{col}@{y1s}-{y1e}']
    out[f'%Δ{col}'] = (out[f'Δ{col}'] / out[f'{col}@{y1s}-{y1e}']) * 100
    ptest = dynamic_paired_ttest(out[f'{col}@{y2s}-{y2e}'], out[f'{col}@{y1s}-{y1e}'])
    from pprint import pprint
    print(f"T- test statistics for {col}")
    pprint(ptest, indent=4, )
    # (Optional) drop the count columns if you don’t need them
    # out = out.drop(columns=['first_n','last_n'])

    return out


def compute_last_minus_first_change(
        data: pd.DataFrame,
        grouping,
        col='SOC_0_15cm_Mg',
        how='inner',  # 'inner' -> only groups present in both years; 'outer' -> keep all (may produce NaN)
        float_round=None  # e.g., 6 -> round float grouping cols to 6 dp to avoid 0.5 vs 0.500000
):
    """
    Compute the change, especially with the soil organic carbon between the last and first year to generate the balance at the end of the simulation.
    :param data: pd.dataframe containing the data.
    :param grouping: grouping e.g by Nitrogen or residue or both.
    :param col: str column name to compute.
    :param how: merging sytle
    :param float_round: not used yet
    :return:
    """
    df = data.copy()

    # choose years
    first = int(df['year'].min())
    last = int(df['year'].max())

    if isinstance(grouping, str):
        gp = [grouping]
    else:
        gp = list(grouping)
    df[gp] = df[gp].astype('str')
    # build a pivot so both years share the same index
    pv = df.pivot_table(
        index=grouping,
        columns='year',
        values=col,
        aggfunc='mean',
        dropna=False,  # keep groups even if some have NaN
    )

    df2 = df[df['year'].astype(int) == last]
    df1 = df[df['year'].astype(int) == first]
    assert len(df1) == len(
        df2), f'soc changes can not be performed varying length between maximum and minimum years: {len(df1), len(df2)}'
    # guard for missing year columns
    if first not in pv.columns or last not in pv.columns:
        raise ValueError(
            f"Requested years not present in data: first={first} in {list(pv.columns)}, last={last} in {list(pv.columns)}")

    # pick join style on groups
    s_first = pv[first]
    s_last = pv[last]
    if how == 'inner':
        common = s_first.dropna().index.intersection(s_last.dropna().index)
        s_first = s_first.reindex(common)
        s_last = s_last.reindex(common)
    # else 'outer' -> keep all; NaN expected where a group is missing

    out = pd.DataFrame({
        f'{col}@{first}': s_first,
        f'{col}@{last}': s_last
    })
    out[f'Δ{col}'] = out[f'{col}@{last}'] - out[f'{col}@{first}']
    out[f'%Δ{col}'] = (out[f'Δ{col}'] / out[f'{col}@{first}']) * 100
    out = out.reset_index()

    return out


import pandas as pd
import numpy as np


def find_equilibrium(df_treat,
                     soc_col='SOC_0_15cm_Mg',
                     year_col="year",
                     w=5,
                     eps=0.05,
                     k=5):
    """
    df_treat = data for ONE combo of (Residue, Nrate)
    Returns: dict with equilibrium_year, equilibrium_SOC, reached_equilibrium(bool)
    """

    d = df_treat.sort_values(year_col).copy()

    # rolling mean (trailing window of length w)
    d["SOC_smooth"] = d[soc_col].rolling(window=w, min_periods=w).mean()

    # first difference of smoothed SOC
    d["dSOC_dt"] = d["SOC_smooth"].diff()

    # we only start checking after we have both rolling mean and diff
    # i.e. after first w years
    d_valid = d.dropna(subset=["SOC_smooth", "dSOC_dt"]).copy()

    # absolute change
    d_valid["abs_change_ok"] = (d_valid["dSOC_dt"].abs() < eps).astype(int)

    # we now want first index where we have k consecutive years of abs_change_ok == 1
    # create a rolling sum over that boolean
    d_valid["stable_run"] = (
        d_valid["abs_change_ok"]
        .rolling(window=k, min_periods=k)
        .sum()
    )

    # find first row where stable_run == k
    hit = d_valid.index[d_valid["stable_run"] == k]

    if len(hit) == 0:
        # never stabilized under this eps/k
        return {
            "equilibrium_year": np.nan,
            "equilibrium_SOC": np.nan,
            "reached_equilibrium": False
        }

    first_hit_idx = hit[0]

    equil_year = d_valid.loc[first_hit_idx, year_col]

    # take SOC_smooth averaged over that stable window
    window_idxs = d_valid.loc[first_hit_idx - k + 1:first_hit_idx].index
    equil_soc = d_valid.loc[window_idxs, "SOC_smooth"].mean()

    return {
        "equilibrium_year": equil_year,
        "equilibrium_SOC": equil_soc,
        "reached_equilibrium": True
    }


def find_equilibrium_tail(
        df_treat,
        soc_col="SOC_0_15cm_Mg",
        year_col="year",
        window_years=20,
        slope_tol=0.01,  # Mg C ha^-1 per year considered "flat"
):
    """
    Estimate long-term SOC equilibrium for one (Residue, Nrate) treatment
    by testing whether SOC has stabilized over the final `window_years`
    of the simulation.

    Logic:
    - Take last `window_years` of data.
    - Fit SOC ~ year (simple linear regression).
    - If |slope| < slope_tol, declare quasi-equilibrium.
    - Equilibrium SOC = mean SOC in that tail window.
    - Equilibrium year = mean year (or last year) in that window.

    Returns:
        dict with:
            equilibrium_year
            equilibrium_SOC
            reached_equilibrium (bool)
            slope_last_window (for transparency)
    """

    d = df_treat.sort_values(year_col).copy()

    # guard: not enough years
    if d.shape[0] < window_years:
        window_years = d.shape[0]

    tail = d.iloc[-window_years:].copy()

    # Fit linear regression SOC = a + b*year
    x = tail[year_col].to_numpy().astype(float)
    y = tail[soc_col].to_numpy().astype(float)

    # slope b from least squares
    # b = Cov(x,y)/Var(x)
    x_mean = x.mean()
    y_mean = y.mean()
    var_x = ((x - x_mean) ** 2).sum()
    if var_x == 0:
        # all same year? weird but handle it
        slope = 0.0
    else:
        cov_xy = ((x - x_mean) * (y - y_mean)).sum()
        slope = cov_xy / var_x  # Mg C ha^-1 per year

    # check if slope is "basically zero"
    if abs(slope) < slope_tol:
        # considered equilibrated
        equil_soc = y.mean()
        # you can report the last year or the midpoint of the tail
        equil_year = x.max()
        return {
            "equilibrium_year": equil_year,
            "equilibrium_SOC": equil_soc,
            "reached_equilibrium": True,
            "slope_last_window": slope,
        }
    else:
        # still trending, no true equilibrium in horizon
        return {
            "equilibrium_year": np.nan,
            "equilibrium_SOC": np.nan,
            "reached_equilibrium": False,
            "slope_last_window": slope,
        }
