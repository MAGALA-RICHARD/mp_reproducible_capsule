from __future__ import annotations

import os.path
from functools import cache
import seaborn as sns
from apsimNGpy.core.apsim import ApsimModel
from apsimNGpy.core.pythonet_config import is_file_format_modified
from apsimNGpy.core_utils.database_utils import delete_table
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sqlalchemy.exc import NoSuchTableError

from read_met import read_apsim_met
from settings import Path, path_to_MP_data, scratch_DIR, met_file

if is_file_format_modified():
    pass
else:
    pass
import numpy as np
import pandas as pd
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent
Y_FONTSIZE = 18
X_FONTSIZE = 18
TIMING_RATES = '1'
from pathlib import Path
import sys
import subprocess
import os

RESULTS = BASE_DIR / 'Results'
xc1 = [549.93, 1.98, 287.12, 139.31, 24.81, 17.64, 100, 700, 0.5, 0.03, ]
xc2 = [574.82, 2.0, 291.62, 91.44, 29.54, 15.86, 103.5, 694.76, 0.45, 0.03]  # the trend is reasonable
xc = [526.36, 1.84, 289.83, 136.11, 26.63, 19.33, 93.39, 705.3, 0.54, 0.04]


def open_file(file):
    p = Path(file)

    if not p.suffix:
        raise ValueError("suffix is required")
    if not p.is_file():
        raise ValueError("File does not exist")

    path = str(p.resolve())

    if sys.platform.startswith("win"):
        # 'start' is a shell builtin; simplest is os.startfile on Windows
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.run(["open", path], check=True)
    else:
        # Linux/BSD
        subprocess.run(["xdg-open", path], check=True)


def read_or_run(apsim_file, finert=0.65):
    """
    Read an APSIM file run and returns the results
    :param finert: change stable carbon fractions
    :param apsim_file: name of the apsim file. should be in the path_to_MP_data
    :return:
    """
    p = Path(apsim_file)
    if Path(apsim_file).is_dir():
        raise ValueError("only file name is allowed")
    if not p.suffix == '.apsimx':
        raise ValueError(f" did you mean'{apsim_file}.apsimx'?")
    file_name = path_to_MP_data / apsim_file
    out_new_name = scratch_DIR / f"out_{apsim_file}"
    in_apsim_file = path_to_MP_data / apsim_file
    if not os.path.isfile(in_apsim_file):
        raise ValueError(f"{in_apsim_file} does not exist on the computer")

    model = ApsimModel(str(in_apsim_file), out_path=out_new_name)
    model.edit_model(model_type='Weather', model_name='Weather', met_file=str(met_file))
    model.edit_model(model_type='Models.Soils.Organic', model_name='Organic', FInert=finert)
    return model


@cache
def base(method):
    match method.lower():
        case 'single':
            base_file = 'base_single.apsimx'

        case 'split':
            base_file = 'base_split.apsimx'

        case 'auto' | 'automatic':
            base_file = 'base_auto.apsimx'
        case _:
            raise ValueError(f"Invalid `{method}` not supported/implemented by utils.create_experiment function")
    model_path = path_to_MP_data / base_file
    out_path = f'out_{base_file}'
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} is not found at{path_to_MP_data}. Perhaps was deleted")

    _model = ApsimModel(model_path, out_path=out_path)
    _model.edit_model(model_type='Models.Soils.Organic', model_name='Organic', FInert=[0.55, 0.90],
                      FBiom=[0.045, 0.040],
                      Carbon=[1.8, 1.2])
    pp = Path(_model.path)
    c_csv = pp.parent / f"{pp.stem}.carbon.csv"
    # c_csv.unlink(missing_ok=True)
    y_csv = pp.parent / f"{pp.stem}.yield.csv"
    # y_csv.unlink(missing_ok=True)
    db = pp.with_suffix('.db')
    # for rep in {'carbon', 'yield'}:
    #     delete_table(db, rep)

    return _model


AUTO = object()


@cache
def create_experiment(base_file, lonlat=None, start=1904, end=2005, out_path=None, site='morrow plots',
                      bin_path: str | Path | AUTO = AUTO):
    """

    :param base_file: base file in directory in `APSIMX FILES`
    :param lonlat:
    :param start:
    :param end: end
    :param out_path:
    :return:
    """
    logger.info(f"Creating base experiment from {base_file}\n which start in {start} and end in: {end} for site {site}")
    if bin_path is not AUTO:
        from apsimNGpy.core.config import set_apsim_bin_path
        set_apsim_bin_path(bin_path)

    cfg = load_manifest()
    paths = cfg['paths']
    nwrec_dir = paths['apsimx_base_dir']

    if site == 'morrow plots':
        dir_path = path_to_MP_data
    else:
        dir_path = Path(__file__).parent / Path(nwrec_dir)
    logger.info(f'Root of the apsimx file: {dir_path}',)
    model_path = dir_path / base_file
    model_path = model_path.with_suffix('.apsimx').resolve()
    out_path = Path(out_path or f'out_{base_file}').resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} \nis not found at {str(path_to_MP_data)}.\n Perhaps was deleted")
    if is_file_format_modified():

        from apsimNGpy.core.experimentmanager import ExperimentManager as Experiment
        _model = Experiment(model=model_path, out_path=out_path)
        _model.init_experiment(permutation=True)

    else:
        from apsimNGpy.core.apsim import ApsimModel
        _model = ApsimModel(model_path, out_path=out_path)

        _model.create_experiment(permutation=True, base_name='changeFert_verity')
    og = _model.inspect_model('Organic', 'Organic')

    # edit model
    # remove 0.65, 0.90  and fbiom 0.035, 0.040
    if lonlat:
        _model.get_soil_from_web(lonlat=lonlat, thinnest_layer=150, max_depth=2600)
    orga = _model.inspect_model_parameters(model_type='Models.Soils.Organic', model_name='Organic')

    if base_file == 'no_till':
        _model.edit_model(model_type='Models.Soils.Organic', model_name='Organic', FInert=[0.65, 0.91],
                          FBiom=[0.045, 0.045],
                          Carbon=[1.8, 1.2])
        _model.edit_model(model_type='Models.Soils.Physical', model_name='Physical', BD=1.40)
    else:
        pass
        _model.save()
        _model.edit_model(model_type='Models.Soils.Organic', model_name='Organic', FInert=[0.55, 0.90],
                          FBiom=[0.035, 0.55],
                          Carbon=[2.4, 2.0])  # higher initial carbon all resulted into negative carbon balance
    # if method == 'split':
    _model.edit_model(model_type='Models.Surface.SurfaceOrganicMatter', model_name='SurfaceOrganicMatter',
                      InitialCNR=100)
    pp = Path(_model.path)
    c_csv = pp.parent / f"{pp.stem}.carbon.csv"
    c_csv.unlink(missing_ok=True)
    y_csv = pp.parent / f"{pp.stem}.yield.csv"
    y_csv.unlink(missing_ok=True)
    db = pp.with_suffix('.db')
    for rep in {'carbon', 'yield', 'mineralization', 'daily', 'water'}:
        try:
            delete_table(db, rep)
        except (NoSuchTableError, Exception):
            pass

    wf = weather_file = BASE_DIR / 'mets/urbana-mp-20250916.met'
    # _model.replace_met_file(weather_file=wf)
    wf = _model.inspect_model_parameters('Models.Climate.Weather', 'Weather')
    if lonlat:
        start, end = 1984, 2022
        _model.get_weather_from_web(lonlat=lonlat, start=start, end=end)
        wf = _model.inspect_model_parameters('Models.Climate.Weather', 'Weather')
    else:
        start, end = start, end
    _model.edit_model(model_type='Models.Clock', model_name='Clock', start=f"{start}-01-01", end=f"{end}-12-31")
    weather_file = _model.inspect_model_parameters('Models.Climate.Weather', 'Weather')
    # df_met = read_apsim_met(weather_f)
    # df_met, _, _ = df_met

    # if Path(wf).name != Path(weather_file).name:
    #     start, end = 1984, 2022
    #     print(start)
    #     print(end)
    #     _model.edit_model(model_type='Models.Clock', model_name='Clock', start=f"{start}-01-01", end=f"{end}-12-31")
    # # print(f"successfully updated weather path to {wf}")
    # # print(start)

    _model.save()

    return _model


def plot_mva(carbon, column, x='year', hue='Nitrogen', col='Residue', style=None,
             errorbar=None, color_palette='deep', ylabel=None, xlabel=None,
             xtick_size=12, ytick_size=None, ylabel_size=Y_FONTSIZE, xlabel_size=X_FONTSIZE):
    if not column.endswith('_roll_mean'):
        column = column + '_roll_mean'
    ylabel = ylabel or column
    xlabel = xlabel or x
    ytick_size = ytick_size or xtick_size

    g = sns.relplot(
        data=carbon, x=x, y=column,
        hue=hue, style=style, col=col,
        kind="line", col_wrap=2,
        linewidth=1.5,
        errorbar=errorbar, palette=color_palette,
        height=6, aspect=1.4
    )

    g.set_axis_labels("", "")

    # Enforce tick sizes for every facet (robustly)
    for ax in g.axes.flat:
        ax.tick_params(axis='x', which='both', labelsize=xtick_size)
        ax.tick_params(axis='y', which='both', labelsize=ytick_size)
        # Fallback in case styles/backends override tick_params
        plt.setp(ax.get_xticklabels(), fontsize=xtick_size)
        plt.setp(ax.get_yticklabels(), fontsize=ytick_size)
        # Also scale the scientific-notation offset text if present
        ax.yaxis.get_offset_text().set_size(ytick_size)

    # Shared labels
    g.fig.supylabel(ylabel, x=0.002, fontsize=ylabel_size)
    g.fig.supxlabel(xlabel, y=0.002, fontsize=xlabel_size)

    # Legend cleanup + size
    leg = getattr(g, "_legend", None) or getattr(g, "legend", None)
    if leg is not None:
        leg.set_title(None)
        for txt in leg.get_texts():
            txt.set_fontsize(min(xtick_size, ytick_size))

    return g


def mva(
        data,
        time_col='year',
        window=7,
        min_period=1,
        col='SOC_0_15CM',
        grouping=("Residue", "Nitrogen"),
        preserve_start=True,  # NEW: keep first floor(window/2) values unchanged
):
    date_col = time_col
    carbon = data.copy()
    new_soc_col = f"{col}_roll_mean"

    # Helper applied per series (per group or whole column)
    def _roll_preserve_start(s):
        # centered rolling mean with your min_period
        r = s.rolling(window=window, center=True, min_periods=min_period).mean()

        if preserve_start and window > 1:
            k = window // 2  # number of starting rows to preserve
            if len(s) <= k:
                # If the series is too short, just keep it unchanged
                return s.astype(r.dtype) if hasattr(r, "dtype") else s
            # overwrite the first k rows with original values
            r.iloc[:k] = s.iloc[:k]
        return r

    if grouping:
        carbon = carbon.sort_values([*grouping, date_col])
        carbon[new_soc_col] = (
            carbon.groupby(list(grouping), sort=False, dropna=False)[col]
            .transform(_roll_preserve_start)
        )
    else:
        carbon = carbon.sort_values([date_col])
        carbon[new_soc_col] = _roll_preserve_start(carbon[col])

    return carbon


def carbon_obj(df):
    mv = mva(df, grouping=None)

    return -mv['SOC_0_15CM_roll_mean'].mean()


def yield_obj(df):
    mv = mva(df, col='maizeyield', grouping=None)
    return -mv['maizeyield_roll_mean'].mean()


def find_soc_equilibrium(
        df: pd.DataFrame,
        group_cols=("Nitrogen", "Residue"),
        year_col="year",
        soc_col="SOC_0_15CM",
        window=7,
        min_stable_years=2,  # need >= this many consecutive years of stability
        center=False,
        eps_abs=0.02,  # absolute band (SOC units) allowed for "no change"
        eps_rel=0.003,  # relative band (fraction of peak; e.g., 0.3%)
        return_reason=True,
):
    """
    Equilibrium = first year AFTER the rolling-mean peak where the rolling mean
    remains within a small band for >= min_stable_years consecutive years.
    The band is max(eps_abs, eps_rel * peak_value).
    Returns one row per (Nitrogen, Residue) with equilibrium & peak info.
    """

    # Ensure Year exists if only a date col is present
    if year_col not in df.columns:
        for cand in ("Date", "Today", "Clock.Today", "date"):
            if cand in df.columns:
                df = df.copy()
                df[year_col] = pd.to_datetime(df[cand]).dt.year
                break
        else:
            raise ValueError(f"'{year_col}' not found and no date-like column detected")

    rows = []
    for keys, g in df.groupby(list(group_cols), dropna=False):

        # yearly aggregate (mean in case of multiple rows per year)
        ann = (g[[year_col, soc_col]]
               .groupby(year_col, as_index=False)
               .mean()
               .sort_values(year_col)
               .reset_index(drop=True))

        # need enough years for rolling + stability check
        if len(ann) < window + min_stable_years:
            res = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            res.update(dict(equilibrium_year=np.nan, equilibrium_value=np.nan,
                            peak_year=np.nan, peak_value=np.nan))
            if return_reason:
                res["reason"] = "insufficient_years"
            rows.append(res)
            continue

        ann["soc_rm"] = ann[soc_col].rolling(window=window, center=center,
                                             min_periods=window).mean()
        rm = ann["soc_rm"]

        if not rm.notna().any():
            res = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            res.update(dict(equilibrium_year=np.nan, equilibrium_value=np.nan,
                            peak_year=np.nan, peak_value=np.nan))
            if return_reason:
                res["reason"] = "all_nan_rolling"
            rows.append(res)
            continue

        # peak (first max)
        peak_idx = int(rm.idxmax())
        peak_val = float(rm.iloc[peak_idx])
        peak_year = int(ann.loc[peak_idx, year_col])

        # tolerance band
        band = max(eps_abs, eps_rel * abs(peak_val))

        eq_year = np.nan
        eq_val = np.nan
        found = False

        # scan after peak: require a run of (min_stable_years + 1) points within band
        # we check the range (max - min) in that run
        run_len = min_stable_years + 1
        start = peak_idx + 1
        for pos in range(start, len(ann) - run_len + 1):
            window_vals = rm.iloc[pos:pos + run_len].dropna()
            cv = (window_vals.std() / window_vals.mean()) * 100
            print(cv, window_vals)
            if len(window_vals) == run_len and cv < 1:  # (window_vals.max() - window_vals.min()) <= band:
                # equilibrium year: first year of the stable run (or use the last by choice)
                eq_year = int(ann.loc[pos, year_col])
                eq_val = float(rm.iloc[pos])
                found = True
                break

        res = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        res.update(dict(equilibrium_year=eq_year, equilibrium_value=eq_val, n_years=eq_year - 1904,
                        peak_year=peak_year, peak_value=peak_val))
        if return_reason and not found:
            res["reason"] = "no_stable_plateau"
        rows.append(res)

    out = pd.DataFrame(rows).sort_values(list(group_cols)).reset_index(drop=True)
    return out


cmds = ['[Phenology].GrainFilling.Target.FixedValue', '[Leaf].Photosynthesis.RUE.FixedValue',
        '[Phenology].Juvenile.Target.FixedValue',
        '[Phenology].MaturityToHarvestRipe.Target.FixedValue',
        '[Root].RootFrontVelocity.PotentialRootFrontVelocity.PreFlowering.RootFrontVelocity.FixedValue',
        '[Root].MaxDailyNUptake.FixedValue', '[Phenology].MaturityToHarvestRipe.Target.FixedValue',
        '[Grain].MaximumGrainsPerCob.FixedValue']


def edit_cultivar(model, x=None, cultivar_name=None):
    if cultivar_name is None:
        cultivar_name = 'B_110'
    if cultivar_name not in model.inspect_model('Models.PMF.Cultivar', fullpath=False):
        raise ValueError(F'cultivar name: {cultivar_name} not found')
    xp = [518.41069723, 1.89807246, 293.74403161, 159.27912538, 20.00361265, 19.93488138]

    if x is None:
        x = xp
        logger.warn(f"No x provided: using \n {dict(zip(cmds, xp))}")
    model.edit_model(
        model_type='Cultivar',
        simulations=None,
        commands=cmds,
        values=[*x[:len(cmds)]],
        new_cultivar_name='pioneer_e',
        model_name='B_110',
        cultivar_manager='Sow using a variable rule')


def calculate_soc_changes(data, col, grouping=('Nitrogen', 'Residue', 'year')):
    import pandas as pd
    if isinstance(data, ApsimModel):
        df = data.get_simulated_output('carbon')
    else:
        df = data.copy()

    # Ensure we have a Year column
    if 'year' not in df.columns:
        for cand in ('Date', 'Today', 'Clock.Today', 'date'):
            if cand in df.columns:
                df['year'] = pd.to_datetime(df[cand]).dt.year
                break
        else:
            raise ValueError("No 'Year' or date-like column found.")

    yr0, yr1 = df.year.min(), df.year.max()

    # Keep only the two target years; average within year if there are multiple rows
    annual = (df[df['year'].isin([yr0, yr1])]
              .groupby([*grouping], observed=True, as_index=False)[col]
              .mean())

    # Wide form: columns=years, values=SOC
    grouped = list(grouping)
    grouped.remove('year')
    wide = annual.pivot(index=[*grouped], columns='year', values=col)

    # Change: 2005 - 1904
    # out = (wide.assign(dSOC_0_15CM=wide[yr1] - wide[yr0])
    #             .reset_index()[['Nitrogen', 'Residue', f'd{col}']])
    # Change and percent change
    change = wide[yr1] - wide[yr0]
    pct = np.divide(change, wide[yr0], out=np.full(change.shape, np.nan), where=wide[yr0] != 0) * 100

    out = (pd.DataFrame({
        f'd{col}': change,
        f'%{col}': pct
    })
           .reset_index()
           .sort_values(grouped))

    # (optional) round percentage
    out[f'%{col}'] = out[f'%{col}'].round(2)
    out[f'd{col}-r'] = (out[f'd{col}'] / 101).round(4)

    return out


def generate_n_rates(base_n, deviation=0.10):
    base_out = [137, 180, 217, 247]

    return sorted(base_out)


nr = generate_n_rates(202)

# N rate for generating scenarios based on MRTN rates (202)
N_RATES = ', '.join(map(str, generate_n_rates(base_n=202, deviation=0.10)))

# generate N rate for fitting quadratic regression
n_ranges = list(np.linspace(0, 326, 80))
n_ranges.append(168)
QN_RATES = ', '.join(map(str, n_ranges))


# ______________ plots _____________________
def test_mva(instance, title='', file_name=None, **kwargs):
    instance.plot_mva(**kwargs)
    plt.title(title)
    x, y, tab = kwargs.get('time_col'), kwargs.get('response'), kwargs.get('table')
    if not isinstance(tab, str):
        tab = ""
    name = file_name or f"mva_single_{x}-{y}-{tab}.svg"
    plt.tight_layout()
    plt.savefig(name, dpi=600)
    os.startfile(name)
    plt.close()


def cat_plot(instance, title="", file_name=None, **kwargs):
    instance.cat_plot(**kwargs)

    if isinstance(kwargs['table'], str):
        df = instance.get_simulated_output(kwargs.get('table'))
        df['Nitrogen'] = pd.Categorical(df['Nitrogen'], ordered=True)
        df['Residue'] = pd.Categorical(df['Residue'], ordered=True)
        kwargs['table'] = df
    x, y, tab = kwargs.get('time_col'), kwargs.get('response'), kwargs.get('table')
    if not isinstance(tab, str):
        tab = ""
    name = file_name or f"series_single_{x}-{y}-{tab}.svg"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(name, dpi=600)
    os.startfile(name)
    plt.close()


def series(instance, title="", file_name=None, **kwargs):
    instance.series_plot(**kwargs)
    plt.title(title)
    plt.tight_layout()
    x, y, tab = kwargs.get('time_col'), kwargs.get('y'), kwargs.get('table')
    if not isinstance(tab, str):
        tab = ""
    name = file_name or f"series_single_{x}-{y}-{tab}.svg"
    plt.savefig(name, dpi=600)
    os.startfile(name)
    plt.close()


def plot_reg_fit(
        X, y,
        data: pd.DataFrame | None = None,
        fig_name: str = None,
        xname: str | None = None,
        yname: str | None = None,
        color_by: str | None = None,  # e.g., 'year' to color groups
        n_points: int = 200,
        xlabel: str = "",
        ylabel: str = "",
        show_eq: bool = True
):
    """
    Plot a simple linear regression fit for a single predictor X vs y.

    - Supports X as 1D array/Series OR (data, xname) with y=(data, yname).
    - Raises if X has more than 1 column.
    - Sorts X for a clean fitted line.
    """
    fig_name = fig_name or f'{xname}corn_grain_yield_Mg{yname}.svg'
    # Resolve X, y from DataFrame if names are given
    if data is not None and xname and yname:
        Xv = data[xname].to_numpy().reshape(-1, 1)
        yv = data[yname].to_numpy().ravel()
    else:
        X = np.asarray(X)
        y = np.asarray(y)
        # Accept Series/list/1D array; also allow shape (n, 1)
        if X.ndim == 1:
            Xv = X.reshape(-1, 1)
        elif X.ndim == 2 and X.shape[1] == 1:
            Xv = X
        else:
            raise ValueError("X must be 1D (n,) or (n,1) to plot a fitted line.")
        yv = y.ravel()

    # Fit
    model = LinearRegression()
    model.fit(Xv, yv)

    # Make a sorted grid for the line
    x_min = np.nanmin(Xv)
    x_max = np.nanmax(Xv)
    x_grid = np.linspace(x_min, x_max, n_points).reshape(-1, 1)
    y_grid = model.predict(x_grid)

    # Coeffs / metrics
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r2 = model.score(Xv, yv)

    # Plot
    plt.figure(figsize=(8, 5))

    # Scatter points
    if data is not None and xname and yname and color_by and color_by in data.columns:
        for key, dsub in data.groupby(color_by):
            plt.scatter(dsub[xname], dsub[yname], label=f"{color_by}={key}", alpha=0.8)
    else:
        # single color scatter
        if data is not None and xname and yname:
            plt.scatter(data[xname], data[yname], alpha=0.8, label="Data")
        else:
            plt.scatter(Xv.ravel(), yv, alpha=0.8, label="Data")

    # Fitted line
    plt.plot(x_grid.ravel(), y_grid, 'r--', lw=2, label='fitted line')

    # Labels
    plt.xlabel(xlabel if xlabel else (xname or "X"))
    plt.ylabel(ylabel if ylabel else (yname or "y"))

    # Equation / R²
    if show_eq:
        eq = f"y = {slope:.3g} x + {intercept:.3g}"
        r2_text = f"R² = {r2:.3f}"
        ax = plt.gca()
        ax.text(0.02, 0.98, eq, transform=ax.transAxes, va="top")
        ax.text(0.02, 0.90, r2_text, transform=ax.transAxes, va="top")

    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(fig_name, dpi=600)
    plt.close()
    open_file(fig_name)

    return model


def pbias(observed, simulated):
    """
    Compute Percent Bias (PBIAS).

    Parameters
    ----------
    observed : array-like
        Observed values.
    simulated : array-like
        Simulated/modelled values.

    Returns
    -------
    float
        Percent Bias (PBIAS). Positive values indicate overestimation,
        negative values indicate underestimation.
    """
    observed = np.asarray(observed, dtype=float)
    simulated = np.asarray(simulated, dtype=float)

    if np.sum(observed) == 0:
        raise ValueError("Sum of observed values is zero; PBIAS is undefined.")
    return 100.0 * np.sum(simulated - observed) / np.sum(observed)


from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def plot_reg_fit2(
        X: str,
        y: str,
        *,
        data: pd.DataFrame,
        fig_name: Optional[str] = None,
        color_by: Optional[str] = None,
        n_points: int = 200,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        show_eq: bool = True,
        fig_size=(8, 5),
        preview=True,
):
    """
    Plot a simple linear regression fit between two columns in a DataFrame.

    Parameters
    ----------
    X : str
        Column name for the predictor variable.
    y : str
        Column name for the response variable.
    data : pd.DataFrame
        DataFrame containing both ``X`` and ``y`` columns.
    fig_name : str, optional
        Output figure filename. If None, a name is generated automatically.
    color_by : str, optional
        Column name used to color/group points (e.g., ``year`` or ``Plotid``).
    n_points : int, optional
        Number of points used to draw the fitted regression line.
    xlabel : str, optional
        Label for the x-axis. Defaults to ``X``.
    ylabel : str, optional
        Label for the y-axis. Defaults to ``y``.
    show_eq : bool, optional
        If True, display the fitted equation and R² on the plot.
    fig_name : tuple of length 2, optional
        height and width, respectively
    preview : bool, optional
        open the figure in external application determined by the figure name extension

    Returns
    -------
    LinearRegression
        The fitted scikit-learn LinearRegression model.

    Raises
    ------
    TypeError
        If ``X`` or ``y`` are not strings.
    KeyError
        If required columns are missing from ``data``.
    """

    # ---- validation ---------------------------------------------------------
    if not isinstance(X, str) or not isinstance(y, str):
        raise TypeError("X and y must be column names (str).")

    missing = {c for c in (X, y, color_by) if c and c not in data.columns}
    if missing:
        raise KeyError(f"Missing columns in data: {missing}")

    # ---- resolve data -------------------------------------------------------
    Xv = data[X].to_numpy(dtype=float).reshape(-1, 1)
    yv = data[y].to_numpy(dtype=float)

    # ---- fit model ----------------------------------------------------------
    model = LinearRegression()
    model.fit(Xv, yv)

    # ---- prediction grid ----------------------------------------------------
    x_min, x_max = np.nanmin(Xv), np.nanmax(Xv)
    x_grid = np.linspace(x_min, x_max, n_points).reshape(-1, 1)
    y_grid = model.predict(x_grid)

    # ---- metrics ------------------------------------------------------------
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r2 = model.score(Xv, yv)

    # ---- plotting -----------------------------------------------------------
    fig_name = fig_name or f"{X}_vs_{y}_regression.svg"
    plt.figure(figsize=fig_size)

    if color_by:
        for key, dsub in data.groupby(color_by):
            plt.scatter(dsub[X], dsub[y], label=f"{color_by}={key}", alpha=0.8)
    else:
        plt.scatter(data[X], data[y], alpha=0.8, label="")

    plt.plot(x_grid.ravel(), y_grid, "r--", lw=2, label="Fitted line")

    plt.xlabel(xlabel or X, fontsize=18)
    plt.ylabel(ylabel or y, fontsize=18)

    if show_eq:
        ax = plt.gca()
        ax.text(
            0.02,
            0.98,
            f"y = {slope:.3g}x + {intercept:.3g}\nR² = {r2:.3f}\nn = {len(yv)}",
            transform=ax.transAxes,
            va="top", fontsize=16,
        )

    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_name, dpi=450)
    if preview:
        open_file(fig_name)
    plt.close()

    return model


def json_safe(obj):
    """
    Recursively convert APSIM / .NET / numpy objects
    into JSON-serializable Python types.
    """
    import datetime
    import numpy as np

    # dict
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}

    # list / tuple
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]

    # numpy scalars
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()

    # datetime
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()

    # pythonnet / .NET iterable
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        try:
            return [json_safe(v) for v in list(obj)]
        except Exception:
            pass

    # primitives
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # fallback: stringify APSIM/C# objects
    return str(obj)


def carbon_to_Mg(data: pd.DataFrame, conc_col: str, depth_col: str, bd_col: str) -> pd.DataFrame:
    """ Assumes that the data concentration column is g/kg"""
    data = data.copy()
    data.eval(f"conc = {conc_col}/10", inplace=True)
    data.eval(f"soc_Mg = conc * {depth_col} * {bd_col} ", inplace=True)
    return data


def load_manifest():
    import yaml
    with open("manifest.yml", "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def collect_used_params(base_files, json_file=None):
    """
    Collect parameters used by Managers, SurfaceOrganicMatter,
    and Clock models from APSIM files and export them as a
    readable, structured JSON file.
    """
    from apsimNGpy.core.apsim import ApsimModel
    from apsimNGpy.core_utils.utils import is_scalar
    from apsimNGpy.core.config import apsim_bin_context
    import dotenv
    import json
    source_files = Path('APSIMx_base_files').resolve()
    # output directory
    manager_params = source_files / "Calibrated Parameters"
    manager_params.mkdir(parents=True, exist_ok=True)
    json_file_path = manager_params / json_file if json_file else manager_params / "params.json"

    # load APSIM bin from environment
    dotenv.load_dotenv()
    bin_7493 = os.getenv("7493")

    if is_scalar(base_files):
        base_files = [base_files]

    big_data = {}
    # names were not changed but OPV does not necessarily mean open-pollinated
    suspected_cultivar_names = ['OPVPH4', 'OPV_untreated', 'OPV_untreated' 'OPVPH4edited', 'OPV',
                                'OPVPH4edited_untreated']
    with apsim_bin_context(bin_7493):
        for base_file in base_files:
            data = {
                "Models.Manager": {},
                "Models.Surface.SurfaceOrganicMatter": {},
                "Models.Clock": {},
                "Models.Soils.Organic": {},
                "Models.Soils.Physical": {},
                'Models.WaterModel.WaterBalance': {},
                'Models.PMF.Cultivar': {}
            }

            base = source_files / base_file
            big_data[os.path.relpath(base)] = data
            with ApsimModel(base) as model:
                apsimx_path = os.path.realpath(model._model)

                # ---------------- cultivars ----------------
                for path in model.inspect_model("Models.PMF.Cultivar"):
                    name = path.split(".")[-1]
                    if name in suspected_cultivar_names:
                        params = model.inspect_model_parameters_by_path(path)

                        data["Models.PMF.Cultivar"][path] = {
                            "path": path,
                            'name': name,
                            "params": params,
                            "apsimx_path": apsimx_path
                        }

                # ---------------- Managers ----------------
                for path in model.inspect_model("Models.Manager"):
                    params = model.inspect_model_parameters_by_path(path)
                    name = path.split(".")[-1]

                    data["Models.Manager"][path] = {
                        "path": path,
                        'name': name,
                        "params": params,
                        "apsimx_path": apsimx_path
                    }
                # __________________ organic _____________________
                for path in model.inspect_model("Models.Soils.Organic"):
                    params = model.inspect_model_parameters_by_path(path)
                    params = {
                        k: (v.tolist() if hasattr(v, "tolist") else v)
                        for k, v in params.items()
                    }
                    name = path.split(".")[-1]

                    data["Models.Soils.Organic"][path] = {
                        "path": path,
                        "params": params,
                        'name': name,
                        "apsimx_path": apsimx_path
                    }
                # ________________ physical __________________________
                for path in model.inspect_model("Models.Soils.Physical"):
                    params = model.inspect_model_parameters_by_path(path)
                    params = {
                        k: (v.tolist() if hasattr(v, "tolist") else v)
                        for k, v in params.items()
                    }
                    name = path.split(".")[-1]

                    data["Models.Soils.Physical"][path] = {
                        "path": path,
                        "params": params,
                        'name': name,
                        "apsimx_path": apsimx_path
                    }
                # -------- Surface Organic Matter ----------
                for path in model.inspect_model("Models.Surface.SurfaceOrganicMatter"):
                    params = model.inspect_model_parameters_by_path(path)
                    name = path.split(".")[-1]

                    data["Models.Surface.SurfaceOrganicMatter"][path] = {
                        "path": path,
                        "params": params,
                        'name': name,
                        "apsimx_path": apsimx_path
                    }

                # ---------------- Clock -------------------
                for path in model.inspect_model("Models.Clock"):
                    params = model.inspect_model_parameters_by_path(path)
                    params["Start"] = params["Start"].isoformat()
                    params["End"] = params["End"].isoformat()
                    name = path.split(".")[-1]

                    data["Models.Clock"][path] = {
                        "path": path,
                        "params": params,
                        'name': name,
                        "apsimx_path": apsimx_path
                    }

    # -------- Write readable JSON --------
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(
            json_safe(big_data),
            f,
            indent=2,
            sort_keys=True,
            ensure_ascii=False
        )

    os.startfile(json_file_path)
    return data


def tabulate_data(_model, paths):
    from apsimNGpy.core.apsim import ApsimModel
    from pandas import DataFrame
    if isinstance(paths, str) or isinstance(_model, (str, Path)):
        raise TypeError('Only list[str] are allowed')
    pa = []
    for m, p in zip(_model, paths):
        print(m)
        plot = p.split('.')[-3]
        with ApsimModel(m) as model:
            params = model.inspect_model_parameters_by_path(p)

            params['plot'] = plot
            pa.append(params)
    df = DataFrame(pa)
    return df


def optimize(bin_key='7844', base_name='Plot3NC1.apsimx'):
    from pandas import read_csv
    from apsimNGpy.core.config import apsim_bin_context
    _maize_yield = read_csv(maize_yield_data)
    import dotenv
    dotenv.load_dotenv()
    bin_path7844 = os.getenv(bin_key)
    base_apsimx = os.path.realpath(source_files / base_name)
    with apsim_bin_context(bin_path7844):
        from apsimNGpy.optimizer.problems.smp import MixedProblem
        from apsimNGpy.optimizer.minimize.single_mixed import MixedVariableOptimizer
        mP = MixedProblem(model=base_apsimx, table='MaizeR', trainer_col=measured_yield_col,
                          pred_col=predicted_yield_col,
                          trainer_dataset=_maize_yield, metric='r2', index=['year', 'Plotid'])
        mP.submit_factor(path='.Simulations.Replacements.Maize.OPVPH4edited',
                         candidate_param=[

                             '[Leaf].Photosynthesis.RUE.FixedValue',
                             '[Phenology].GrainFilling.Target.FixedValue',
                             '[Phenology].Juvenile.Target.FixedValue',
                             '[Phenology].FloweringToGrainFilling.Target.FixedValue'
                         ],
                         bounds=[[1, 2.4], [600, 850], [200, 260], [100, 200]],
                         start_value=[1.8, 800, 200, 100],
                         other_params={'sowed': True},
                         cultivar=True)
        optimizer = MixedVariableOptimizer(problem=mP)
        mn = optimizer.minimize_with_local()
        print(mn)


if __name__ == '__main__':
    single = create_experiment('base_single.apsimx', lonlat=(-93.01134, 42.0124), start=2019,
                               out_path=f"{path_to_MP_data}/single_test.apsimx")
    met = read_apsim_met(single.inspect_model_parameters('Models.Climate.Weather', 'Weather'))
    split = create_experiment('base_split.apsimx', lonlat=(-93.0134, 42.0124),
                              out_path=f"{path_to_MP_data}/split_test.apsimx")

    # other = create_experiment('other', lonlat=(-93.0134, 42.0124))
    # unittest.main(test_experiment)
# model.preview_simulation()
# single.replace_soils(lonlat=(-93.0134, 42.0124), simulation_names='changeFert_verity')
