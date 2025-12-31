import os
from pathlib import Path
from pprint import pformat
from apsimNGpy.settings import logger
from settings import RESULTS, RESIDUE_RATES, N_RATES
from utils import open_file

import pandas as pd
import numpy as np
from apsimNGpy.core.config import apsim_version


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, coerce numerics (except 'Depth'), make comparison stable."""
    d = df.copy()
    for c in d.columns:
        if c != "Depth":
            d[c] = pd.to_numeric(d[c], errors="ignore")
    return d.sort_index().sort_index(axis=1)


def frames_close(a: pd.DataFrame, b: pd.DataFrame, rtol=1e-6, atol=1e-8) -> tuple[bool, str]:
    try:
        pd.testing.assert_frame_equal(_normalize(a), _normalize(b),
                                      check_like=True, check_dtype=False,
                                      rtol=rtol, atol=atol)
        return True, ""
    except AssertionError as e:
        diff = a.compare(b, keep_shape=True, keep_equal=False)
        return False, f"{e}\n\nDiff (first rows):\n{diff.head()}"


def _tidy_for_text(df: pd.DataFrame, float_dec=3) -> pd.DataFrame:
    """Make a DF print nicely in plain text."""
    d = df.copy()
    # Put Depth first if present
    if "Depth" in d.columns:
        cols = ["Depth"] + [c for c in d.columns if c != "Depth"]
        d = d[cols]

    # Make list/array cells readable
    def fmt(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return ", ".join(f"{v:.{float_dec}f}" if isinstance(v, (int, float, np.floating)) else str(v) for v in x)
        if isinstance(x, (int, float, np.floating)):
            return f"{x:.{float_dec}f}"
        return x

    return d.applymap(fmt)


def df_paragraph(df: pd.DataFrame, title: str) -> str:
    with pd.option_context("display.max_columns", None,
                           "display.width", 2000,
                           "display.max_colwidth", 1000):
        return f"{title}:\n{_tidy_for_text(df).to_string(index=False)}\n"


def make_report(single_model, model2, preview=False):
    """
    This function is very critical, because it also investigates if inputs were accidentally different between apsim files representing  split and single N application
    :return: None
    """
    lines = []

    # ___________________weather__________________________________________________
    single_weather_file = single_model.inspect_model_parameters(model_type='Models.Climate.Weather',
                                                                model_name='Weather')
    split_weather_file = model2.inspect_model_parameters(model_type='Models.Climate.Weather', model_name="Weather")
    lines.append("weather_path:\n" +
                 pformat(split_weather_file))
    if Path(single_weather_file).stat().st_size != Path(split_weather_file).stat().st_size:
        raise ValueError(
            f'Both single and split should have the same weather file got: single:{single_model}, split:{split_weather_file}')
    # _____________________planting info _____________________________________________
    single_planting = single_model.inspect_model_parameters(model_type='Models.Manager',
                                                            model_name='Sow on a fixed date')
    split_planting = model2.inspect_model_parameters(model_type='Models.Manager', model_name="Sow on a fixed date")
    if single_planting != split_planting:
        raise ValueError('sowing script should have the same information across single and split')
    lines.append("planting information:\n" + pformat(split_planting))
    # ________________fertilization information __________________________________________
    split_fertilizer_application = model2.inspect_model_parameters(
        model_type='Models.Manager', model_name='fertilize in phases'
    )
    split_fertilizer_application.pop('Amount', None)  # avoid KeyError
    lines.append("fertilizer application split:\n" +
                 pformat(split_fertilizer_application))

    single_fertilizer_application = single_model.inspect_model_parameters(
        model_type='Models.Manager', model_name='single_N_at_sowing'
    )
    single_fertilizer_application.pop('Amount', None)  # avoid KeyError
    lines.append("fertilizer application single:\n" +
                 pformat(single_fertilizer_application))

    # ______________simulation period info_________________________________________________________
    clock_info_split = model2.inspect_model_parameters(model_type='Models.Clock', model_name='Clock',
                                                       parameters=['Start', 'End'])

    clock_info_single = single_model.inspect_model_parameters(model_type='Models.Clock', model_name='Clock',
                                                              parameters=['Start', 'End'])
    if clock_info_single != clock_info_split:
        raise ValueError("Clock has different start between single and split models")
    start, end = clock_info_single['Start'], clock_info_single['End']
    start, end = start.strftime('%m/%d/%Y'), end.strftime('%m/%d/%y')
    lines.append(f"simulation period: {start}-{end}")

    # _______________organic ________________________________________
    organ_single = single_model.inspect_model_parameters(model_type='Models.Soils.Organic', model_name='Organic')
    organ_split = model2.inspect_model_parameters(model_type='Models.Soils.Organic', model_name='Organic')
    if not organ_single.equals(organ_split):
        raise ValueError("organic variables different from split and single")
    lines.append(f"organic:\n {pformat(organ_single)}")

    # _______________soil physical ________________________________________
    phy_single = single_model.inspect_model_parameters(model_type='Models.Soils.Physical', model_name='Physical')
    phy_split = model2.inspect_model_parameters(model_type='Models.Soils.Physical', model_name='Physical')
    if not phy_split.equals(phy_single):
        raise ValueError("organic variables different from split and single")

    lines.append(f"organic:\n {pformat(phy_single)}")

    # ________________ tillage info _________________________________________
    tillage_single = single_model.inspect_model_parameters(model_type='Models.Manager', model_name='Tillage')
    tillage_split = model2.inspect_model_parameters(model_type='Models.Manager', model_name='Tillage')
    if tillage_split != tillage_single:
        raise ValueError(
            f" tillage variables for split is different from single, single: {tillage_single} \n split:{tillage_split}")
    phy_single.to_csv(RESULTS / f'phy_single.csv', index=False)

    lines.append(f"tillage:\n   {pformat(tillage_single)}")
    text = "\n\n".join(lines) + "\n"

    path = RESULTS / 'methods.txt'
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    if preview:
        open_file(path)
    return text


def make_mgt_text(model, preview=False):
    """
    This function is very critical, because it also investigates if inputs were accidentally different between apsim files representing  split and single N application
    :return: None
    """
    lines = []

    # ___________________weather__________________________________________________
    weather_file = model.inspect_model_parameters(model_type='Models.Climate.Weather',
                                                  model_name='Weather')

    lines.append("Weather_path:\n" +
                 pformat(weather_file))

    # _____________________planting info _____________________________________________
    managers = {'Sow on a fixed date', 'Sow using a variable rule'}
    for names in {'Sow on a fixed date', 'Sow using a variable rule'}:
        if names in model.inspect_model('Models.Manager', fullpath=False):
            name = names
            break

    else:
        raise ValueError(f'no planting script found expected any of these parameters: {managers}')
    planting = model.inspect_model_parameters(model_type='Models.Manager',
                                              model_name=name)

    lines.append("Planting was done using the following parameters:\n" + pformat(planting))

    fertilizer_application = model.inspect_model_parameters(
        model_type='Models.Manager', model_name='fertilize in phases')

    lines.append("Fertilizer application base parameters:\n" +
                 pformat(fertilizer_application))

    # ______________simulation period info_________________________________________________________
    clock_info = model.inspect_model_parameters(model_type='Models.Clock', model_name='Clock',
                                                parameters=['Start', 'End'])

    start, end = clock_info['Start'], clock_info['End']
    start, end = start.strftime('%m/%d/%Y'), end.strftime('%m/%d/%Y')
    lines.append(f"Simulation dates: {start}-{end}")

    # ________________ tillage info _________________________________________
    tillage = model.inspect_model_parameters(model_type='Models.Manager', model_name='Tillage')

    lines.append(f"Tillage parameters:\n   {pformat(tillage)}")
    som = model.inspect_model_parameters(model_type='Models.Surface.SurfaceOrganicMatter',
                                         model_name='SurfaceOrganicMatter')
    lines.append(f"Initial surface organic matter parameters:\n   {pformat(som)}")
    lines.append(f"Experimented residue retention rates:\n" + pformat(RESIDUE_RATES))
    lines.append(f"Experimented nitrogen fertilizer rates:\n" + pformat(N_RATES))
    lines.append(f"Last opened in: {apsim_version()}")
    lines.append(f"Generated by: =================== apsimNGpy==========================\n")
    text = "\n\n".join(lines) + "\n"

    path = RESULTS / 'methods.txt'
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    if preview:
        open_file(path)
    return text


import pandas as pd
from pathlib import Path


def summarize_outputs(model,
                      *,
                      by=("Nitrogen",),  # group keys (e.g., ("Nitrogen", "Residue", "year"))
                      year=None,  # int, list/tuple, slice, or callable(df)->mask
                      outfile=None,  # 'summary.csv' or 'summary.xlsx' (optional)
                      agg_overrides=None):  # per-metric agg (e.g., {'lai':'sum'})
    """
    Build a single table of summary stats from APSIM outputs.

    Parameters
    ----------
    model : object with .get_simulated_output(kind)
        kinds used: 'yield', 'daily', 'carbon', 'water'
    by : tuple[str]
        Grouping columns (default: ('Nitrogen',))
    year : int | iterable[int] | slice | callable | None
        Filter rows where df['year'] matches; callable receives df and must return a boolean mask.
        Examples: year=1904, year=[1904,1905], year=slice(1900,1905)
    outfile : str | Path | None
        If provided, writes the table to CSV/Excel based on file extension.
    agg_overrides : dict | None
        Optional per-metric aggregation, e.g. {'lai': 'sum'}.
        Default aggregates are 'mean' for all metrics except 'lai' ('sum').

    Returns
    -------
    pandas.DataFrame
    """
    # defaults
    default_aggs = {
        'surface_carbon': 'mean',
        'lai': 'sum',
        'SOC_0_15CM': 'mean',
        'InCropMeanSoilWaterTopFirstLayer': 'mean',
        'SOM': 'mean',
        'mineralN_ly1': 'mean',
        'maizeyield': 'mean',
        'BBiomass': 'mean',
    }
    if agg_overrides:
        default_aggs.update(agg_overrides)

    def _fetch(kind):
        df = model.get_simulated_output(kind)
        # filter by year if requested
        if year is not None and 'year' in df.columns:
            if callable(year):
                df = df[year(df)]
            elif isinstance(year, slice):
                df = df[(df['year'] >= year.start) & (
                        df['year'] < (year.stop if year.stop is not None else df['year'].max() + 1))]
            elif isinstance(year, (list, tuple, set)):
                df = df[df['year'].isin(year)]
            else:  # int
                df = df[df['year'] == year]
        return df

    # pull the needed tables once
    df_yld = _fetch('yield')
    df_daily = _fetch('daily')
    df_carb = _fetch('carbon')
    df_watr = _fetch('water')

    # helper: safe group/agg if column exists
    def _agg(df, col, label=None):
        if col not in df.columns:
            return pd.Series(dtype=float, name=label or col)
        s = (df.groupby(list(by))[col]
             .agg(default_aggs.get(col, 'mean')))
        s.name = label or col
        return s

    # build each metric series
    s_surface_c = _agg(df_yld, 'surface_carbon', 'mean_surface_carbon')
    s_lai = _agg(df_daily, 'lai', 'lai_sum')
    s_soc015 = _agg(df_carb, 'SOC_0_15CM', 'mean_SOC_0_15CM')
    s_sw_top = _agg(df_watr, 'InCropMeanSoilWaterTopFirstLayer', 'mean_SW_top_0_15cm')
    s_som = _agg(df_daily, 'SOM', 'mean_SOM')
    s_minN1 = _agg(df_daily, 'mineralN_ly1', 'mean_mineralN_ly1')
    s_agb = _agg(df_yld, 'maizeyield', 'mean_maize_yield')
    s_bgb = _agg(df_yld, 'BBiomass', 'mean_belowground_biomass')

    # join on group keys
    summary = pd.concat(
        [s_surface_c, s_lai, s_soc015, s_sw_top, s_som, s_minN1, s_agb, s_bgb],
        axis=1
    ).reset_index()

    # write if requested
    if outfile:
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        if outfile.suffix.lower() == '.xlsx':
            summary.to_excel(outfile, index=False)
        else:
            summary.to_csv(outfile, index=False)
    return summary


from datetime import datetime
from typing import List, Dict, Any

def add_run_memo(model, memo_title: str = "Data inputs") -> None:
    """
    Build and attach a well-formatted memo to an APSIM model with sections for:
    - Date
    - Clocks
    - Managers
    - Reports

    The function inspects the model for common APSIM NG components using:
      - model.inspect_model("Models.Clock")
      - model.inspect_model("Models.Manager")
      - model.inspect_model("Models.Report")

    and then queries their parameters via:
      - model.inspect_model_parameters_by_path(<path>)

    Finally, it calls:
      - model.add_memo(memo_text=<str>)

    Parameters
    ----------
    model : Any
        APSIM model wrapper/adapter exposing `inspect_model`,
        `inspect_model_parameters_by_path`, and `add_memo`.
    memo_title : str, optional
        Title for the memo header.

    Returns
    -------
    None
    """
    def _fmt_kv(d: Dict[str, Any], keys: List[str]) -> List[str]:
        """Format selected key/value pairs if present in dict `d`."""
        out = []
        for k in keys:
            if k in d and d[k] not in (None, "", []):
                out.append(f"    - {k}: {d[k]}")
        return out

    lines: List[str] = []
    hr = "-" * 72

    # Header
    lines.append(hr)
    lines.append(f"{memo_title}")
    lines.append(hr)

    # =========================
    # Date
    # =========================
    lines.append("Date")
    lines.append("~~~~")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # =========================
    # Clocks
    # =========================
    lines.append("Clocks")
    lines.append("~~~~~~")
    try:
        clock_paths = model.inspect_model("Models.Clock") or []
        if not clock_paths:
            lines.append("- (No Clock models found)")
        for cpath in clock_paths:
            cparams = model.inspect_model_parameters_by_path(cpath) or {}
            lines.append(f"- {cpath}")
            # Try common APSIM NG clock parameter names
            lines.extend(_fmt_kv(cparams, [
                "StartDate", "EndDate", "Start", "End", "Clock.Today", "Clock.Start", "Clock.End"
            ]))
    except Exception as e:
        lines.append(f"- Error reading Clock models: {e}")
    lines.append("")

    # =========================
    # Managers
    # =========================
    lines.append("Managers")
    lines.append("~~~~~~~~")
    try:
        manager_paths = model.inspect_model("Models.Manager") or []
        if not manager_paths:
            lines.append("- (No Manager models found)")
        for mpath in manager_paths:
            mparams = model.inspect_model_parameters_by_path(mpath) or {}
            lines.append(f"- {mpath}")
            # Show a few helpful fields when present
            lines.extend(_fmt_kv(mparams, [
                "Name", "Enabled", "Script", "Action", "SowingRule", "Population",
                "FertiliserRule", "Amount", "Depth"
            ]))
    except Exception as e:
        lines.append(f"- Error reading Manager models: {e}")
    lines.append("")

    # =========================
    # Reports
    # =========================
    lines.append("Reports")
    lines.append("~~~~~~~")
    try:
        report_paths = model.inspect_model("Models.Report") or []
        if not report_paths:
            lines.append("- (No Report models found)")
        for rpath in report_paths:
            rparams = model.inspect_model_parameters_by_path(rpath) or {}
            lines.append(f"- {rpath}")
            # Common Report fields in APSIM NG
            vars_ = rparams.get("VariableNames") or rparams.get("Variables") or []
            evts_ = rparams.get("EventNames") or rparams.get("Events") or []
            if isinstance(vars_, str):
                vars_ = [v.strip() for v in vars_.split(",") if v.strip()]
            if isinstance(evts_, str):
                evts_ = [e.strip() for e in evts_.split(",") if e.strip()]
            lines.append(f"    - Variables: {len(vars_) if vars_ else 0}")
            if vars_:
                lines.append(f"      e.g., {', '.join(vars_[:5])}{' ...' if len(vars_) > 5 else ''}")
            lines.append(f"    - Events: {len(evts_) if evts_ else 0}")
            if evts_:
                lines.append(f"      e.g., {', '.join(evts_[:5])}{' ...' if len(evts_) > 5 else ''}")
    except Exception as e:
        lines.append(f"- Error reading Report models: {e}")
    lines.append(hr)

    memo_text = "\n".join(lines)
    model.add_memo(memo_text=memo_text)


