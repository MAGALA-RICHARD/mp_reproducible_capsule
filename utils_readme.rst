# open_file
"""
Open a file with the system’s default application.

Parameters
----------
file : str | pathlib.Path
    Path to an existing file. The path must include a suffix (extension).

Raises
------
ValueError
    If the path has no suffix or the file does not exist.

Notes
-----
- Windows uses `os.startfile`.
- macOS uses `open`.
- Linux/BSD uses `xdg-open`.
"""


# read_or_run
"""
Load an APSIM .apsimx file, apply simple edits, and return an ApsimModel.

Parameters
----------
apsim_file : str | pathlib.Path
    File name of the APSIM NG model (must end with ``.apsimx``). The file
    is resolved relative to ``path_to_MP_data``.
finert : float, default 0.65
    Stable carbon fraction used to update the Organic soil model
    (``Models.Soils.Organic -> FInert``).

Returns
-------
ApsimModel
    The edited APSIM model instance ready for execution.

Raises
------
ValueError
    If a directory is provided, the extension is not ``.apsimx``,
    or the file cannot be found on disk.
"""


# base
"""
Create and cache a preconfigured base ApsimModel for a given method.

Parameters
----------
method : {"single", "split", "auto", "automatic"}
    High-level experiment type controlling which base file to load and
    minimal soil organic parameters to set.

Returns
-------
ApsimModel
    The prepared model with basic soil parameters set and stale outputs removed.

Raises
------
FileNotFoundError
    If the base APSIM file cannot be found at ``path_to_MP_data``.
ValueError
    If an unsupported ``method`` is requested.

Notes
-----
Removes prior CSV/DB outputs for ``carbon`` and ``yield`` to ensure a clean run.
"""


# merge_tables
"""
Merge two APSIM output tables on common keys and derive convenience fields.

This helper accepts either:
1) a path or filename to an APSIM results database (tables are read with
   :func:`read_db_table`), or
2) an :class:`~apsimNGpy.core.apsim.ApsimModel` /
   :class:`~apsimNGpy.core.experimentmanager.ExperimentManager` (tables fetched
   via ``get_simulated_output``), or
3) a pair of preloaded :class:`pandas.DataFrame` objects.

The two tables are merged on ``('SimulationID', 'Nitrogen', 'Residue', 'year')``
(and ``'timing'`` if present). After merging, numeric convenience columns and
derived metrics are added (e.g., ``R``, ``N``, ``n_r``, biomass partitions, C:N).

Parameters
----------
data : str | pathlib.Path | ApsimModel | ExperimentManager | tuple[pd.DataFrame, pd.DataFrame] | list[pd.DataFrame]
    Source of the two tables.
tables : Sequence[str]
    Names of the **two** tables to merge (e.g., ``("yield", "carbon")``).
how : {"inner", "left", "right", "outer"}, default "inner"
    Merge strategy passed to :meth:`pandas.DataFrame.merge`.

Returns
-------
pandas.DataFrame
    Merged output with convenience and derived columns.

Raises
------
ValueError
    If inputs are invalid (wrong number/type of tables or data).
KeyError
    If required columns for derived metrics are missing.

Examples
--------
>>> dat = merge_tables("results.db", ("carbon", "yield"))
>>> dat = merge_tables((df_carbon, df_yield), ("carbon", "yield"), how="left")
"""


# create_experiment
"""
Create (and cache) an experiment-ready APSIM model from a base file.

Parameters
----------
base_file : str
    Base model name (with or without ``.apsimx`` extension) located in
    ``path_to_MP_data``.
lonlat : tuple[float, float]
    Longitude and latitude used for site-specific edits (placeholder for
    weather/soil retrieval in this workflow).
start : int, default 1904
    Simulation start year (YYYY).
end : int, default 2005
    Simulation end year (YYYY).

Returns
-------
ApsimModel | ExperimentManager
    Configured model/experiment with updated soil, SOM, weather, and clock.

Raises
------
FileNotFoundError
    If the ``base_file`` is not found at ``path_to_MP_data``.
"""


# plot_mva
"""
Plot centered moving-average (rolling mean) series across facets.

Parameters
----------
carbon : pandas.DataFrame
    Data containing the rolling-mean column (``<col>_roll_mean``) produced by :func:`mva`.
column : str
    Base column name (the plotted column is ``<column>_roll_mean``).
x : str, default "year"
    X-axis variable.
hue : str, default "Nitrogen"
    Hue variable for line color.
col : str, default "Residue"
    Column facet variable.
style : str, optional
    Style variable for line dashing/markers.
errorbar : str | None, optional
    Error bar spec passed to seaborn (e.g., ``"sd"``); ``None`` to disable.
color_palette : str, default "deep"
    Seaborn palette name.
ylabel, xlabel : str, optional
    Axis labels; defaults to names in ``x`` and ``column``.
xtick_size : int, default 12
    X tick label font size.
ytick_size : int | None, optional
    Y tick label font size (defaults to ``xtick_size``).
ylabel_size : int, default 18
    Shared y-label font size.
xlabel_size : int, default 18
    Shared x-label font size.

Returns
-------
seaborn.axisgrid.FacetGrid
    The created FacetGrid, with shared labels and legend adjusted.

Notes
-----
- Ensures tick sizes are applied robustly across facets.
- Expects the rolling-mean column to exist; use :func:`mva` beforehand.
"""


# mva
"""
Compute a centered moving average with an option to preserve initial values.

Parameters
----------
data : pandas.DataFrame
    Input data.
time_col : str, default "year"
    Temporal column used for ordering within groups.
window : int, default 7
    Rolling window size.
min_period : int, default 1
    Minimum periods for the rolling window.
col : str, default "SOC_0_15CM"
    Column to smooth.
grouping : tuple[str, ...] | None, default ("Residue", "Nitrogen")
    Grouping columns for independent smoothing per group. Use ``None`` for
    a global rolling mean.
preserve_start : bool, default True
    If True, keeps the first ``floor(window/2)`` original values unchanged
    to avoid artificial dips from centering.

Returns
-------
pandas.DataFrame
    Copy of input with a new column ``<col>_roll_mean`` containing the smoothed series.

Notes
-----
The rolling mean is centered; when grouping, each group is sorted by ``time_col``.
"""


# carbon_obj
"""
Objective function: negate the mean rolling SOC for optimization routines.

Parameters
----------
df : pandas.DataFrame
    Data containing ``SOC_0_15CM``.

Returns
-------
float
    Negative mean of ``SOC_0_15CM_roll_mean`` (computed via :func:`mva` with default settings).
"""


# yield_obj
"""
Objective function: negate the mean rolling maize yield for optimization.

Parameters
----------
df : pandas.DataFrame
    Data containing ``maizeyield``.

Returns
-------
float
    Negative mean of ``maizeyield_roll_mean`` (computed via :func:`mva` on ``maizeyield``).
"""


# find_soc_equilibrium
"""
Detect SOC equilibrium year based on stability of a rolling mean after its peak.

For each (Nitrogen, Residue) group, compute a yearly rolling mean and define
equilibrium as the **first year after the rolling-mean peak** where the rolling
mean remains within a small band for at least ``min_stable_years`` consecutive
years. The stability band is ``max(eps_abs, eps_rel * peak_value)``.

Parameters
----------
df : pandas.DataFrame
    Input data with at least ``year`` and ``soc_col``.
group_cols : tuple[str, ...], default ("Nitrogen", "Residue")
    Grouping columns to evaluate independently.
year_col : str, default "year"
    Year column. If absent, attempts to infer from a date-like column.
soc_col : str, default "SOC_0_15CM"
    SOC column to analyze.
window : int, default 7
    Rolling window length for the yearly mean.
min_stable_years : int, default 2
    Minimum number of consecutive years (after peak) within the stability band.
center : bool, default False
    Whether to center the yearly rolling mean.
eps_abs : float, default 0.02
    Absolute tolerance for stability (SOC units).
eps_rel : float, default 0.003
    Relative tolerance (fraction of peak value).
return_reason : bool, default True
    If True, includes a ``reason`` column explaining missing equilibria.

Returns
-------
pandas.DataFrame
    One row per group with columns: ``equilibrium_year``, ``equilibrium_value``,
    ``peak_year``, ``peak_value``, and optionally ``reason``.

Notes
-----
If there are insufficient years or all rolling values are NaN, the output row
contains NaNs and a diagnostic ``reason``.
"""


# calculate_soc_changes
"""
Compute absolute and percent changes in SOC between the first and last year.

Parameters
----------
data : pandas.DataFrame | ApsimModel
    DataFrame with SOC and year (or an ApsimModel exposing a 'carbon' table).
col : str
    SOC column to compare (e.g., ``"SOC_0_15CM"``).
grouping : tuple[str, ...], default ("Nitrogen", "Residue", "year")
    Grouping keys used to compute annual means and pivot across the first and
    last year.

Returns
-------
pandas.DataFrame
    Table with columns:
    - ``d<col>`` : absolute change (last - first)
    - ``%<col>`` : percent change relative to the first year
    - ``d<col>-r`` : scaled change (``/ 101``) for downstream heuristics

Raises
------
ValueError
    If no year or date-like column is present and cannot be inferred.
"""


# generate_n_rates
"""
Generate a sorted list of nitrogen rates around a base value.

Parameters
----------
base_n : int | float
    Base nitrogen rate.
deviation : float, default 0.10
    Fractional deviation used to create symmetric variants (±d and ±2d).
    Must be less than 1.

Returns
-------
list[int]
    Sorted list containing canonical rates (e.g., 161, 165, 326, 300, 280,
    base) plus ±d and ±2d multiples of ``base_n`` (rounded to ints).

Raises
------
ValueError
    If ``deviation`` is not in (0, 1].
"""


# test_mva
"""
Render and save a moving-average plot using an object’s ``plot_mva`` method.

Parameters
----------
instance : object
    Object exposing ``plot_mva(**kwargs)``.
title : str, optional
    Figure title.
file_name : str | None, optional
    Output file name; defaults to ``mva_single_{x}-{y}-{table}.png``.
**kwargs
    Passed through to ``instance.plot_mva``; expected to include
    ``time_col``, ``response``, and ``table`` for naming.

Returns
-------
None
    Saves the figure to disk and opens it with the system viewer.
"""


# cat_plot
"""
Render and save a categorical time-series plot using ``instance.cat_plot``.

Parameters
----------
instance : object
    Object exposing ``cat_plot(**kwargs)`` and ``get_simulated_output(table)``.
title : str, optional
    Figure title.
file_name : str | None, optional
    Output file name; defaults to ``series_single_{x}-{y}-{table}.png``.
**kwargs
    Passed to ``instance.cat_plot``. If ``table`` is a string, the table is
    fetched and categorical ordering is enforced for ``Nitrogen`` and ``Residue``.

Returns
-------
None
    Saves the figure to disk and opens it.
"""


# series
"""
Render and save a generic series plot using an object’s ``series_plot`` method.

Parameters
----------
instance : object
    Object exposing ``series_plot(**kwargs)``.
title : str, optional
    Figure title.
file_name : str | None, optional
    Output file name; defaults to ``series_single_{x}-{y}-{table}.png``.
**kwargs
    Passed through to ``instance.series_plot``; expected to include ``time_col``,
    ``y``, and ``table`` for naming.

Returns
-------
None
    Saves the figure to disk and opens it.
"""
