How to run
==================================================
Check the APSIM version in the .env and install it
replace the bin_path with the installed one

# Project Folder Structure
===========================

.. note::
Per request, this documentation includes **only** directories, `.py` files, and Jupyter notebooks (`.ipynb`). It **excludes** `.apsimx`, `.db`, `.bak`, `.db-wal/.db-shm`, `.met`, `.csv`, images, archives, and other file types.

## Conventions
----------------

* Directory names appear as **bold** headings.
* Files are shown as inline code (`like_this.py`) with concise descriptions.

## Directories
-------------------

**.git**
Git metadata for version control.

**.idea**
JetBrains IDE project settings.

**.ipynb_checkpoints**
Jupyter notebook checkpoints (auto-created).

**APSIM2025.8.7844.0**
Pinned assets/configs for APSIM NG version `2025.8.7844.0`. This intended to be the pinned APsim version

**Data-analysis-Morrow-Plots**
Notebooks, code, and outputs related to Morrow Plots analyses. this has files from the previous analysis

**figs**
Working figures generated during analysis.

**Figures**
Curated/publication-ready figures.

**Install**
Installation helpers and setup scripts.

**managers**
APSIM Manager components organized by feature.

**manager_scripts**
Standalone scripts for APSIM operations/automation.

**mets**
Weather data resides here.

**Morrow Plots Observed_Data**
Observational datasets and loaders for Morrow Plots.

**morrows**
Supplemental Morrow-related data/models. Not used if shipped with the project

**Others**
Miscellaneous items not fitting primary categories.

**OutPuts**
all outputs from apsimNGpy touches here.

**APSIMx_base_files\Calibrated Parameters**
On runtime, some management and cultivar apsim inputs are backed up here. The files might eb stored by date

**Results**
Post-processed summaries, figure, simulated results all are found here. Generated during runtime

**Re_ Soil organic carbon cornundrum**
Materials related to SOC email thread exchanged with Dr Fernando

**Manuscripts 20251212**
contain the manuscript and the supporting data as the appendix.

**APSIM2025.8.7844.0**
APSIM version used mainly in model calibration and scenario analysis, other scripts are locked to this version.

**APSIMx_base_files**
Contains calibrated apsimx files, majorly open in APSIM V2024.5.7493. the two critical files are Plot3_7493.apsimx and Plot4_7493.apsimx
for plot 3 and 4, respectively. Each file has two simulation representing NC and NB sub-plots.

**scratch**
Temporary workspace for apsimx files. generated at run time if the out_path argument is not populated

**weatherdata**
Weather datasets/preprocessed inputs. normally data downloaded by apsimNGpy is stored here, depending on the location of the script


## Python Modules (`.py`)
--------------------------

`change_metrics.py`
Utilities to compute event/period change metrics.

`correlation.py`
Correlation analysis and plotting helpers.

`fit_model.py`
Model fitting/calibration utilities (e.g., regression).

`load.py`
Lightweight loaders/initialization routines.

`main.py`
Minimal entry-point/launcher.

`mgt_reporter.py`
Summaries of management operations/scenarios.

`moess.py`
Module (context-specific utilities; name-based).

`morrow_plots_observed_data.py`
Accessors/parsers for Morrow Plots observed data.

`mult-obj.py`
Multi-objective analysis/driver (e.g., Pareto front generation).

`no_date_rolling_regression.py`
Rolling regression without explicit date indexing.

`nrates.py`
Nitrogen-rate helpers and scenario utilities.

`one_run.py`
Single-run driver for quick checks/benchmarks.

`plotting.py`
Plot style wrappers and figure exporters.

`quad.py`
Quadratic/quadratic-plateau modeling helpers.

`read_met.py`
Parsers for meteorological inputs (used by Python workflows).

`results.py`
Result collation and I/O helpers.

`roll_regression.py`
Rolling regression implementation with windows/groups.

`scen.py`
Scenario construction and execution utilities.

`scenario_re_analysis.py`
Scripted re-analysis pipeline for scenarios.

`schemas.py`
Data schemas / validation models (e.g., pydantic).

`seperate.py`
Data separation/partition utilities (filename retains original spelling).

`settings.py`
Runtime constants and configuration flags.
Also acts as a utility script for configuring APSIM NG runs and organizing paths/inputs
for Morrow Plots analyses.

This module:
- Defines project paths (APSIM files, MET files, outputs, observed data).
- Loads environment variables (e.g., APSIM bin path).
- Establishes constant parameter grids (N rates, residue rates, tillage dates, planting dates).
- Prepares color palettes and site coordinates.
- Provides an ObservedData dataclass to read and hold observed data.
- Verifies and sets the APSIM binary path via apsimNGpy config.

`single_N_application_runs.py`
Batch runner for single N application scenarios.

`split_application_runs.py`
Batch runner for split N application scenarios.

`split_application_runs_early.py`
Early-timing variant for split-application runs.

`stat_module.py`
Statistical metrics, tests, and summaries.

`tmp.py`
Temporary scratch utilities.

`tt.py`
Small test/utility script.

`utils.py`
General-purpose helpers used across the project.

`__init__.py`
Package initializer (enables module-style imports from repo root).

## Jupyter Notebooks (`.ipynb`)

`pd (rmagala@iastate.edu).ipynb`
Notebook saved with user tag in filename.

`pd.ipynb`
Compact variant of the `pd` notebook.

`scenario_re_analysis.ipynb`
Interactive notebook for scenario re-analysis.

trends.py
=========
Quick wrapper around apsimNGpy's PlotManager to produce and save
moving-window average (MVA) plots from APSIM simulation outputs.

- Defines a lightweight Plot subclass that holds a copy of the input data
  and exposes matplotlib via `self.plot`.
- `mva_plot(...)` configures labels, palette, and typography; calls
  `Plot.plot_mva` to draw an MVA time-series by group; then applies
  consistent tick/font sizing across facets, adds shared axis labels,
  and writes the figure to `RESULTS` with a generated filename.
- The script section (`__main__`) pulls a table from the APSIM datastore
  and renders several MVA views for SOC (0–15 cm), faceting by Residue
  or Nitrogen and toggling the line hue.

Args (mva_plot):
    data (pd.DataFrame): Tidy table containing APSIM outputs.
    time_col (str): Time column for the MVA (default: 'year').
    table (Any): Passed through to PlotManager (defaults to `data`).
    response (str): Column to smooth/plot (e.g., 'SOC_0_15cm_Mg').
    hue/col/col_wrap: Faceting and color encoding options.
    height/palette/errorbar/window: Visual and smoothing controls.

Side effects:
    - Saves a high-resolution PNG to `RESULTS`.
    - Optionally opens the saved file via `open_file(...)`. open_file is defined in the utils module

qp_model_yield.py
=================
Fits a quadratic model to the average corn grain yield
Created on Sunday Oct 26 2025:

qp_model_soc_balance.py
=======================
Fits a quadratic model to the averageSOC balance, the difference ein SOC between the last and the fist year of the simulation
Created on Sunday Oct 27 2025:

model_evaluation.py
====================
contains code for testing apsim prediction of soil organic carbon and yield from the morrow plots


## Excluded Items (for clarity)

.. admonition:: Excluded by Rule
:class: note

The following types are intentionally **not** documented here,
`.apsimx`,  `.bak`, `.db-wal`/`.db-shm`, `.met`, `.csv`, images (e.g., `.png`), archives (e.g., `.zip`), `.md/.rst` docs, and PDFs.

requirements.txt

contains the project dependencies

.. tip::

  pip install -r requirements # pip managed env only


