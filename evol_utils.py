import pandas as pd
from apsimNGpy.core.apsim import ApsimModel
from apsimNGpy.validation.evaluator import Validate
from utils import edit_cultivar
from organise_nwrec_data import dir_data, out_apsimx, datastore, yield_table, lonlat_sable
from apsimNGpy.core_utils.database_utils import read_db_table
import numpy as np
from functools import lru_cache
from apsimNGpy.exceptions import ApsimRuntimeError
from pprint import pprint

dYield = read_db_table(datastore, yield_table)
dYield['year'] = dYield['year'].astype(int)
dYield['plotid'] = dYield['plotid'].astype(int).astype(str)
dYield.set_index(['year', 'plotid'], inplace=True)
dYield['corn_grain_yield_Mg'] = dYield['corn_grain_yield_kg'] / 1000

base = {"model": None}  # set to ApsimModel instance or path string elsewhere


def func(x=None):
    """
    Run APSIM, compute yield & SOC metrics, and return an objective value.

    Expects:
      - base["model"] to be either an ApsimModel instance or a path (str)
      - dYield, Validate, dir_data, edit_cultivar, ApsimModel to be available in scope
    """
    apsim_model = base.get("model")

    # Instantiate if a path string was provided
    if isinstance(apsim_model, str):
        apsim_model = ApsimModel(apsim_model)
    elif apsim_model is None:
        raise ValueError("No base model provided: set base['model'] to an ApsimModel or a .apsimx path.")

    try:
        # Optional: apply cultivar & soil C edits from decision vector x
        if x is not None:
            x = list(x)
            edit_cultivar(apsim_model, x)
            if len(x) >= 2:
                apsim_model.edit_model(model_type="Organic", model_name="Organic",
                                       FBiom=x[-1], FInert=x[-2])

        # --- Yield metrics ---
        predicted = apsim_model.run(report_name="yield", verbose=False).results.copy()
        predicted.eval("apsim = grainwt / 1000.0", inplace=True)
        predicted["Plotid"] = predicted["Plotid"].astype(str)
        predicted["plotid"] = predicted["Plotid"]  # normalize name
        predicted.set_index(["year", "plotid"], inplace=True)

        rdata = dYield.join(predicted, how="inner")
        val = Validate(rdata["corn_grain_yield_Mg"], rdata["apsim"])
        yield_metrics = {k: float(v) for k, v in val.evaluate_all().items()}

        # (Unused here but left for clarity)
        yield_min_func = ((yield_metrics.get("CCC", 0.0) * -100.0) +
                          (yield_metrics.get("R2", 0.0) * -100.0)) / 2.0

        # --- Carbon metrics ---
        aps = apsim_model.get_simulated_output("Carbon").copy()
        aps["plotid"] = aps["SIM_ID"].astype(float)

        cb = pd.read_csv(dir_data / "nwrec_carbon_2.csv")
        obs_pred_carbon = aps.merge(cb, on=["plotid", "year"])
        obs_pred_carbon.eval("soc_from_apsim = SOC1 / (BD * depth)", inplace=True)

        ev = Validate(obs_pred_carbon["Carbon"], obs_pred_carbon["soc_from_apsim"])

        rmse = yield_metrics.get("RMSE", float("inf"))
        if x is None or rmse < 1.6:
            pprint({"yield_metrics": yield_metrics})

        print(yield_metrics)

        # Objective to minimize: return RRMSE (keep your original choice)
        return yield_metrics.get("RRMSE", float("inf"))

    finally:
        # Always try to release DB locks & temporary artifacts
        try:
            apsim_model.clean_up(db=True)
        except Exception:
            pass


def cache_objective(func, *, dtype=np.float64, round_digits=None, static_args=()):
    """
    Wrap `func(x, *extra)` so `x` (np.ndarray) becomes a hashable cache key.
    - round_digits: if set, rounds x before hashing (helps repeat hits).
    - static_args: tuple of fixed args to include in the cache key.
    """

    @lru_cache(maxsize=50)
    def _cached(key):
        dtype_str, shape, buf, extra = key
        x = np.frombuffer(buf, dtype=np.dtype(dtype_str)).reshape(shape)
        return func(x, *extra)

    def wrapped(x, *extra):
        xx = np.asarray(x, dtype=dtype)
        if round_digits is not None:
            xx = xx.round(round_digits)
        key = (xx.dtype.str, xx.shape, xx.tobytes(), tuple(extra) + tuple(static_args))
        return _cached(key)

    return wrapped


track_solutions = {}


def evaluate_objectives(x):
    from inspect import unwrap
    x = np.round(x, 2)
    tuple_x = tuple(x)
    if track_solutions.get(tuple_x):
        return track_solutions[tuple_x]
    try:
        ans = func(x)
        if np.abs(ans) < 0.18:
            track_solutions[tuple(x)] = ans
            print(ans, ':', [float(i) for i in x])
        return ans
    except ApsimRuntimeError:
        print(f'We encountered an error while running {x}')
        return float(max(track_solutions.values())) or 1e12

    # This function runs APSIM and compares the predicted maize yield results with observed data.
