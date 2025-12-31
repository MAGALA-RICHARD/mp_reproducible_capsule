from apsimNGpy.core_utils.database_utils import read_db_table
from typing import Union, Sequence, Tuple, List
from pathlib import Path
import pandas as pd
from logger import get_logger
from settings import RESULTS
from apsimNGpy.core_utils.database_utils import read_db_table
logger = get_logger(__name__)


from pathlib import Path
from shutil import copy2
from settings import RESULTS
from apsimNGpy.core_utils.database_utils import read_db_table

def read_db(db, table):
    """
    Copy <db> to a read-only scratch copy and read <table> into a DataFrame.

    Parameters
    ----------
    db : str | Path
        Path to the SQLite datastore (with or without .db suffix).
    table : str
        Table name to read.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    FileNotFoundError : if the source DB does not exist.
    RuntimeError      : if copy/read fails or the table is missing.
    """
    src = Path(db)
    if src.suffix != ".db":
        src = src.with_suffix(".db")
    if not src.exists():
        raise FileNotFoundError(f"Database not found: {src}")

    base = src.stem
    RESULTS.mkdir(parents=True, exist_ok=True)

    tmp = RESULTS / f"{base}_{table}_datastore_readonly.db"
    try:
        tmp.unlink(missing_ok=True)
        copy2(src, tmp)
        if not tmp.exists():
            raise RuntimeError(f"Failed to create scratch copy: {tmp}")

        df = read_db_table(tmp, table)
        if df is None or getattr(df, "empty", False) and table not in ("sqlite_master",):
            # read_db_table should raise if table missing; this is an extra guard
            raise RuntimeError(f"Table '{table}' not found or empty in {tmp}")
        return df

    except Exception as e:
        # Surface a clear message while keeping the original exception context
        raise RuntimeError(f"Unable to read table '{table}' from DB '{src}': {e}") from e



def merge_tables(
        data: Union[
            str, Path, "ApsimModel", "ExperimentManager", Tuple[pd.DataFrame, pd.DataFrame], List[pd.DataFrame]],
        tables: Sequence[str],
        how: str = "inner",
):
    """
    Merge two APSIM output tables on common keys and derive convenience fields.

    This helper accepts either:
    1) a path/filename to an APSIM results database (from which the two tables
       will be read), or
    2) an :class:`~apsimNGpy.core.apsim.ApsimModel` /
       :class:`~apsimNGpy.core.experimentmanager.ExperimentManager` instance
       (from which the two tables will be fetched via
       :meth:`get_simulated_output`), or
    3) a pair of preloaded :class:`pandas.DataFrame` objects.

    The two input tables are merged on the shared APSIM keys
    ``('SimulationID', 'Nitrogen', 'Residue', 'year')``. After merging,
    numeric convenience columns and derived metrics are added:

    - ``R``: float-cast of ``Residue``
    - ``N``: float-cast of ``Nitrogen``
    - ``Residue`` and ``Nitrogen`` recast as ordered categoricals
    - ``Incorporated_Biomass`` = ``BBiomass`` + ``(total_biomass - grainwt) * R``
    - ``Residue_Biomass``    = ``BBiomass`` + ``(total_biomass - grainwt)``
    - ``cnr``                = ``SurfaceOrganicMatter_Carbon / SurfaceOrganicMatter_Nitrogen``

    Parameters
    ----------
    data : str | pathlib.Path | ApsimModel | ExperimentManager | tuple[pd.DataFrame, pd.DataFrame] | list[pd.DataFrame]
        Source of the two tables. If a path/filename is provided, the function
        expects a helper ``read_db_table(path, table_name)`` to be available
        in scope. If an APSIM model/manager is provided, it must expose
        :meth:`get_simulated_output`.
    tables : Sequence[str]
        Names of the **two** tables to merge (e.g., ``("yield", "carbon")``).
        Must be a non-string iterable of length 2.
    how : {"inner", "left", "right", "outer"}, default "inner"
        Merge strategy passed through to :meth:`pandas.DataFrame.merge`.

    Returns
    -------
    pandas.DataFrame
        The merged table with additional convenience and derived columns
        described above.

    Raises
    ------
    ValueError
        If ``tables`` is not a non-string iterable of length two,
        if ``data`` is not one of the supported types,
        or if a tuple/list does not contain exactly two
        DataFrames.

    Notes
    -----
    - This function assumes the presence of columns required for the derived
      expressions (``BBiomass``, ``total_biomass``, ``grainwt``,
      ``SurfaceOrganicMatter_Carbon``, ``SurfaceOrganicMatter_Nitrogen``).
      If they are missing, :class:`KeyError` will be raised by pandas.
    - The categorical casting of ``Residue`` and ``Nitrogen`` preserves their
      ordering for plotting and grouped analyses.

    Examples
    --------
    Merge two APSIM output tables read from a results database:

    .. code-block:: python

       dat = merge_tables("results/single_model.db", ("carbon", "yield"))

    Merge when you already have two DataFrames:

    .. code-block:: python

       dat = merge_tables((df_carbon, df_yield), ("carbon", "yield"), how="left")
    """
    # Local imports to avoid heavy top-level dependencies in some environments
    from apsimNGpy.core.apsim import ApsimModel

    if len(tables) != 2:
        raise ValueError("Only two tables can be merged")
    if isinstance(tables, str):
        raise ValueError("`tables` must be a non-string iterable of two names")

    # Resolve inputs to two DataFrames
    if isinstance(data, (str, Path)):
        # Expect a helper in scope:
        #   read_db_table(path: str|Path, table_name: str) -> pd.DataFrame
        df1 = read_db_table(data, tables[0])  # type: ignore[name-defined]
        df2 = read_db_table(data, tables[1])  # type: ignore[name-defined]
    elif hasattr(data, "get_simulated_output"):
        logger.info('Data loaded from apsimNGpy objects')
        df1 = data.get_simulated_output(tables[0])
        df2 = data.get_simulated_output(tables[1])
    elif isinstance(data, (list, tuple)):
        if len(data) != 2:
            raise ValueError("When providing DataFrames, supply exactly two dfs")
        df1, df2 = data  # type: ignore[assignment]
    else:
        raise ValueError(f"Unsupported data type: {type(data)!r}")

    commons = ["SimulationID", "Nitrogen", "Residue", "year"]
    if 'timing' in df1.columns:
        commons.append("timing")
    dat = df1.merge(df2, on=commons, how=how)

    # Convenience numeric fields and ordered categorical data
    dat["R"] = dat["Residue"].astype(float)
    dat["N"] = dat["Nitrogen"].astype(float)
    dat.eval('n_r= R*N', inplace=True)
    dat["Residue"] = pd.Categorical(dat["Residue"], ordered=True)
    dat["Nitrogen"] = pd.Categorical(dat["Nitrogen"], ordered=True)

    # Derived metrics
    dat.eval("Incorporated_Biomass = BBiomass + ((total_biomass - grainwt) * R)", inplace=True)
    dat.eval("Incorporated_Biomass_Mg = Incorporated_Biomass/1000", inplace=True)
    dat.eval('Below_ground_biomass_Mg =  BBiomass', inplace=True)
    dat.eval('Surf_Org_Carbon_Mg = SurfaceOrganicMatter_Carbon/1000', inplace=True)
    dat.eval("Residue_Biomass = BBiomass + (total_biomass - grainwt)", inplace=True)
    dat.eval("Residue_Biomass_Mg = Residue_Biomass/1000", inplace=True)
    dat.eval('corn_yield_Mg = grainwt/1000', inplace=True)
    dat.eval('SOC_0_15cm_Mg=SOC_0_15CM', inplace=True)
    dat.eval('SOC1=SOC_0_15CM', inplace=True)
    dat.eval('total_biomass_Mg = total_biomass/1000', inplace=True)
    dat.eval('SOC_balance = SOC_0_15cm_Mg - 52.25', inplace=True)

    dat.eval('_ce = SOC_0_15cm_Mg/top_carbon_mineralization * SOC_0_15cm_Mg', inplace=True)
    dat.eval("cnr = SurfaceOrganicMatter_Carbon / SurfaceOrganicMatter_Nitrogen", inplace=True)
    logger.info('Tables merged successfully!')

    return dat
