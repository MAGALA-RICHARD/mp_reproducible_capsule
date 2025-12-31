"""
Utility script for configuring APSIM NG runs and organizing paths/inputs
for Morrow Plots analyses.

This module:
- Defines project paths (APSIM files, MET files, outputs, observed data).
- Loads environment variables (e.g., APSIM bin path).
- Establishes constant parameter grids (N rates, residue rates, tillage dates, planting dates).
- Prepares color palettes and site coordinates.
- Provides an ObservedData dataclass to read and hold observed data.
- Verifies and sets the APSIM binary path via apsimNGpy config.

Modified on: 10/20/2025
"""
import os
import seaborn as sns
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from apsimNGpy.core.config import set_apsim_bin_path, get_apsim_bin_path
from dotenv import load_dotenv
from loguru import logger
from pandas import read_csv
import sys

SEED = 40  # for reproducibility
TILLAGE_DEPTH = 150
# Please do not change these variable names
BASE = Path(__file__).parent
# path to the apsim file
path_to_MP_data = Path.cwd() / 'Data-analysis-Morrow-Plots/APSIMX FILES'
# path to the met file
met_file = path_to_MP_data / 'urbana_mp_128yrs_of_met.met'
# path to the outputs
scratch_DIR = Path.cwd() / 'OutPuts'
scratch_DIR.mkdir(parents=True, exist_ok=True)
# path to csv files observed data
MP_Observed_Data = BASE / 'Morrow Plots Observed_Data'
nr = range(100, )

nr = [int(i) for i in nr]
nr.append(165)
nr.append(142)
nr = ",".join(map(str, nr))
N_RATES = "100, 186, 200, 244, 326"  # "100, 186,209, 233, 256, 279"
residue_rate_aray = np.linspace(0.1, .90, 4).round(2)
RESIDUE_RATES = ', '.join(map(str, residue_rate_aray))

TILLAGE_DATES = "1-Nov, 10-apr"

PD = "4,5,6, 7, 8.23,9,9.5,10"

load_dotenv()

palette = (
    "tan",  # blue
    "orange",  # orange
    "green",  # green
    "red",  # red
    'darkorchid'
)

# ___________ site,lon,lat __________________ from Hanna J. Poffenbarger et al. (2017)
Central = -93.783333, 42.016667
Southeast = -91.483333, 41.183333
South = -93.416667, 40.966667
Northwest = -95.533333, 42.916667

sns.set_style("darkgrid", rc={"font.family": "DejaVu Sans"})


class ConfigurationError(Exception):
    pass


# bin_path = BASE / 'APSIM2025.8.7844.0'

bin_path = os.getenv('7844')
# _________________________________
# validate presence of the .env file
# __________________________________

env_path = Path('.env').resolve()
bin_str = ("This file must be defined as follow:",
           '=======================================',
           '7844 = APSIM2025.8.7844.0', '7493 = APSIM2024.5.7493.0', '7939 = APSIM2025.12.7939.0')
if not env_path.exists():
    logger.info(f"Environment file '.env' is missing at this location; `{env_path.parent}`")
    print(*bin_str, sep='\n')
    sys.exit(1)
if not bin_path or not env_path:
    logger.info(
        f"please provide path in  to to apsim versions as:\n  7844 = APSIM2025.8.7844.0\n 7493 = APSIM2024.5.7493.0\n7939 = APSIM2025.12.7939.0")
    sys.exit(1)
else:

    if Path(bin_path).resolve() != Path(get_apsim_bin_path()).resolve():  # set it once
        set_apsim_bin_path(bin_path)
    # otherwise ignore


@dataclass(slots=True, frozen=True)
class ObservedData:
    """
    Observed data is recalculated here
    """
    # finally, not used
    # convert
    bd = 1.45 / 10  # plot 4 and 3
    bd5 = 1.39 / 10  # plot 5
    layer_thickness = 15
    plot3nb = read_csv(MP_Observed_Data / 'CC_npk_after_utc.csv')
    npk = read_csv(MP_Observed_Data / "npk data.csv")


# ______________________________________
# create directories needed at runtime
# _______________________________________

RESULTS = BASE / 'Results'
LOG_PATH = BASE / 'logs'
LOG_PATH.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
large = RESULTS / 'single_model_N_50 to 340 step 5.db'
