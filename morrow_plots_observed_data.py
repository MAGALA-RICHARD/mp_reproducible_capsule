from pandas import read_csv
from pathlib import Path
import shutil, os
# create dir to keep the measured data

MP_Observed_Data = base / 'Morrow Plots Observed_Data'
MP_Observed_Data.mkdir(exist_ok=True)
bb = Path(
    r"C:\Users\rmagala\OneDrive\simulations\objective_2\illnois\DATA\APSIM_WORKSPACE_FILES\Morrow_plots_new_calibration\Integrated_morrow_plotsSim")
ob = bb.rglob('*after_utc.csv')
for i in list(ob):
    shutil.copy(i, MP_Observed_Data/i.name)
    print(i)
    os.startfile(i)
