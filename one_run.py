import os
import shutil
from logger import get_logger
from pathlib import Path
import pandas as pd
from apsimNGpy.core.apsim import ApsimModel
from utils import generate_n_rates
from settings import RESIDUE_RATES, RESULTS
from mgt_reporter import make_report, make_mgt_text
import matplotlib.pyplot as plt
from utils import create_experiment, merge_tables, N_RATES
from utils import open_file
from xlwings import view
from apsimNGpy.manager.weathermanager import read_apsim_met
from settings import Central as South
from utils import TIMING_RATES
path_to_MP_data = Path.cwd() / 'Data-analysis-Morrow-Plots/APSIMX FILES'
scratch_DIR = Path.cwd() / 'OutPuts'
from mgt_reporter import summarize_outputs
import seaborn as sns
from datetime import datetime as dtm
from apsimNGpy.core.config import set_apsim_bin_path
TILLAGE_DEPTH = 150  # mm
DPI = 600
scratch_DIR.mkdir(parents=True, exist_ok=True)
sns.set(style="whitegrid")
set_apsim_bin_path(r"C:\Users\rmagala\AppData\Local\Programs\APSIM2025.1.7643.0\bin")
figures = Path(__file__).parent / 'Figures'
figures.mkdir(parents=True, exist_ok=True)
Y_FONTSIZE = 18
X_FONTSIZE = 18
SINGLE_METHOD_STR = 'Single'
SPLIT_METHOD_STR = 'Split'
n_rates = N_RATES.split(',')
time_STAMP = dtm.now().strftime('%Y%m')
palette = (
    "tan",  # blue
    "orange",  # orange
    "green",  # green
    "red",  # red
    'darkorchid'
)

COLOR_PALETTE = 'deep'
previously_tested = ((-88.226111, 40.104167),
                     (-95.114313, 42.131437), (-89.999216, 43.762696))
lonlatTest = -88.226111, 40.104167

logger = get_logger()  # writes to ./morrow_plots_simulation_logs.log
logger.info("Simulation started\n ============================================================================\n\n ")
logger.info(f"running no split fertilizer model under:\n {RESIDUE_RATES} residue  rates {N_RATES} N rates \n and N timing = {TIMING_RATES} ")
model2 = create_experiment(base_file='one_file.apsimx', lonlat=lonlatTest)

# model2.edit_model(model_type='Models.Soils.Organic', model_name='Organic', FInert=0.55, FBiom =[0.035, 0.035])
# add nitrogen levels 0, 165, 244, 326
model2.add_factor(specification=f"[fertilize in phases].Script.Amount = {N_RATES}", factor_name='Nitrogen')
model2.add_factor(specification=f"[fertilize in phases].Script.FractionToApplyFirst = {TIMING_RATES}", factor_name='timing')
# add residue removal levels, 0, 0.5, 0.75, 1 as fraction
model2.add_factor(specification=f"[Tillage].Script.Fraction = {RESIDUE_RATES}", factor_name='Residue')
# single_model.add_factor(specification=f"[Tillage].Script.Depth = 100, 150, 200, 250", factor_name='Depth')
model2.edit_model(model_type='Models.Manager', model_name='Tillage', Depth=TILLAGE_DEPTH)

# model2.add_factor(specification=f"[Tillage].Script.TillageDate = {TILLAGE_DATES}", factor_name='TillageDate')
# to avoid changes from gui edit immediately
#model2.edit_model(model_type='Models.Manager', model_name='fertilize in phases', AmountType=False)

try:
    model2.run('carbon')
except FileNotFoundError as ie:
    logger.error(ie, exc_info=True)
    model2.preview_simulation()
    raise ie
model2.results.to_csv(scratch_DIR / f'{TILLAGE_DEPTH}_one grand simulation.csv', index=False)
logger.info(f'saved simulated results as csv to: {scratch_DIR / f'{TILLAGE_DEPTH}_one grand simulation.csv'}')
df_soc_split = model2.get_simulated_output('mineralization')
df_soc_split['timing'] = SPLIT_METHOD_STR
df_biomass_split = model2.get_simulated_output('yield')
df_biomass_split['timing'] = SPLIT_METHOD_STR
logger.info('succeeded running')

mineralized_yl = fr"Mineralized nitrogen ($\mathrm{{kg\,ha^{{-1}}}}$), 5-year rolling mean"

soc_mg_yl = fr"SOC ($\mathrm{{Mg\,ha^{{-1}}}}$; 0–{TILLAGE_DEPTH} mm; BD = 1.45 $\mathrm{{g\,cm^{{-3}}}}$), 5-year rolling mean"
som_kg_yl = r"SOM ($\mathrm{kg\,ha^{-1}}$; 0–150 mm; BD = 1.45 $\mathrm{g\,cm^{-3}}$), 5-year rolling mean"
shutil.copy(model2.datastore, RESULTS / 'one_single_split_single.db')
logger.info(f'\n Saved data to: {RESULTS / 'one_single_split_single.db'}\n ======================')
logger.info(f"Saved data is for for:`{N_RATES}`, {TIMING_RATES}, `{TILLAGE_DEPTH}`, `{RESIDUE_RATES} `")
# sg.results.to_csv(RESULTS/'gd.csv')
if __name__ == '__main__':
    from utils import series, test_mva, cat_plot

    # single_model.preview_simulation()
    # test_mva(model2, table='carbon', response='SurfaceOrganicMatter_Carbon', time_col='year',
    #          grouping=("Residue", 'Nitrogen'), estimator='mean',
    #          errorbar=None, col='Residue', col_wrap=2, hue="Nitrogen")
    d = model2.get_simulated_output('carbon')
    dam = d
    test_mva(model2, table=dam, title='Moving average SOC', response='min_ratio',
             expression="min_ratio =  SOC_0_15CM", time_col='year',
             grouping=("Residue", 'Nitrogen'), estimator='mean',
             errorbar=None, col='Residue', col_wrap=2, hue="Nitrogen")

    # single_model.preview_simulation()
    data = merge_tables(model2, ['carbon', 'yield'])
    data_split = model2.get_simulated_output('carbon')
    data['cnr'] = data['SurfaceOrganicMatter_Carbon'] / data['SurfaceOrganicMatter_Nitrogen']
    data_split['cnr'] = data_split['SurfaceOrganicMatter_Carbon'] / data_split['SurfaceOrganicMatter_Nitrogen']

    # test_mva(model2, table=data, response='cnr', time_col='year', window=5,
    #          grouping=("Residue", 'Nitrogen'), estimator='mean',
    #          errorbar=None, col='Residue', col_wrap=2, hue="Nitrogen")

    test_mva(model2, table='carbon', response='microbial_carbon', time_col='year', hue="Nitrogen",
             grouping=("Residue", "Nitrogen", 'timing'), estimator='mean',
             errorbar=None, col='Residue', col_wrap=2, )

    data['N'] = pd.Categorical(data['Nitrogen'], ordered=True)
    data['R'] = pd.Categorical(data['Residue'], ordered=True)

    data_split['N'] = pd.Categorical(data_split['Nitrogen'], ordered=True)
    data_split['R'] = pd.Categorical(data_split['Residue'], ordered=True)

    from mgt_reporter import add_run_memo

    add_run_memo(model2)

    from utils import calculate_soc_changes

    calculate_soc_changes(data, col="SOC_0_15CM")
    single_soc_changes = calculate_soc_changes(data, col="SOC_0_15CM")
    single_soc_changes['timing'] = 'Single'
    split_soc_changes = calculate_soc_changes(data_split, col="SOC_0_15CM")
    split_soc_changes['timing'] = 'Split'
    ch = pd.concat([single_soc_changes, split_soc_changes])

    # view(single_model.get_simulated_output('carbon_change'))
    yd = model2.get_simulated_output('yield')
    yd['R'] = yd['Residue'].astype(float)
    yd.eval('Biomass = (total_biomass - grainwt) * R', inplace=True)
    yd.eval('RBiomass =BBiomass + ((total_biomass - grainwt) * R)', inplace=True)
    yd.eval('RB =BBiomass + ((total_biomass - grainwt))', inplace=True)

#     test_mva(model2, table=yd, title='Residue Biomass', expression=None, response='RBiomass', time_col='year',
#              grouping=("Residue", 'Nitrogen'), estimator='mean',
#              errorbar=None, col='Residue', col_wrap=2, hue="Nitrogen")
#
#     yd['Nitrogen'] = pd.Categorical(yd['Nitrogen'], ordered=True)
#     yd.sort_values(by='Residue', ascending=True, inplace=True)
#
    cat_plot(model2, table=yd, title='RE Biomass', expression=None, y='RB', x='Nitrogen',
             estimator='mean', kind='box', palette='tab10',
             errorbar=None, hue="Nitrogen", showfliers=False)
#
#     cat_plot(model2, table=yd, title='Below Ground Biomass', expression=None, y='BBiomass', x='Residue',
#              estimator='mean', kind='box', palette='tab10',
#              errorbar=None, hue="Nitrogen", showfliers=False)
#
#     carbon = model2.get_simulated_output('carbon')
#     # carbon.sort_values(by='')
#     carbon.sort_values(by=['Residue', 'Nitrogen'], inplace=True)
#
#     cat_plot(model2, table=data, title='SOC', expression=None, y='cnr', x='Residue',
#              estimator='mean', kind='box', palette='tab10', showfliers=False,
#              errorbar=None, hue="Nitrogen")
#
# # carbon mineralisation
# soil carbon
# grain yield
#
