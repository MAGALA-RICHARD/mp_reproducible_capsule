"""
Created on 10/21/2025
"""
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd
from data_manager import merge_tables
from logger import get_logger
from settings import RESIDUE_RATES, RESULTS, LOG_PATH, TILLAGE_DEPTH
from utils import create_experiment, N_RATES
from utils import edit_cultivar, xc, load_manifest

cfg = load_manifest()

path_to_MP_data = Path.cwd() / 'Data-analysis-Morrow-Plots/APSIMX FILES'
scratch_DIR = Path.cwd() / 'OutPuts'
from datetime import datetime as dtm

DPI = 600
scratch_DIR.mkdir(parents=True, exist_ok=True)
# set_apsim_bin_path(r"C:\Users\rmagala\AppData\Local\Programs\APSIM2025.1.7643.0\bin")
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

data_db_save = RESULTS / 'scenarios_db_data.db'
script_name = Path(__file__).stem
logger = get_logger(name='{}'.format(script_name), log_dir=LOG_PATH)


# writes to ./morrow_plots_simulation_logs.log
def spin_up(model, table, params, time_col='year'):
    if not isinstance(params, Iterable):
        raise ValueError('params must be an iterable of dict')
    model.edit_model(model_type='Organic', model_name='Organic', FBiom=xc[-1], FInert=xc[-2])
    for param in params:
        simulations = param.get('simulation') or param.get('simulations')
        start = param.get('start') or param.get('Start') or param.get('start_date')
        end = param.get('end') or param.get('End') or param.get('end_date')
        p_aram = [start, end, simulations]
        assert all(p_aram), 'All parameters are required'
        date = model.inspect_model_parameters(model_type='Models.Clock', model_name='Clock', simulations=simulations)

        model.edit_model(model_type='Models.Clock', model_name='Clock', start_date=start, end_date=end)

        print('succeeded')
    model.run(table)
    dfd = model.results
    carbon = dfd.SurfaceOrganicMatter_Carbon
    n = dfd.SurfaceOrganicMatter_Nitrogen
    cnr = carbon / n
    dfd['cnr'] = cnr
    dm = dfd[dfd[time_col] == dfd[time_col].max()]
    cn = dm['cnr'].iloc[0]

    model.edit_model(model_type='Models.Surface.SurfaceOrganicMatter', model_name='SurfaceOrganicMatter', InitialCNR=cn)
    model.edit_model(model_type='Models.Clock', model_name='Clock', start_date='01-01-1904', end_date='12-31-2005')

    # model.edit_model('Organic', 'Organic', )


# _____________data storage variable names for importing else where___________________________
file_name = Path(__file__).stem
datastore = RESULTS / f'{file_name}.db'
table_name = f'{file_name}'
from apsimNGpy.core.experimentmanager import ExperimentManager

if __name__ == '__main__':
    logger.info(
        "Simulation started\n ============================================================================\n\n ")
    logger.info(
        f"running experiment:\n {RESIDUE_RATES} residue  rates {N_RATES} N rates \n and N  ")

    scenario_model = create_experiment(base_file='one_file.apsimx', lonlat=None, start =1905, end=2005, # lonlatTest,
                                       out_path=path_to_MP_data / 'out_scenario_dat.apsimx')

    # scenario_model.edit_model(model_type='Models.Soils.Organic', model_name='Organic', FInert=0.55, FBiom =[0.035,
    # 0.035]) add nitrogen levels 0, 165, 244, 326
    scenario_model.add_factor(specification=f"[fertilize in phases].Script.Amount = {N_RATES}", factor_name='Nitrogen')

    scenario_model.add_factor(specification=f"[Tillage].Script.Fraction = {RESIDUE_RATES}", factor_name='Residue')
    # some must be constants are set here to avoid confusion
    scenario_model.edit_model(model_type='Models.Manager', model_name='Tillage', Depth=TILLAGE_DEPTH)
    scenario_model.edit_model(model_type='Models.Manager', model_name='fertilize in phases', FractionToApplyFirst=1,
                              FertiliserType='UreaN')

    try:
        # scenario_model.edit_model(
        #     model_type='Cultivar',
        #     simulations='all',
        #     commands=['[Grain].MinimumNConc.FixedValue',],
        #     values=[0.009, ],
        #     new_cultivar_name='B_110-e',
        #     model_name='Pioneer_33M54',
        #     cultivar_manager='Sow using a variable rule')
        # scenario_model.preview_simulation(watch=True)
        # xc[2] = 240
        from utils import cmds


        # params = {
        #     "[Leaf].Photosynthesis.RUE.FixedValue": 2,
        #      "[Phenology].GrainFilling.Target.FixedValue": 815,
        #     "[Grain].MaximumGrainsPerCob.FixedValue": 770,
        #     "[Phenology].FloweringToGrainFilling.Target.FixedValue": 215,
        #     "[Phenology].MaturityToHarvestRipe.Target.FixedValue": 100,
        #     "[Maize].Grain.MaximumPotentialGrainSize.FixedValue": 0.867411373063701,
        #     "[Grain].MaximumNConc.InitialPhase.InitialNconc.FixedValue": 0.05,
        #     '[Maize].Root.SpecificRootLength.FixedValue': 135,
        #     '[Maize].Root.RootFrontVelocity.PotentialRootFrontVelocity.PreFlowering.RootFrontVelocity.FixedValue': 22,
        #     '[Rachis].DMDemands.Structural.DMDemandFunction.MaximumOrganWt.FixedValue': 36
        # }
        from apsim_validation_nwrec import params

        scenario_model.edit_model(model_type="Manager", model_name='Sow using a variable rule',
                                  CultivarName='B_110')
        # scenario_model.edit_model_by_path('.Simulations.Replacements.Maize.CultivarFolder.Generic.B_110',
        #                                   commands=params.keys(),
        #                                   values=params.values(),
        #                                   sowed=True,
        #                                   )
        scenario_model.run('carbon', verbose=True)
    except FileNotFoundError as ie:
        logger.error(ie, exc_info=True)

        raise ie

    scenario_model.results.to_csv(scratch_DIR / f'{TILLAGE_DEPTH}_one grand simulation.csv', index=False)
    logger.info(f'saved simulated results as csv to: {scratch_DIR / f'{TILLAGE_DEPTH}_one grand simulation.csv'}')
    df_soc_split = scenario_model.get_simulated_output('mineralization')
    df_soc_split['timing'] = SPLIT_METHOD_STR
    df_biomass_split = scenario_model.get_simulated_output('yield')
    df_biomass_split['timing'] = SPLIT_METHOD_STR
    logger.info('succeeded running')

    from utils import test_mva

    d = scenario_model.get_simulated_output('carbon')
    dam = d
    test_mva(scenario_model, table=dam, title='Moving average SOC', response='min_ratio',
             expression="min_ratio =  SOC_0_15CM", time_col='year',
             grouping=("Residue", 'Nitrogen'), estimator='mean',
             errorbar=None, col='Residue', col_wrap=2, hue="Nitrogen")

    # single_model.preview_simulation()
    data = merge_tables(scenario_model, ['carbon', 'yield'])
    store_this = merge_tables(scenario_model, ['carbon', 'yield'])

    from sqlalchemy import create_engine

    engine = create_engine(f'sqlite:///{str(datastore)}')

    rows_saved = store_this.to_sql(table_name, con=engine, if_exists='replace', index=False)
    logger.info(f"saved {rows_saved} to table {table_name} to db:{datastore}")
    # test_mva(scenario_model, table='yield', title='Moving average SOC', response='total_biomass_Mg',
    #          expression="total_biomass_Mg =  total_biomass-grainwt", time_col='year',
    #          grouping=("Residue", 'Nitrogen'), estimator='mean',
    #          errorbar=None, col='Residue', col_wrap=2, hue="Nitrogen")

    data_split = scenario_model.get_simulated_output('carbon')
    data['cnr'] = data['SurfaceOrganicMatter_Carbon'] / data['SurfaceOrganicMatter_Nitrogen']
    data_split['cnr'] = data_split['SurfaceOrganicMatter_Carbon'] / data_split['SurfaceOrganicMatter_Nitrogen']

    data['N'] = pd.Categorical(data['Nitrogen'], ordered=True)
    data['R'] = pd.Categorical(data['Residue'], ordered=True)

    data_split['N'] = pd.Categorical(data_split['Nitrogen'], ordered=True)
    data_split['R'] = pd.Categorical(data_split['Residue'], ordered=True)

    from mgt_reporter import add_run_memo
    from mgt_reporter import make_mgt_text

    txt = make_mgt_text(scenario_model)
    scenario_model.add_memo(txt)
    from utils import calculate_soc_changes

    calculate_soc_changes(data, col="SOC_0_15CM")
    single_soc_changes = calculate_soc_changes(data, col="SOC_0_15CM")

    single_soc_changes['timing'] = 'Single'
    split_soc_changes = calculate_soc_changes(data_split, col="SOC_0_15CM")
    split_soc_changes['timing'] = 'Split'
    ch = pd.concat([single_soc_changes, split_soc_changes])

    # view(single_model.get_simulated_output('carbon_change'))
    from apsimNGpy.core.config import apsim_version

    logger.info(f"Simulated using APSIM version: {apsim_version()}")
    # scenario_model.preview_simulation()
    from apsimNGpy.core.model_tools import ModelTools, CastHelper
    import Models

    fv = ModelTools.find_child(scenario_model.Simulations, Models.Functions.Constant, 'ShootLag')
    fv = CastHelper.CastAs[Models.Functions.Constant](fv)
    fv.set_FixedValue(100)
    scenario_model.save()

#
