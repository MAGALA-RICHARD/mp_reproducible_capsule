from pathlib import Path
from data_manager import merge_tables
from logger import get_logger
from settings import RESIDUE_RATES, RESULTS, LOG_PATH, path_to_MP_data, TILLAGE_DEPTH  # mm
from utils import TIMING_RATES, QN_RATES
from utils import create_experiment
import seaborn as sns
from datetime import datetime as dtm
from apsimNGpy.core.config import set_apsim_bin_path
from sqlalchemy import create_engine
from utils import edit_cultivar, xc, cmds, load_manifest

config_data = load_manifest()

path_to_MP_data = Path.cwd() / 'Data-analysis-Morrow-Plots/APSIMX FILES'
scratch_DIR = Path.cwd() / 'OutPuts'

data_base_qp_gen = RESULTS / 'qp_db_input_dataset.db'
DPI = 600
scratch_DIR.mkdir(parents=True, exist_ok=True)
sns.set(style="whitegrid")

figures = Path(__file__).parent / 'Figures'
figures.mkdir(parents=True, exist_ok=True)
Y_FONTSIZE = 18
X_FONTSIZE = 18
SINGLE_METHOD_STR = 'Single'
SPLIT_METHOD_STR = 'Split'
n_rates = QN_RATES.split(',')
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
script_name = Path(__file__).stem
# data storage info
file_name = Path(__file__).stem
datastore = RESULTS / f'{file_name}.db'
table_name = f'{file_name}'
apsimx_file_out_path = f"{path_to_MP_data}/{file_name}.apsimx"
import os
from dotenv import load_dotenv
load_dotenv()
bin_7939 = Path.home() / os.getenv('7939')
if __name__ == '__main__':
    from utils import test_mva

    logger = get_logger(log_dir=LOG_PATH, name=f'{script_name}')
    logger.info(
        "Simulation started\n ============================================================================\n\n ")
    logger.info(
        f"running no split fertilizer model under:\n {RESIDUE_RATES} residue  rates {QN_RATES} N rates \n and N timing = {TIMING_RATES} ")

    qp_data_model = create_experiment(base_file='nwrec_calibrated_1101_7493.apsimx',  start=1987, end=2020,
                                      out_path=apsimx_file_out_path, site='nwrec',  bin_path=bin_7939)

    qp_data_model.add_factor(specification=f"[fertilize in phases].Script.Amount = {QN_RATES}", factor_name='Nitrogen')

    # add residue removal levels, 0, 0.5, 0.75, 1 as fraction
    qp_data_model.add_factor(specification=f"[Tillage].Script.Fraction = {RESIDUE_RATES}", factor_name='Residue')
    # single_model.add_factor(specification=f"[Tillage].Script.Depth = 100, 150, 200, 250", factor_name='Depth')
    qp_data_model.edit_model(model_type='Models.Manager', model_name='Tillage', Depth=TILLAGE_DEPTH)
    # qp_data_model.edit_model(model_type='Models.Manager', model_name='fertilize in phases', FractionToApplyFirst=1,
    #                          FertiliserType='UreaN')

    try:
        # view the file before running
        # print('preview the file for any errors in GUI, simulation will resume after the this process has terminated')
        # qp_data_model.preview_simulation(watch=True)
        # print('Previewing the file for any errors in GUI has completed successfully simulation has resumed')
        # xc=  [5.22280546e+02, 1.99926221e+00, 2.59000701e+02, 1.53547113e+02,
        #   2.01114416e+01, 1.69878885e+01, 9.13472143e+01, 7.19520453e+02,
        #   4.79876155e-01, 3.27310034e-02]
        # edit_cultivar(qp_data_model, xc[:len(cmds)])
        # #qp_data_model.edit_model(model_type='Models.Manager', model_name='Sow using a variable rule', CultivarName='B_110')
        # qp_data_model.edit_model(model_type='Organic', model_name='Organic', FBiom=xc[-1], FInert = xc[-2])
        # if xc[-1]>1:
        #     raise ValueError('FBIOM IS ABNORMALLY high')
        qp_data_model.inspect_model('Models.Manager')

        # params = {
        #     # "[Leaf].Photosynthesis.RUE.FixedValue": 1.8984705340394,
        #     "[Phenology].GrainFilling.Target.FixedValue": 815,
        #     "[Grain].MaximumGrainsPerCob.FixedValue": 770,
        #     # "[Phenology].FloweringToGrainFilling.Target.FixedValue": 215,
        #     # "[Phenology].MaturityToHarvestRipe.Target.FixedValue": 100,
        #     # "[Maize].Grain.MaximumPotentialGrainSize.FixedValue": 0.867411373063701,
        #     # "[Grain].MaximumNConc.InitialPhase.InitialNconc.FixedValue": 0.05,
        #     # '[Maize].Root.SpecificRootLength.FixedValue': 135,
        #     # '[Maize].Root.RootFrontVelocity.PotentialRootFrontVelocity.PreFlowering.RootFrontVelocity.FixedValue': 22,
        #     # '[Rachis].DMDemands.Structural.DMDemandFunction.MaximumOrganWt.FixedValue': 36
        # }
        from apsim_validation_nwrec import params

        qp_data_model.set_params(path='.Simulations.1101.Field1.Sow using a variable rule',
                                 CultivarName='B_110')
        # qp_data_model.edit_model_by_path('.Simulations.Replacements.Maize.CultivarFolder.Generic.B_110',
        #                                  commands=params.keys()
        #                                  , sowed=True,
        #                                  values=params.values())
        qp_data_model.run('carbon')
    except FileNotFoundError as ie:
        logger.error(ie, exc_info=True)
        qp_data_model.preview_simulation()
        raise ie

    logger.info(f'succeeded running {file_name}')

    d = qp_data_model.get_simulated_output('carbon')

    test_mva(qp_data_model, table=d, title='Moving average SOC', response='min_ratio',
             expression="min_ratio =  SOC_0_15CM", time_col='year',
             grouping=("Residue", 'Nitrogen'), estimator='mean',
             errorbar=None, col='Residue', col_wrap=2, hue="Nitrogen")

    # single_model.preview_simulation()
    data = merge_tables(qp_data_model, ['carbon', 'yield'])
    store_this = data.copy()
    engine = create_engine(f'sqlite:///{datastore}')
    rows_saved = store_this.to_sql(table_name, con=engine, if_exists='replace', index=False)

    logger.info(f'{rows_saved} of data saved to table: {table_name}\n in: {datastore}\n')
    data_split = qp_data_model.get_simulated_output('carbon')
    data['cnr'] = data['SurfaceOrganicMatter_Carbon'] / data['SurfaceOrganicMatter_Nitrogen']
    data_split['cnr'] = data_split['SurfaceOrganicMatter_Carbon'] / data_split['SurfaceOrganicMatter_Nitrogen']

    test_mva(qp_data_model, table='carbon', response='microbial_carbon', time_col='year', hue="Nitrogen",
             grouping=("Residue", "Nitrogen"), estimator='mean',
             errorbar=None, col='Residue', col_wrap=2, )

    from apsimNGpy.core.config import apsim_version

    logger.info(f"Simulated using APSIM version: {apsim_version()}")
