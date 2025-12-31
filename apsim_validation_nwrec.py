import os

import dotenv
import pandas as pd
from apsimNGpy.core.config import apsim_bin_context, get_apsim_bin_path, set_apsim_bin_path, configuration
from loguru import logger
from pathlib import Path

dotenv.load_dotenv()
apsim_7939 = Path.home() / os.getenv('7939')
with apsim_bin_context(apsim_7939):
    from apsimNGpy.optimizer.problems.back_end import final_eval
    from apsimNGpy.core.apsim import ApsimModel
    from pandas import read_csv
from pathlib import Path
import os
from utils import carbon_to_Mg, load_manifest, pbias, RESULTS
from loguru import logger
cfg = load_manifest()
paths = cfg['paths']
apsimx_dir = paths['apsimx_base_dir']
dotenv.load_dotenv()
higher_version = os.getenv('7939')
home_path = Path.home() / higher_version
# because I open the base path in one the higher version, it has to be this version forever


# __________________________________
# define parameters
# ___________________________________

# nwrec carbon
dir_data = Path('./yield_DATA')
cb = pd.read_csv(dir_data / 'nwrec_carbon_2.csv')
cb['observed_carbon'] *= 10
cb['Plotid'] = cb['plotid']
nwrec_soc = carbon_to_Mg(cb, conc_col='observed_carbon', bd_col='BD', depth_col='depth')

params = {
    "[Leaf].Photosynthesis.RUE.FixedValue": 2,
    "[Phenology].Juvenile.Target.FixedValue": 211,
    "[Phenology].Photosensitive.Target.XYPairs.X": '0, 12.5, 24',
    "[Phenology].Photosensitive.Target.XYPairs.Y": '0, 0, 0',
    "[Phenology].FlagLeafToFlowering.Target.FixedValue": 1,
    "[Phenology].FloweringToGrainFilling.Target.FixedValue": 170,
    "[Phenology].GrainFilling.Target.FixedValue": 815,
    "[Phenology].Maturing.Target.FixedValue": 1,
    "[Phenology].MaturityToHarvestRipe.Target.FixedValue": 1,
    "[Rachis].DMDemands.Structural.DMDemandFunction.MaximumOrganWt.FixedValue": 36,
    "[Grain].MaximumGrainsPerCob.FixedValue": 770,
}

if __name__ == "__main__":
    stem = Path(__file__).stem
    val_res = RESULTS / stem
    val_res.mkdir(exist_ok=True)
    try:
        dir_data = Path('yield_DATA')
        base_path = dir_data / 'base_opt.apsimx'
        obs = read_csv('tmp.csv')
        nwrec = obs.reset_index()
        nwrec.to_csv(dir_data / 'measured_yield_data_nwrec.csv', index=False)
        cmds_value_pair = {'[Phenology].GrainFilling.Target.FixedValue': 700,

                           '[Leaf].Photosynthesis.RUE.FixedValue': 2}

        obs['yr'] = obs['year'].astype('float')
        obs = obs[obs['yr'] > 2011]
        # base = ApsimModel(base_path)
        # base.edit_model_by_path(
        #
        # )

        choices = ['B_110', 'B_120', 'P1197', 'A_103', 'Laila', 'A_112']
        from apsimNGpy.core.config import configuration

        nwrec_apsimx = Path(apsimx_dir) / cfg['apsimx']['nwrec']
        with ApsimModel(nwrec_apsimx) as model:
            model.edit_model('Manager', 'Sow using a variable rule', CultivarName='A_112')
            model.edit_model_by_path('.Simulations.Replacements.Maize.CultivarFolder.Generic.A_112',
                                     commands=params.keys()
                                     , sowed=True,
                                     values=params.values())
            model.run(verbose=True)
            predicted = model.get_simulated_output('gyield')
            predicted = predicted[predicted['year'] > 2012]
            logger.info('reporting metrics for corn grain yield ')
            y_ev = final_eval(obs, predicted, pred_col='grainwt', index=['year', 'Plotid'],
                              obs_col='observed',
                              )
            ydata = y_ev['data']
            pb = pbias(observed=ydata['observed'], simulated=ydata['grainwt'])
            y_ev['metrics']['Pbias'] = pb
            yield_met = pd.DataFrame([y_ev.get('metrics')])
            yield_met = yield_met.assign(variable='corn_grain_yield')
            reps = model.inspect_model('Models.Report')
            # model.preview_simulation(watch=True)
            logger.info('reporting metrics for soil carbon')
            carbon_ev = model.evaluate_simulated_output(ref_data=nwrec_soc, table="Annual", target_col='SOC_0_15CM',
                                                        index_col=['year', 'Plotid'],
                                                        ref_data_col='soc_Mg')
            data = carbon_ev['data']
            pb = pbias(observed=data['soc_Mg'], simulated=data['SOC_0_15CM'])
            mets = carbon_ev['metrics']
            mets['Pbias'] = pb
            met_df = pd.DataFrame([mets])
            carbon_metrics = met_df.assign(variable='soil carbon')
            all_v_metrics = pd.concat([carbon_metrics, yield_met])
            csv_name = os.path.realpath(val_res / f'{stem}.csv')
            logger.info(f'results saved to: {csv_name}')
            all_v_metrics.to_csv(csv_name, index=False)
            p_df =  pd.DataFrame([params])
            p_df.to_csv(val_res/f'{stem}_params.csv', index=False)

            carbon_ev['metrics']['Pbias'] = pb
            nwerec_metrics = pd.DataFrame([carbon_ev['metrics'], y_ev['metrics']])[
                ['RMSE', 'RRMSE', 'MAE', 'ME', 'WIA', 'Pbias']]
            nwerec_metrics.to_csv(Path(cfg['paths']['Results']) / 'nwerec_metrics.csv', index=False)


    finally:
        pass
