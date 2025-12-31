import numpy as np

from organise_nwrec_data import dir_data, out_apsimx, datastore, yield_table, lonlat_sable
from apsimNGpy.core_utils.database_utils import read_db_table
from apsimNGpy.core.apsim import ApsimModel
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error
from apsimNGpy.validation.eval_methods import Validate
from functools import lru_cache
import sys
from pathlib import Path

module_name = Path(__file__).stem
if __name__ == "__main__":
    apsim_model = ApsimModel(model=out_apsimx, out_path=dir_data / f'{module_name}_scratch.apsimx')
    apsim_model.get_weather_from_web(lonlat=lonlat_sable, start=1986, end=2022, source='nasa')
    apsim_model.edit_model(model_name='Clock', model_type='Models.Clock', end_date='12-31-2016',
                           start_date='01-01-2011')
    # apsim_model.preview_simulation(watch=True)
    apsim_model.edit_model(model_name='fertilize in phases', model_type='Models.Manager', Amount=245,
                           exclude='Simulation')
    apsim_model.edit_model(model_name='Tillage', model_type='Models.Manager', Depth=350, TillageDate='1-oct',
                           exclude='Simulation')
    print(apsim_model.simulations.__len__())

    for i in {0, '1102'}:
        if str(i) in [s.Name for s in apsim_model.simulations]:
            print(f"deleting {i} simulation")
            apsim_model.remove_model(model_type='Models.Core.Simulation', model_name=str(i))
    apsim_model.edit_model(model_type='Models.Manager', model_name='Sow using a variable rule',
                           CultivarName='Pioneer_33M54',
                           RowSpacing=750)
    dYield = read_db_table(datastore, yield_table)
    dYield['year'] = dYield['year'].astype(int)
    dYield['plotid'] = dYield['plotid'].astype(int).astype(str)
    dYield.set_index(['year', 'plotid'], inplace=True)
    dYield['corn_grain_yield_Mg'] = dYield['corn_grain_yield_kg'] / 1000
    if '1101' in [s.Name for s in apsim_model.simulations]:
        apsim_model.edit_model(model_type='Physical', model_name='Physical', SAT=[0.504], simulations='1101')

    @lru_cache(maxsize=100)
    def func(x):

        apsim_model = ApsimModel(model=out_apsimx, out_path=dir_data / f'scratch.apsimx')
        apsim_model.get_weather_from_web(lonlat=lonlat_sable, start=1986, end=2022, source='nasa')
        apsim_model.edit_model(model_name='Clock', model_type='Models.Clock', end_date='12-31-2016',
                               start_date='01-01-2011')
        # apsim_model.preview_simulation(watch=True)
        apsim_model.edit_model(model_name='fertilize in phases', model_type='Models.Manager', Amount=245,
                               exclude='Simulation')
        apsim_model.edit_model(model_name='Tillage', model_type='Models.Manager', Depth=350, TillageDate='1-oct',
                               exclude='Simulation')
        if '1101' in [s.Name for s in apsim_model.simulations]:
            apsim_model.edit_model(model_type='Physical', model_name='Physical', SAT=[0.504], simulations='1101')
        apsim_model.edit_model(
            model_type='Cultivar',
            simulations=None,
            commands=['[Phenology].GrainFilling.Target.FixedValue', '[Leaf].Photosynthesis.RUE.FixedValue',
                      '[Phenology].Juvenile.Target.FixedValue', ],
            values=[x[0], x[1], x[2]],
            new_cultivar_name='pioneer_e',
            model_name='B_100',
            cultivar_manager='Sow using a variable rule')
        model = apsim_model
        # This function runs APSIM and compares the predicted maize yield results with observed data.
        predicted = model.run(report_name="yield", verbose=False).results
        predicted.eval('apsim = grainwt/1000', inplace=True)
        predicted['Plotid'] = predicted['Plotid'].astype(str)
        predicted.eval('plotid=Plotid', inplace=True)

        predicted.set_index(['year', 'plotid'], inplace=True)
        rdata = dYield.join(predicted, how='inner')
        val = Validate(rdata['corn_grain_yield_Mg'], rdata['apsim'])
        metrics = {k: float(v) for k, v in val.evaluate_all().items()}
        # Use root mean square error or another metric.
        min_func = ((metrics['ccc'] * -100) + (metrics['r2'] * -100)) / 2
        if min_func > 25:
            print(x)
        return min_func


    def evaluate_objectives(x):
        x = np.round(x, decimals=2)
        return func(x)


    from scipy.optimize import differential_evolution

    # Objective must return a scalar (to minimize), just like with `minimize`
    # bounds correspond to each decision variable (same as your Nelder–Mead call)
    bounds = [(500, 800), (1.5, 2.5), (200, 300)]
