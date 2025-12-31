import numpy as np
import pandas as pd

from scipy.optimize import NonlinearConstraint, LinearConstraint
from evol_utils import evaluate_objectives, cache_objective, func
import settings
from organise_nwrec_data import dir_data, out_apsimx, datastore, yield_table, lonlat_sable
from apsimNGpy.core_utils.database_utils import read_db_table
from apsimNGpy.core.apsim import ApsimModel
from apsimNGpy.core.model_tools import ModelTools
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error
from apsimNGpy.validation.eval_methods import Validate
from functools import lru_cache
from apsimNGpy.exceptions import ApsimRuntimeError
from utils import edit_cultivar
from apsimNGpy.core.pythonet_config import Models
from evol_utils import base
if __name__ == "__main__":
    select_simulation= True
    apsim_model = ApsimModel(model=dir_data / 'x.apsimx', out_path=dir_data / 'scratch.apsimx')
    if select_simulation:
        sel_sim= '2031'
        for ch in apsim_model.simulations:
            if sel_sim != ch.Name:
               ModelTools.DELETE(ch)
    apsim_model.get_weather_from_web(lonlat=lonlat_sable, start=1986, end=2022, source='nasa')

    apsim_model.save()
    apsim_model.edit_model(model_name='Clock', model_type='Models.Clock', end_date='12-31-2016',
                           start_date='01-01-2011', exclude = None)
    # apsim_model.preview_simulation(watch=True)
    apsim_model.edit_model(model_name='fertilize in phases', model_type='Models.Manager', Amount=245,
                           exclude='Simulation')
    apsim_model.edit_model(model_name='Tillage', model_type='Models.Manager', Depth=350, TillageDate='1-oct',
                           exclude='Simulation')
    print(apsim_model.simulations.__len__())

    # for i in {0, '1102'}:
    #     if str(i) in [s.Name for s in apsim_model.simulations]:
    #         print(f"deleting {i} simulation")
    #         apsim_model.remove_model(model_type='Models.Core.Simulation', model_name=str(i))
    apsim_model.edit_model(model_type='Models.Manager', model_name='Sow using a variable rule',
                           CultivarName='Pioneer_33M54',
                           RowSpacing=750)

    apsim_model.edit_model(model_name='Clock', model_type='Models.Clock', end_date='12-31-2011',
                           start_date='01-01-2000')
    if '1101' in [s.Name for s in apsim_model.simulations]:
        apsim_model.edit_model(model_type='Physical', model_name='Physical', SAT=[0.48], simulations='1101')
    apsim_model.edit_model(model_name='Tillage', model_type='Models.Manager', Depth=350, TillageDate='1-oct',
                           exclude='Simulation')
    apsim_model.add_report_variable(variable_spec=['[Simulation].Name as Plotid as pid'], report_name='carbon')

    dYield = read_db_table(datastore, yield_table)
    dYield['year'] = dYield['year'].astype(int)
    dYield['plotid'] = dYield['plotid'].astype(int).astype(str)
    dYield.set_index(['year', 'plotid'], inplace=True)
    dYield['corn_grain_yield_Mg'] = dYield['corn_grain_yield_kg'] / 1000
    if '1101' in [s.Name for s in apsim_model.simulations]:
        apsim_model.edit_model(model_type='Physical', model_name='Physical', SAT=[0.484], simulations='1101')

    apsim_model = ApsimModel(model=dir_data / 'scratch.apsimx', out_path=dir_data / f'scratch2.apsimx')
    apsim_model.edit_model('Models.Report', model_name='carbon', variable_spec=['[Simulation].Name as SIM_ID'], report_name='carbon')
    apsim_model.edit_model(model_name='Sow using a variable rule', model_type='Models.Manager', EndDate='15-may',
                           StartDate='1-may',
                           exclude='Simulation')
    apsim_model.edit_model(model_name='Clock', model_type='Models.Clock', end_date='12-31-2016',
                           start_date='01-01-2011')
    # apsim_model.preview_simulation(watch=True)
    apsim_model.edit_model(model_name='fertilize in phases', model_type='Models.Manager', Amount=245,
                           exclude='Simulation')
    apsim_model.edit_model(model_name='Tillage', model_type='Models.Manager', Depth=350, TillageDate='1-oct',
                           exclude='Simulation')
    if '1101' in [s.Name for s in apsim_model.simulations]:
        apsim_model.edit_model(model_type='Physical', model_name='Physical', SAT=[0.484], simulations='1101')
    print('Metrics before optimization')
    base['model'] = apsim_model
    xf = func(x=None)
    print('===================================================')
    obj = cache_objective(evaluate_objectives,round_digits=2)
    from scipy.optimize import differential_evolution


    nlc = NonlinearConstraint(obj, 0, 0.17)
    #lc = LinearConstraint([1, 2], lb = -100, ub=-25)

    # Objective must return a scalar (to minimize), just like with `minimize`

    bounds = [(500, 700), (1.6, 2), (200, 260), (80, 160), (20, 30), (15, 20), (90,120), (680,720),(0.4, 0.55), (0.03, 0.045)]

    result = differential_evolution(
        func=obj,
        bounds=bounds,
        strategy="best1bin",  # good default
        maxiter=15,  # ~generations
        popsize=10,  # population size multiplier (>= 5 is common)
        tol=1e-6,
        mutation=(0.5, 1.0),  # differential weight (can be tuple for dithering)
        recombination=0.7,  # crossover prob
        seed=42,  # reproducibility
        polish=True,  # local search at the end (uses L-BFGS-B)
        constraints=nlc,
        workers=1
    )

    print("Success:", result.success)
    print("Message:", result.message)
    print("Best x:", result.x)
    print("Best f(x):", result.fun)

    # ________________________________________
    # mixed problem
    from apsimNGpy.optimizer.problems.smp import MixedProblem

    mp = MixedProblem(model='Maize', trainer_dataset=obs, pred_col='Yield', metric='RRMSE',
                      index='year', trainer_col='observed')
