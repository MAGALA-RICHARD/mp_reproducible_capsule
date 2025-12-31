import sys

import matplotlib.pyplot as plt
import pandas as pd

import settings
from organise_nwrec_data import dir_data, out_apsimx, datastore, yield_table, lonlat_sable
from apsimNGpy.core_utils.database_utils import read_db_table
from apsimNGpy.core.apsim import ApsimModel
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error
from apsimNGpy.validation.eval_methods import Validate
from utils import plot_reg_fit, open_file
from utils import edit_cultivar, xc, cmds

xcp = [505.97, 1.97, 294.04, 107.8, 20.24, 16.34, 0.53, 0.04]
xc = 521.23, 1.8, 258.95, 130.92, 29.98, 17.94, 101.5, 712.84, 0.45, 0.04
setting = settings
if __name__ == "__main__":
    apsim_model = ApsimModel(model=dir_data / 'nwrec1.apsimx', out_path=dir_data / 'scratch3.apsimx')
    apsim_model.get_weather_from_web(lonlat=lonlat_sable, start=1986, end=2022, source='nasa',
                                     filename=dir_data / 'scratch4.met')
    apsim_model.edit_model(model_name='Clock', model_type='Models.Clock', end_date='12-31-2011',
                           start_date='01-01-2000')
    if '1101' in [s.Name for s in apsim_model.simulations]:
        apsim_model.edit_model(model_type='Physical', model_name='Physical', SAT=[0.48], simulations='1101')
    apsim_model.edit_model(model_name='Tillage', model_type='Models.Manager', Depth=350, TillageDate='1-oct',
                           exclude='Simulation')

    apsim_model.edit_model(model_name='Sow using a variable rule', model_type='Models.Manager', EndDate='15-may',
                           StartDate='1-may',
                           exclude='Simulation')
    apsim_model.run()
    carbon = apsim_model.get_simulated_output('carbon')
    cb = carbon[carbon['year'] == 2011].copy()
    for sim in cb.Plotid.unique():
        pda = cb[cb['Plotid'] == sim]
        N = pda['SurfaceOrganicMatter_Nitrogen'].iloc[0]
        cnr = pda['SurfaceOrganicMatter_Carbon'].iloc[0] / N
        print(cnr)
        apsim_model.edit_model(model_type='SurfaceOrganicMatter', model_name='SurfaceOrganicMatter', InitialCNR=cnr,
                               simulations=sim, exclude='Simulations')
    apsim_model.save(dir_data / 'x.apsimx')
    apsim_model.inspect_model_parameters(model_type='SurfaceOrganicMatter', model_name='SurfaceOrganicMatter')
    apsim_model.edit_model(model_name='Clock', model_type='Models.Clock', end_date='12-31-2016',
                           start_date='01-01-2011')
    # edit_cultivar(apsim_model, [518.41069723, 1.89807246, 293.74403161, 159.27912538, 20.00361265, 19.93488138])
    edit_cultivar(apsim_model, xc[:len(cmds)])
    # apsim_model.preview_simulation(watch=True)
    apsim_model.edit_model(model_name='fertilize in phases', model_type='Models.Manager', Amount=240,
                           exclude='Simulation')
    apsim_model.edit_model(model_name='Tillage', model_type='Models.Manager', Depth=350, TillageDate='1-oct',
                           exclude='Simulation')
    apsim_model.preview_simulation(watch=True)
    print(apsim_model.simulations.__len__())

    for i in {0, '11022', '11021'}:
        if str(i) in [s.Name for s in apsim_model.simulations]:
            print(f"deleting {i} simulation")
    apsim_model.remove_model(model_type='Models.Core.Simulation', model_name=str(i))
    # apsim_model.edit_model(model_type='Models.Manager', model_name='Sow using a variable rule',
    #                        CultivarName='Pioneer_33M54',
    #                        RowSpacing=760)

    dYield = read_db_table(datastore, yield_table)
    dYield['year'] = dYield['year'].astype(int)
    dYield['plotid'] = dYield['plotid'].astype(int).astype(str)
    dYield.set_index(['year', 'plotid'], inplace=True)
    dYield['corn_grain_yield_Mg'] = dYield['corn_grain_yield_kg'] / 1000

    apsimx = apsim_model  # func([546 ,1.96])
    apsimx.run(report_name='yield', verbose=False)
    df = apsimx.results
    df.eval('apsim=grainwt/1000', inplace=True)
    df['Plotid'] = df['Plotid'].astype(str)
    df.eval('plotid=Plotid', inplace=True)

    df.set_index(['year', 'plotid'], inplace=True)
    from apsimNGpy.validation.eval_methods import Validate

    data = dYield.join(df, how='inner')
    rmse = root_mean_squared_error(data['corn_grain_yield_Mg'], data['apsim'])
    val = Validate(data['corn_grain_yield_Mg'], data['apsim'])
    metrics = {k: float(v) for k, v in val.evaluate_all().items()}
    from pprint import pprint

    print('\n Yield evaluation metric part 1')
    pprint(metrics)
    r_rmse = rmse / data['corn_grain_yield_Mg'].mean()
    mae = mean_absolute_error(data['corn_grain_yield_Mg'], data['apsim'])
    dff = data.reset_index()
    dff.eval('year= year.astype("int")', engine='python', inplace=True)
    dff['year'] = pd.Categorical(dff['year'].astype('int'), ordered=True)
    plot_reg_fit(dff[['apsim']].values / 0.85, dff.corn_grain_yield_Mg.values, data=dff, xname='apsim',
                 yname='corn_grain_yield_Mg')
    print(
        f'rmse: {rmse}, rrmse: {r_rmse},mean apsim:{data['apsim'].mean()}, mean of observed: {data['corn_grain_yield_Mg'].mean()} ')

    # constitute data from a rigorous manual soil calibration
    apsimx = dir_data / 'nwrec_all_plots.apsimx'
    model_b = ApsimModel(apsimx, out_path=dir_data / 'nw_b.apsimx')

    # xc = [505.97, 1.97, 294.04, 107.8, 20.24, 16.34, 0.53, 0.04]
    xc = [5.22280546e+02, 1.99926221e+00, 2.59000701e+02, 1.53547113e+02,
          2.01114416e+01, 1.69878885e+01, 9.13472143e+01, 7.19520453e+02,
          4.79876155e-01, 3.27310034e-02]
    xc = [5.22579649e+02, 1.97516019e+00, 2.58203641e+02, 1.35035475e+02,
          2.83058903e+01, 1.91092362e+01, 9.87984215e+01, 6.88004710e+02,
          5.11478099e-01, 3.38229670e-02]
    edit_cultivar(model_b, xc[:len(cmds)])
    model_b.edit_model('Organic', 'Organic', FBiom=xc[-1], FInert=xc[-2])
    model_b.add_report_variable(
        variable_spec=['[Soil].Physical.BD[1] as Bd', '[Nutrient].TotalC[1]/1000 as apsim_soc_predicted'],
        report_name='Annual')

    model_b.run()

    apsim = model_b.get_simulated_output('Annual')

    apsim['soc_sim'] = apsim['apsim_soc_predicted']
    apsim.eval('year=an_Year', inplace=True)
    apsim['plotid'] = apsim['plotid'].astype(int)
    yd2 = pd.read_csv(dir_data / 'nwec_yield_2.csv')
    cb = pd.read_csv(dir_data / 'nwrec_carbon_2.csv')
    obs_pred_carbon = apsim.merge(cb, on=['plotid', 'year'])
    obs_pred_carbon.eval('soc_from_apsim = soc_sim/(Bd * depth)', inplace=True)
    print('\npart two evaluation soil carbon evaluation metrics')
    ev = Validate(obs_pred_carbon.Carbon, obs_pred_carbon.soc_from_apsim, )
    from utils import create_experiment

    plot_reg_fit(obs_pred_carbon[['Carbon']].values, obs_pred_carbon['soc_from_apsim'], data=obs_pred_carbon,
                 xname='Carbon', fig_name='phase_1.png', color_by='year',
                 yname='soc_from_apsim')

    ev.evaluate_all(verbose=True)
    # Evaluate yield part two
    dy = model_b.get_simulated_output('MaizeR').drop('mYield', axis=1)
    dy['grain_apsim'] = dy['Maize_Yield']
    dy.eval('year=Year', inplace=True)
    dy.eval('plotid= plotid.astype("int")', engine='python', inplace=True)
    yd2.drop('Maize_Yield', axis=1, inplace=True, errors='ignore')
    all_yield = yd2.merge(dy, on=['plotid', 'year'], how='inner')
    print('\npart two evaluation yield evaluation metrics')
    va_yield = Validate(all_yield.obs / 1000, (all_yield.grain_apsim) / 1000).evaluate_all(verbose=True)
    model_b.preview_simulation()
