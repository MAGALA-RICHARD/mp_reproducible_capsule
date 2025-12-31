from apsimNGpy.core.apsim import ApsimModel
from organise_nwrec_data import dir_data
from settings import path_to_MP_data, RESIDUE_RATES
from utils import edit_cultivar, cmds, pd, plot_reg_fit, create_experiment, N_RATES,  test_mva
from apsimNGpy.validation.evaluator import Validate
from pathlib import Path
import sys
apsimx = dir_data / 'nwrec_all_plots.apsimx'
model_b = ApsimModel(apsimx, out_path=dir_data / 'nw_b.apsimx')
TILLAGE_DEPTH =100
# xc = [505.97, 1.97, 294.04, 107.8, 20.24, 16.34, 0.53, 0.04]
xc = [523.94, 1.6, 210, 139.9, 28.26, 16.6, 111.68, 700.62, 0.65, 0.035]
edit_cultivar(model_b, x=xc[:len(cmds)])
model_b.edit_model('Organic', 'Organic', FInert = xc[-2], FBiom=xc[-1])
model_b.add_report_variable(
    variable_spec=['[Soil].Physical.BD[1] as Bd', '[Nutrient].TotalC[1]/1000 as apsim_soc_predicted'],
    report_name='Annual')
model_b.edit_model(model_name='Clock', model_type='Models.Clock', end_date='12-31-2016',
                       start_date='01-01-2000')

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
ev = Validate(obs_pred_carbon.Carbon, obs_pred_carbon.soc_from_apsim, ).evaluate_all(verbose=True)
from utils import create_experiment

plot_reg_fit(obs_pred_carbon[['Carbon']].values, obs_pred_carbon['soc_from_apsim'], data=obs_pred_carbon,
             xname='Carbon', fig_name='phase_1.png', color_by='year',
             yname='soc_from_apsim')
sys.exit()
if __name__ =='__main__':
    pass
    scenario_model = create_experiment(base_file=model_b.path, lonlat=None,  # lonlatTest,
                                       out_path=path_to_MP_data / 'nwerec_scenario_data.apsimx')
    # scenario_model.edit_model(model_type='Models.Soils.Organic', model_name='Organic', FInert=0.55, FBiom =[0.035,
    # 0.035]) add nitrogen levels 0, 165, 244, 326
    scenario_model.add_factor(specification=f"[MaizeNitrogenManager].Script.Amount = {N_RATES}", factor_name='Nitrogen')

    scenario_model.add_factor(specification=f"[Tillage].Script.Fraction = {RESIDUE_RATES}", factor_name='Residue')
    # some must be constants are set here to avoid confusion
    scenario_model.edit_model(model_type='Models.Manager', model_name='Tillage', Depth=TILLAGE_DEPTH)
    scenario_model.edit_model(model_type='Models.Manager', model_name='fertilize in phases', FractionToApplyFirst=1,
                              FertiliserType='UreaN')
    scenario_model.edit_model("Weather", model_name='Weather', weather_file=str(Path('mets/urbana_mp_128yrs.met').resolve()))
    scenario_model.run()
    d = scenario_model.get_simulated_output('Annual')
    dam = d
    test_mva(scenario_model, table=dam, title='Moving average SOC', response='SOC1',
             time_col ='an_Year',
             grouping=("Residue", 'Nitrogen'), estimator='mean',
             errorbar=None, col='Residue', col_wrap=2, hue="Nitrogen")