import os

import pandas as pd
from apsimNGpy.core.config import apsim_bin_context, get_apsim_bin_path, set_apsim_bin_path
from pathlib import Path
import os
import dotenv
from utils import carbon_to_Mg, load_manifest, pbias

cfg = load_manifest()
paths= cfg['paths']
apsimx_dir = paths['apsimx_base_dir']
dotenv.load_dotenv()
higher_version = os.getenv('higher_version')
home_path = Path.home() / higher_version
# because I open the base path in one the higher version, it has to be this version forever
with apsim_bin_context(home_path):
    from apsimNGpy.core.apsim import ApsimModel
    from apsimNGpy.optimizer.minimize.single_mixed import MixedVariableOptimizer

# __________________________________
# define parameters
# ___________________________________

# nwrec carbon
dir_data = Path('./yield_DATA')
cb = pd.read_csv(dir_data / 'nwrec_carbon_2.csv')
cb['observed_carbon'] *= 10
cb['Plotid'] = cb['plotid']
nwrec_soc = carbon_to_Mg(cb, conc_col='observed_carbon', bd_col='BD', depth_col='depth')
if __name__ == "__main__":
    from apsimNGpy.optimizer.problems.variables import ChoiceVar
    from apsim_maize_cultivar import choice_cultivars

    current_bin = get_apsim_bin_path()
    set_apsim_bin_path(home_path)
    with apsim_bin_context(home_path):
        from apsimNGpy.optimizer.problems.smp import MixedProblem
        from apsimNGpy.optimizer.problems.back_end import final_eval
        from apsimNGpy.optimizer.variables import QrandintVar, UniformVar
    from pathlib import Path
    from pandas import read_csv

    dir_data = Path('yield_DATA')
    base_path = dir_data / 'base_opt.apsimx'

    # model = ApsimModel(base_path, out_path='opt.apsimx')
    # model.get_weather_from_web(lonlat=lonlat_sable, start=2000, end=2018, source='daymet')

    # dYield = read_db_table(datastore, yield_table)
    # dYield['year'] = dYield['year'].astype(int)
    # dYield['Plotid'] = dYield['plotid'].astype(int).astype(str)
    # dYield.eval('observed= grain_yield_Mg * 1000', inplace=True)
    # dYield.to_csv( 'tmp.csv', index=False)
    obs = read_csv('tmp.csv')
    nwrec = obs.reset_index()
    nwrec.to_csv(dir_data / 'measured_yield_data_nwrec.csv', index=False)
    cmds_value_pair = {'[Phenology].GrainFilling.Target.FixedValue': 700,

                       '[Leaf].Photosynthesis.RUE.FixedValue': 2}
    cultivar_params = {
        "path": ".Simulations.Replacements.Maize.CultivarFolder.Generic.B_110",
        "vtype": [QrandintVar(400, 900, q=10),
                  UniformVar(1.5, 2.2)],
        "start_value": [i for i in cmds_value_pair.values()],
        "candidate_param": [i for i in cmds_value_pair.keys()],
        "other_params": {"sowed": True, },
        "cultivar": True,  # Signals to apsimNGpy to treat it as a cultivar parameter
    }
    soil_param = {
        "path": ".Simulations.Replacements.Muscatune:244911.Organic",
        'vtype': [UniformVar(200, 550)],
        'start_value': [250],
        "candidate_param": ['FOM']

    }
    nitrate = dict(path='.Simulations.Replacements.Muscatune:244911.SoilWater',
                   vtype=[UniformVar(0, 3)],
                   candidate_param=['SWCON'],
                   start_value=[1])
    nh4 = dict(path='.Simulations.Replacements.Muscatune:244911.NH4',
               vtype=[UniformVar(0, 1)],
               candidate_param=['InitialValues'],
               start_value=[1])

    tillage = dict(path='.Simulations.Replacements.Tillage',
                   vtype=[UniformVar(0.2, 1), QrandintVar(100, 350, 50)],
                   start_value=[1, 250, ],
                   candidate_param=['Fraction', 'Depth'])
    obs['yr'] = obs['year'].astype('float')
    obs = obs[obs['yr'] > 2011]
    # base = ApsimModel(base_path)
    # base.edit_model_by_path(
    #
    # )
    mp = MixedProblem(model=base_path, trainer_dataset=obs, table='gyield', index=['year', 'Plotid'],
                      pred_col='grainwt',
                      trainer_col='observed',
                      metric='rrmse')
    choices = ['B_110', 'B_120', 'P1197', 'A_103', 'Laila', 'A_112']
    from apsimNGpy.core.config import configuration

    with apsim_bin_context(apsim_bin_path=home_path):
        print(configuration.bin_path)
        from apsimNGpy.core.pythonet_config import configuration

        print(configuration.bin_path)
        from apsimNGpy.core.apsim import ApsimModel
        nwrec_apsimx = Path(apsimx_dir)/ cfg['apsimx']['nwrec']
        with ApsimModel(nwrec_apsimx) as model:

            model.edit_model('Manager', 'Sow using a variable rule', CultivarName='A_112')
            # model.edit_model('weather', model_name='Weather',
            #                  met_file=r'D:\My_BOX\Box\PhD thesis\Objective two\morrow plots 20250821\mets\iem_168years.met')
            params = {
                "[Leaf].Photosynthesis.RUE.FixedValue": 1.719184705340394,
                # "[Phenology].GrainFilling.Target.FixedValue": 700,
                # "[Grain].MaximumGrainsPerCob.FixedValue": 700,
                # "[Phenology].FloweringToGrainFilling.Target.FixedValue": 215,
                # "[Phenology].MaturityToHarvestRipe.Target.FixedValue": 100,
                # "[Maize].Grain.MaximumPotentialGrainSize.FixedValue": 0.867411373063701,
                # "[Grain].MaximumNConc.InitialPhase.InitialNconc.FixedValue": 0.0026155907670621074,
                # '[Maize].Root.SpecificRootLength.FixedValue': 135,
                # '[Maize].Root.RootFrontVelocity.PotentialRootFrontVelocity.PreFlowering.RootFrontVelocity.FixedValue': 22,
            }
            params = {
                "[Leaf].Photosynthesis.RUE.FixedValue": 2,
                "[Phenology].Juvenile.Target.FixedValue": 211,
                "[Phenology].Photosensitive.Target.XYPairs.X": '0, 12.5, 24',
                "[Phenology].Photosensitive.Target.XYPairs.Y": '0, 0, 2',
                "[Phenology].FlagLeafToFlowering.Target.FixedValue": 1,
                "[Phenology].FloweringToGrainFilling.Target.FixedValue": 170,
                "[Phenology].GrainFilling.Target.FixedValue": 815,
                "[Phenology].Maturing.Target.FixedValue": 1,
                "[Phenology].MaturityToHarvestRipe.Target.FixedValue": 1,
                "[Rachis].DMDemands.Structural.DMDemandFunction.MaximumOrganWt.FixedValue": 36,
                "[Grain].MaximumGrainsPerCob.FixedValue": 770,
            }

            model.edit_model_by_path('.Simulations.Replacements.Maize.CultivarFolder.Generic.A_112',
                                     commands=params.keys()
                                     , sowed=True,
                                     values=params.values())
            # model.preview_simulation()
            # import time
            # time.sleep(2000)

            model.run()
            predicted = model.get_simulated_output('gyield')
            predicted = predicted[predicted['year'] > 2012]

            y_ev = final_eval(obs, predicted, pred_col='grainwt', index=['year', 'Plotid'],
                              obs_col='observed',
                              )
            data= y_ev['data']
            pb =pbias(observed=data['observed'], predicted= data['grainwt'])
            reps = model.inspect_model('Models.Report')
           # model.preview_simulation(watch=True)
            print(reps)
            carbon_ev = model.evaluate_simulated_output(ref_data=nwrec_soc, table="Annual", target_col='SOC_0_15CM',
                                                        index_col=['year', 'Plotid'],
                                                        ref_data_col='soc_Mg')
            print(carbon_ev)

    # mp.submit_factor(path='.Simulations.Replacements.Sow using a variable rule',
    #                  vtype=[ChoiceVar([*choices]), ],
    #                  start_value=['P1197'], candidate_param=['CultivarName'])
    # mp.submit_factor(**cultivar_params)
    # mp.submit_factor(**nitrate)
    # mp.submit_factor(**nh4)
    # mp.submit_factor(**tillage)
    calibration_params = {
        "[Leaf].Photosynthesis.RUE.FixedValue": (1.4, 2.4),
        "[Phenology].GrainFilling.Target.FixedValue": (600, 900),
        "[Grain].MaximumGrainsPerCob.FixedValue": (600, 900),
        "[Phenology].FloweringToGrainFilling.Target.FixedValue": (150, 230),
        '[Phenology].MaturityToHarvestRipe.Target.FixedValue': (40, 60),
        '[Maize].Grain.MaximumPotentialGrainSize.FixedValue': (0.3, 0.5),
        # '[Maize].Root.SpecificRootLength.FixedValue': 135,
        # '[Maize].Root.RootFrontVelocity.PotentialRootFrontVelocity.PreFlowering.RootFrontVelocity.FixedValue': 22,
        '[Grain].MaximumNConc.InitialPhase.InitialNconc.FixedValue': 0.03
    }
    # mp.submit_factor(path='.Simulations.Replacements.Maize.CultivarFolder.Generic.A_112',
    #                  vtype=[
    #                      UniformVar(0, 2.2),
    #
    #                  ],
    #                  start_value=[[2]],
    #                  candidate_param=[i for i in params.keys()],
    #                  cultivar=True,
    #                  other_params={'sowed': True}
    #                  )
    # mn = MixedVariableOptimizer(problem=mp)
    # # try:
    # #     de = mn.minimize_with_de(use_threads=True, workers=12, popsize=100, maxiter=1800)
    # #     print(de)
    # # except ValueError as e:
    # #     pass
    #
    # # try:
    # #     nelda = mn.minimize_with_local(method="Nelder-Mead", options={'maxiter': 2400,})
    # #     print(nelda)
    # # except ValueError:
    # #    pass
    # # #
    # # # # with ApsimModel(base_path) as apsimx:
    # # #
    # # #     apsimx.run(report_name='gyield', verbose=False, )
    # # #     apsimx.preview_simulation()
    # # #     df = apsimx.results

    # # plots
    # import matplotlib.pyplot as plt
    # import os
    # from utils import plot_reg_fit
    #
    # plt.figure(figsize=(8, 6))
    # df = ev['data']
    #
    # df.eval('ayield = grainwt/1000', inplace=True)
    # df.eval('oyield =observed/1000', inplace=True)
    # # observed → scatter points
    # plot_reg_fit(df['ayield'], df['oyield'], color_by='Plotid', data=df, xname='ayield', yname='oyield')
    # dfc = df.copy()
    # dfc = dfc.drop(['ayield', 'oyield'], axis=1)
    # dfm = dfc.melt(id_vars=['year', 'Plotid'])
    # model.relplot(table=dfm, x='year', y='value', hue='variable', kind='line', errorbar=None)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("figures.png")
    # if hasattr(os, 'startfile'):
    #     os.startfile("figures.png")
    # plt.close()
