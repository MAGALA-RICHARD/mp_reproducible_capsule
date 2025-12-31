from apsimNGpy.core.apsim import ApsimModel
from apsimNGpy.core.experimentmanager import ExperimentManager

if __name__ == '__main__':
    model = ApsimModel('Maize')
    model.inspect_model('Models.Manager', fullpath=False)
    model.inspect_model_parameters_by_path('.Simulations.Simulation.Field.Sow using a variable rule',
                                           parameters=['Population', 'MinRain'])
    model.run()
    res_before = model.results
    print(f"Mean before editing; {res_before.Yield.mean()}")
    # _______________edit the model to see the changes in the parameter
    model.edit_model(model_type='Models.Manager', model_name='Fertilise at sowing', Amount=180)
    model.edit_model(model_type='Models.Manager', Population=10, model_name='Sow using a variable rule')
    model.run()
    res_after = model.results
    print(f"Mean after editing; {res_after.Yield.mean()}")
    assert res_before.Yield.mean() != res_after.Yield.mean(), 'Oops editing was unsuccessful'

    from apsimNGpy.core.mult_cores import MultiCoreManager
    from apsimNGpy.core.runner import run_from_dir
    from pathlib import Path

    data_folder = Path(".").parent.parent / 'demo'
    data_folder.mkdir(parents=True, exist_ok=True)
    import shutil

    jobs = (shutil.copy(model.path, f"{data_folder}/{i}.apsimx") for i in range(100))
    taskmanager = MultiCoreManager(db_path="demo.db", agg_func='mean')
    taskmanager.run_all_jobs(jobs=jobs, n_cores=10, threads=False)
