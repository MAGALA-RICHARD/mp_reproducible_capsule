from utils import base, create_experiment
import seaborn as sns
from matplotlib import pyplot as plt
if __name__ == '__main__':
    from apsimNGpy.optimizer.moo import MultiObjectiveProblem, compute_hyper_volume, NSGA2
    from pymoo.optimize import minimize

    model = create_experiment(base_file='split')
    runner = base(method='single')
    problem = MultiObjectiveProblem(runner, objectives=[carbon_obj, yield_obj])

    problem.add_control(
        path='.Simulations.changeFert_verity.Field.single_N_at_sowing',
        Amount='?', bounds=[50, 300], v_type='float')

    problem.add_control(
        path='.Simulations.changeFert_verity.Field.Postharvestillagmaize',
        Fraction='?', bounds=[0.25, 1.0], v_type='float')

    algorithm = NSGA2(pop_size=20)

    result = minimize(
        problem.get_problem(),
        algorithm,
        ('n_gen', 20),
        seed=1,
        verbose=True
    )
    hv = compute_hyper_volume(result.F, normalize=True)
    print("Hyper volume:", hv)

    plt.scatter(result.F[:, 0] * -1, result.F[:, 1] / 1000 * -1)
    plt.tight_layout()
    plt.xlabel("Soil organic carbon (Mg/ha)")
    plt.ylabel("Corn grain yield")
    plt.title("Pareto Front")
    plt.savefig("Pareto Front_yield_carbon.png")
    os.startfile('Pareto Front.png')
    plt.close()

    plt.scatter(result.F[:, 1] * -1, result.X[:, 1] * 100)
    plt.tight_layout()
    plt.xlabel("Yield (Mg/ha)")
    plt.ylabel("Residue retention (%)")
    plt.title("Pareto Front")
    plt.savefig("Pareto Front_re_yield.png")
    os.startfile("Pareto Front_re_yield.png")
    plt.close()

    g = sns.relplot(x=result.F[:, 0] * -1, y=result.X[:, 0], height=6, aspect=1.5)
    plt.xlabel("Soil organic carbon (Mg/ha)")
    plt.ylabel("Nitrogen (kg/ha)")
    plt.title("Pareto Front")
    plt.savefig("Pareto Front_carbon_yield.png")
    os.startfile("Pareto Front_carbon_yield.png")
    plt.close()

    g = sns.relplot(x=result.F[:, 1] * -1, y=result.X[:, 0], height=6, aspect=1.5)
    plt.xlabel("Soil organic carbon (Mg/ha)")
    plt.ylabel("Nitrogen (kg/ha)")
    plt.title("Pareto Front")
    plt.savefig("Pareto Front_nit_yield.png")
    os.startfile("Pareto Front_nit_yield.png")
    plt.close()

    from moess import moess

    moess(result.F, w=[0.5, 0.5])
    # the best value that maximized both soc and yield were 169.4531935 , 0.99991249, for nitrogen and residue,
    # respectively
    hv = compute_hyper_volume(result.F, normalize=True)
    print("Hyper volume:", hv)
