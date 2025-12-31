from utils import read_or_run


if __name__ == "__main__":
    #
    model = read_or_run(apsim_file='tillage_scenario.apsimx')

    model.series_plot(data=model.results, x='SOC_0_15CM', y='SOC_0_15CM', hue="SimulationName")

    model.inspect_model(model_type="Report")
    pass
