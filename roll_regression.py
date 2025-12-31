import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingWLS as RollingOLS
from settings import RESULTS
from utils import  open_file
from data_manager import merge_tables
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from plotting import LABELS, PlotConstants
from simulate_quadratic_fit_data import data_base_qp_gen
pd.plotting.register_matplotlib_converters()
from apsimNGpy.settings import logger
from apsimNGpy.core_utils.database_utils import read_db_table
from simulate_scenario_data import datastore, table_name
if __name__ == '__main__':
    sns.set_style("darkgrid")
    # Sort by year
    data = read_db_table(datastore, table_name)
    data_at = float(data[data['year'] == data['year'].min()]['SOC1'].iloc[0])
    data.eval('SOC1 = SOC_0_15CM - 52.26', inplace=True)
    data['years'] = data['year'].astype(int)

    data = data.sort_values(['year', 'Nitrogen', 'Residue'])
    data.set_index('years', inplace=True)
    data['Nitrogen'] = data['Nitrogen'].astype('float')
    data['Residue'] = data['Residue'].astype('float')
    # data.eval('Residue = 1-Residue', inplace=True)
    response = 'SOC1'

    # Define predictors and response
    data['date'] = data['year'].astype('int')
    # data = data.groupby(['date', primary_factor])[response].mean().reset_index()
    res = []
    data.reset_index(inplace=True, drop=True)
    data_copy = data.copy()
    base_window = 14
    primary_factor = 'Nitrogen'
    secondary_factor = 'Residue'
    for val in list(data_copy[primary_factor].unique()):
        data = data_copy[data_copy[primary_factor] == val]
        assert not data.empty, f'data is empty for : {val}'
        # data = data.groupby(['date'])[response].mean().reset_index()
        exog_var = ['date', 'Residue']
        data['id'] = data['date']
        data.sort_values(by=exog_var, inplace=True)
        data.set_index(['id'], inplace=True)
        if secondary_factor in exog_var:
            size_of_sec_factor = len(data[secondary_factor].unique())
            window = base_window * size_of_sec_factor
        else:
            window = base_window
        X = sm.add_constant(data[exog_var])
        y = data[response]

        # Choosing a rolling window

        # Fit rolling regression
        model = RollingOLS(endog=y, exog=X, window=window)
        result = model.fit(params_only=False, cov_type='HCCM')

        # Extract time-varying coefficients
        coeffs = result.params

        # coeffs.index = data['date']
        coeffs.head()

        import seaborn as sns

        c = coeffs.copy()

        # c.drop( axis=1, inplace=True)
        c['year'] = data['date']

        c['N'] = val
        print(val, '; completed')
        res.append(c)
    all_res = pd.concat(res)
    all_res['N'] = pd.Categorical(all_res['N'], ordered=True)
    all_res.reset_index(inplace=True)
    max_year = int(data_copy.year.max())


    def print_for_year(year, show_max=True):
        year = int(year)
        ydata = all_res[all_res['year'] == year]

        if show_max:
            max_index = np.argmax(ydata['date'])
            print(f"maximum date slope in {year}\n", ydata.iloc[max_index])
        print('========================')
        return ydata


    first = print_for_year(data_copy.year.min() + base_window)
    last = print_for_year(data_copy.year.max())
    all_res.drop_duplicates(inplace=True)
    # when were the changes in carbon?
    max_indeX = int(np.argmax(all_res['date']))
    highest = all_res.iloc[max_indeX]
    mean_grp = all_res.groupby('N').mean()

    logger.info(f'mean of all groups:\n {mean_grp}')
    w_index = np.argmax(mean_grp['date'])
    print(mean_grp.iloc[w_index])
    for ex_var in [*exog_var]:
        # sns.relplot( x=c['year'], y=c['n_r'],  kind='line', height=10, aspect=1.5)
        g = sns.relplot(data=all_res, x='year', y=ex_var, kind='line', height=10, aspect=1.5, errorbar=None, hue='N')

        # horizontal line at y = 0 across the full x-range
        g.refline(y=0, linestyle='--', linewidth=1)
        plt.savefig(RESULTS / f'_{ex_var}_roll_effects2.png', dpi=600)
        # open_file(RESULTS / f'_{ex_var}_roll_effects2.png')

    data_copy['Nitrogen'] = pd.Categorical(data_copy['Nitrogen'], ordered=True)
    g = sns.relplot(data=data_copy, x='year', y='SOC1', kind='line', height=10, aspect=1.5, errorbar=None,
                    hue='Nitrogen', col='Residue', col_wrap=2)

    # horizontal line at y = 0 across the full x-range
    # for ax in (g.axes.flat if hasattr(g, "axes") else [g.ax]):
    #     ax.axhline(0, linestyle='--', linewidth=1)

    plt.tight_layout()
    # plt.savefig(RESULTS / f'_{ex_var}_roll_effects3.png')
    # open_file(RESULTS / f'_{ex_var}_roll_effects3.png')
    # fig = result.plot_recursive_coefficient(variables=None, figsize=(14, 6))
    plt.savefig(RESULTS / 'roll_effects.png', dpi=600)
# open_file(RESULTS / 'roll_effects.png')

import pandas_datareader as pdr

if __name__ == '__main__':
    sns.set_style("darkgrid", rc={"font.family": "DejaVu Sans"})
    from utils import find_soc_equilibrium, calculate_soc_changes
    from change_metrics import compute_last_minus_first_change
    from plotting import relplot

    df = read_db_table(datastore, table_name)
    df1 = read_db_table(datastore, table_name)
    soc_changes = compute_last_minus_first_change(data=df1.copy(), grouping=['Nitrogen', 'Residue'], col='SOC_0_15CM')
    soc_changes['Nitrogen'] = soc_changes['Nitrogen'].astype(int)
    soc_changes['Nitrogen'] = pd.Categorical(soc_changes['Nitrogen'], ordered=True)
    from plotting import relplot
    relplot(data=soc_changes, show=True, add_scatter=False, x='Nitrogen', y='ΔSOC_0_15CM', hue='Residue',  kind='line')
    soc_changes.sort_values(by=['Residue', 'ΔSOC_0_15CM'], ascending=False, inplace=True)
    with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", 0,  # no wrapping
            "display.max_colwidth", None,
            "display.expand_frame_repr", False, ):
        print(soc_changes)
