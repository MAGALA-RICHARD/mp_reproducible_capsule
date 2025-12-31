import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingWLS as RollingOLS
from settings import RESULTS
from utils import open_file
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from apsimNGpy.core_utils.database_utils import read_db_table
from data_manager import merge_tables
from simulate_scenario_data import datastore, table_name

pd.plotting.register_matplotlib_converters()
if __name__ == '__main__':
    sns.set_style("darkgrid")
    # Sort by year
    data = read_db_table(datastore, table_name)
    data['years'] = data['year'].astype(int)

    data[['yy', 'Nn', 'Rr']] = data[['year', 'Nitrogen', 'Residue']]
    data.set_index(['yy', 'Nn', 'Rr'], inplace=True)
    data = data.sort_values(['year', 'Nitrogen', 'Residue'])
    data['Nitrogen'] = data['Nitrogen'].astype('float')
    data['Residue'] = data['Residue'].astype('float')
    response = 'SOC1'

    # Define predictors and response
    data['date'] = data['year'].astype('int')
    # data = data.groupby(['date', primary_factor])[response].mean().reset_index()
    res = []
    data_copy = data.copy()
    base_window = 14
    primary_factor = 'Nitrogen'
    secondary_factor = 'Residue'
    exog_var = ['Nitrogen',]
    data = data_copy
    X = sm.add_constant(data[exog_var])
    y = data[response]

    # Choosing a rolling window

    # Fit rolling regression
    window = base_window * 4 * 4
    model = RollingOLS(endog=y, exog=X, window=window)
    result = model.fit(params_only=True)

    # Extract time-varying coefficients
    coeffs = result.params

    # coeffs.index = data['date']
    coeffs.head()

    import seaborn as sns

    c = coeffs.copy()

    # c.drop( axis=1, inplace=True)
    c['year'] = data['year']
    c['N'] = pd.Categorical(data['Nitrogen'], ordered=True)
    c['R'] = pd.Categorical(data['Residue'], ordered=True)

    res.append(c)
    all_res = pd.concat(res)

    all_res.reset_index(inplace=True)
    max_year = int(data_copy.year.max())

    all_res.drop_duplicates(inplace=True)
    # when were the changes in carbon?

    c.reset_index(inplace=True)
    c.drop_duplicates(inplace=True)
    for ex_var in [*exog_var]:
        # sns.relplot( x=c['year'], y=c['n_r'],  kind='line', height=10, aspect=1.5)
        g = sns.relplot(data=c, x='year', y=ex_var, kind='line', height=10, aspect=1.5, errorbar=None, hue='N')

        # horizontal line at y = 0 across the full x-range
        if not all(c[ex_var] > 0):
            g.refline(y=0, linestyle='--', linewidth=1)

        plt.savefig(RESULTS / f'n_{ex_var}_roll_effects2.png')
        open_file(RESULTS / f'n_{ex_var}_roll_effects2.png')

    data_copy['Nitrogen'] = pd.Categorical(data_copy['Nitrogen'], ordered=True)
    g = sns.relplot(data=data_copy, x='year', y='SOC1', kind='line', height=10, aspect=1.5, errorbar=None,
                    hue='Nitrogen', col='Residue', col_wrap=2)
    fig = result.plot_recursive_coefficient(variables=None, figsize=(14, 18))
    plt.savefig(RESULTS / f'_{ex_var}_roll_effects4.png')
    open_file(RESULTS / f'_{ex_var}_roll_effects4.png')
