"""
Created on 2025/12/12
"""
import os
from typing import Union

from apsimNGpy.core_utils.database_utils import read_db_table, get_db_table_names
from apsimNGpy.core_utils.utils import timer
from dotenv import load_dotenv
from pathlib import Path
import yaml
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from pandas import read_csv, DataFrame, to_numeric, concat, Categorical
from utils import plot_reg_fit2, RESULTS, pbias, open_file, carbon_to_Mg
import numpy as np
from loguru import logger
with open("manifest.yml", "r") as f:
    cfg = yaml.safe_load(f)
load_dotenv()
Base_dir = Path(__file__).parent
observed_data_source = 'Morrow Plots Observed_Data'
observed_data_path = Base_dir / observed_data_source
carbon_data_path = observed_data_path / 'all_extracted.csv'
maize_yield_data = observed_data_path / 'mp_yield_sorted.csv'
carbon = read_csv(carbon_data_path, )
maize_yield = read_csv(maize_yield_data)
maize_yield.eval('Plotid = plot', inplace=True)
METRICS = {}
from apsimNGpy.core.config import apsim_bin_context

source_files = Base_dir / 'APSIMx_base_files'
bin_path = os.getenv('7844')
# with apsim_bin_context(bin_path):
#  from apsimNGpy.core.apsim import ApsimModel
apsimCarbonColumn = 'SOC_0_15CM'
carbon['year'] = carbon['year'].astype('int')
carbon['Plotid'] = carbon['plotid']
carbon.to_csv(carbon_data_path, index=False)
# ____________________________
# Constant variable names
# ______________________________
database_info = cfg['database_info']
measured_soc_col = database_info['columns']['morrow_plots']['measured_soc']
measured_yield_col = database_info['columns']['morrow_plots']['measured_yield']
predicted_soc_col = database_info['columns']['morrow_plots']['predicted_soc']
predicted_yield_col = database_info['columns']['morrow_plots']['predicted_yield']

LABELS = {
    measured_soc_col: 'Measured soil organic carbon (Mg ha⁻¹)',
    predicted_soc_col: 'APSIM predicted soil organic (Mg ha⁻¹)',
    measured_yield_col: 'Measured yield (Mg ha⁻¹)',
    predicted_yield_col: 'APSIM predicted yield (Mg ha⁻¹)',
}


def convert_conc_to_kg(concentration, bd, depth):
    SOC_kg = concentration * bd * depth * 1000
    return SOC_kg


def plot(data: Union[dict, DataFrame], predicted_col, measured_col, plot_no, **kwargs):
    fig_name = kwargs.get('fig_name') or RESULTS / f"{plot_no}{predicted_col}-{measured_col}.svg"
    kwargs['fig_name'] = fig_name
    if isinstance(data, dict):
        data = data.get('data')
    else:
        data = data
    data = data.copy()
    poped = ['x', 'y', 'data', 'xlabel', 'ylabel']

    plot_reg_fit2(X=measured_col, y=predicted_col, data=data,
                  xlabel=LABELS.get(measured_col), ylabel=LABELS.get(predicted_col, ), **kwargs)


converted_carbon = carbon_to_Mg(carbon, 'soc_g_per_kg', 'depth_cm', 'BD')


#  ___________________________________
# Plot 3
# --------------------------------------
@timer
def plot3():
    print('Plot 3NC')
    from apsimNGpy.core.apsim import ApsimModel

    with ApsimModel(source_files / 'Plot3NC.apsimx') as apsim:
        logger.info('Running ApsimModel...')
        apsim.run(verbose=False)
        # evaluate carbon
        carbon_ev = apsim.evaluate_simulated_output(ref_data=converted_carbon, table='Annual',
                                                    index_col=['year', 'Plotid'],
                                                    target_col=apsimCarbonColumn, ref_data_col='soc_Mg')

        # plot carbon
        plot(carbon_ev, predicted_col=predicted_soc_col, measured_col=measured_soc_col, plot_no=3, fig_size=(10, 5))
        data = carbon_ev['data']
        logger.info(pbias(data[measured_soc_col], simulated=data[predicted_soc_col]))
        print()
        # evaluate maize yield
        ev_yield = apsim.evaluate_simulated_output(ref_data=maize_yield, table='MaizeR',
                                                   index_col=['year', 'Plotid'], target_col='grainwt',
                                                   ref_data_col='measured_yield')
        data = ev_yield['data']
        print(pbias(data[measured_yield_col], simulated=data[predicted_yield_col]))
        # Plot fitted yield among observed and predicted
        data["phases"] = np.select(
            condlist=[
                data["year"] <= 1955,
                data["year"] > 1955,
                data["year"] > 1968,
            ],
            choicelist=[
                "Phase 1",
                "Phase 2",
                'Phase 2'
            ]
        )
        plot(data, predicted_col=predicted_yield_col, measured_col=measured_yield_col, plot_no=3, color_by='phases', )


# obserevd predicted plot plots
def each_plot_pred_obs(df, measured_col, predicted_col, fig_name='_.png'):
    import seaborn as sns
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MultipleLocator
    from utils import open_file

    data = df.copy()
    ylab = ''

    # Rename columns for plotting
    data.rename(columns={measured_col: 'Measured'}, inplace=True)
    data.rename(columns={predicted_col: 'APSIM'}, inplace=True)

    # Axis labels
    if measured_col == measured_yield_col:
        ylab = 'Corn grain yield (Mg ha⁻¹)'
    if predicted_col == predicted_soc_col:
        ylab = 'Soil organic carbon (Mg ha⁻¹)'

    # Melt data for seaborn
    d_melted = data.melt(id_vars=['Plotid', 'year'])
    d_melted.sort_values(by='year', ascending=True, inplace=True)
    d_melted['variable'] = Categorical(d_melted['variable'], ordered=True)
    for p in d_melted['Plotid'].unique():
        print(p)

        di_f = d_melted[d_melted['Plotid'] == p].copy()

        # Create plot (figure-level)
        g = sns.relplot(
            data=di_f,
            x='year',
            y='value',
            hue='variable',
            kind='line',
            errorbar=None,
            height=8,
            aspect=1.4,
            lw=2.5
        )

        # ---- IMPORTANT: work with the correct axis ----
        ax = g.ax

        # Titles and labels
        ax.set_title(p, fontsize=16)
        ax.set_xlabel('Time (Years)', fontsize=18)
        ax.set_ylabel(ylab or 'value', fontsize=18)

        # Tick formatting
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)

        # Ensure consistent x-limits
        ax.set_xlim(di_f['year'].min(), di_f['year'].max())

        # # Apply decade ticks conditionally
        if  measured_col ==measured_soc_col:
            if p.endswith('NB'):
                ax.xaxis.set_major_locator(MultipleLocator(1))
            else:

                ax.xaxis.set_major_locator(MultipleLocator(2.5))
            #ax.xaxis.set_minor_locator(MultipleLocator(5))
            ax.tick_params(axis='x', which='minor', length=4)

        # Legend cleanup
        g.legend.set_title('')

        plt.tight_layout()

        # Save figure
        fig_n = RESULTS / f"_{p}-{measured_col}-{predicted_col}.svg"
        plt.savefig(fig_n)
        open_file(fig_n)

        plt.close()


def plot_pred(df, measured_col, predicted_col, fig_name='_.svg'):
    import seaborn as sns
    from matplotlib import pyplot as plt
    from utils import open_file
    data = df.copy()
    ylab = ''

    data.rename(columns={measured_col: 'Measured'}, inplace=True)
    data.rename(columns={predicted_col: 'APSIM'}, inplace=True)

    if measured_col == measured_yield_col:
        ylab = 'Corn grain yield (Mg ha⁻¹)'
    if predicted_col == predicted_soc_col:
        ylab = 'Soil organic carbon (Mg ha⁻¹)'
    d_melted = data.melt(id_vars=['Plotid', 'year'], )
    d_melted.sort_values(by='year', ascending=True, inplace=True)
    g = sns.relplot(data=d_melted, x='year', y='value', hue='variable', kind='line', errorbar=None, height=8,lw=2.5,
                    aspect=1.5, col='Plotid', col_wrap=2)

    g.legend.set_title('')
    plt.tight_layout()
    plt.ylabel(ylab or 'value', fontsize=18)
    plt.xlabel('Time (Years)', fontsize=18)
    plt.savefig(RESULTS / fig_name)
    #open_file(RESULTS / fig_name)
    plt.close()
    gg = sns.catplot(data=d_melted, x='Plotid', y='value', hue='variable', kind='box', showfliers=False)
    gg.legend.set_title('')
    plt.xlabel('')
    plt.ylabel(ylab or 'value', fontsize=18)
    cat_plot_fignam = f"_cat_plot{fig_name}.svg"
    plt.savefig(RESULTS / cat_plot_fignam)
   # open_file(RESULTS / cat_plot_fignam)
    plt.close()
    return d_melted


@timer
def evaluate(base_apsimx_file_path, plot_no, bin_key='7493', ):
    _plots = RESULTS / 'plot'
    _plots.mkdir(parents=True, exist_ok=True)
    plot_no_dir = _plots / plot_no
    plot_no_dir.mkdir(parents=True, exist_ok=True)
    print('analysing for plot{}'.format(plot_no))
    bin_7493 = os.getenv(bin_key)

    """
    Due to the bug in the current apsim version,apsimx files could not serialize properly and could not allow us to insert cultvar in soybean and oats,
    :param bin_key: key pointing to the apsim binaries path in .env file
    
    .. note::
    
        the file is run in GUI, and here we are only accessing its database 
        
    :return: 
    """
    apsimx_file_path = source_files / base_apsimx_file_path
    db = apsimx_file_path.with_suffix('.db')
    if not bin_key:
        raise ValueError('plot 4 requires bin_key pointing to the proper apsim binaries preferably APSIM version 7493')
    with apsim_bin_context(bin_7493):
        from apsimNGpy.core.pythonet_config import load_pythonnet, configuration
        from apsimNGpy.core.apsim import ApsimModel as Apsim
        with Apsim(source_files / apsimx_file_path) as apsim:
            predicted_carbon = read_db_table(db, 'Annual')

            ev_carbon = apsim.evaluate_simulated_output(ref_data=converted_carbon, table=predicted_carbon,
                                                        index_col=['year', 'Plotid'],
                                                        target_col=apsimCarbonColumn, ref_data_col='soc_Mg')
            data = ev_carbon['data']

            ev_carbon.get('metrics')['Pbias'] = pbias(data[measured_soc_col], simulated=data[predicted_soc_col])

            METRICS[f"soc_{plot_no}"] = ev_carbon.get('metrics')

            METRICS[f"soc_{plot_no}"]['Pbias'] = pbias(data[measured_soc_col], simulated=data[predicted_soc_col])

            print('pbias carbon', pbias(data[measured_soc_col], simulated=data[predicted_soc_col]))

            plot(ev_carbon, predicted_col=predicted_soc_col, measured_col=measured_soc_col, plot_no=plot_no,
                 fig_size=(8, 6), fig_name=plot_no_dir / f'{predicted_soc_col}.svg')

            # _____________________________________
            # Yield
            # ---------------------------------------
            predicted_yield = read_db_table(db, 'MaizeR')
            predicted_yield[predicted_yield_col] /= 1000
            my = maize_yield.copy()
            my[measured_yield_col] = to_numeric(
                my[measured_yield_col], errors="coerce"
            )

            my = my.dropna(subset=[measured_yield_col])

            my[measured_yield_col] /= 1000
            ev3 = apsim.evaluate_simulated_output(ref_data=my, table=predicted_yield,
                                                  index_col=['year', 'Plotid'], target_col='grainwt',
                                                  ref_data_col='measured_yield')
            METRICS[f"yield_{plot_no}"] = ev3.get('metrics')
            data = ev3.get('data')
            pb = pbias(data[measured_yield_col], data[predicted_yield_col])
            ev3.get('metrics')['Pbias'] = pb
            print('pbias', pb)
            data['year'] = data['year'].astype(int)
            data["phases"] = np.where(
                data["year"] <= 1937,
                "Phase 1",
                np.where(
                    data["year"] <= 1945,
                    "Phase 2",
                    np.where(
                        data["year"] <= 1967,
                        "Phase 3",
                        "Phase 4"
                    )
                )

            )

            plot(ev3, predicted_col=predicted_yield_col, measured_col=measured_yield_col, plot_no=plot_no,
                 fig_size=(8, 6))
            print(pb)
            METRICS[f"yield_{plot_no}"]['Pbias'] = pb
            print('==================================')
            import pandas as pd
            df = ev3['data'][['year', 'Plotid', predicted_yield_col, measured_yield_col]]

            # with pd.option_context('display.max_rows', None):
            #     print(df)
            return df, ev_carbon['data']


if __name__ == '__main__':
    from utils import collect_used_params, tabulate_data
    from xlwings import view

    # ________________________
    # Plot 3
    # ________________________
    # evaluate(base_apsimx_file_path='Plot3NC1.apsimx', plot_no='3')
    grain3, carbon3 = evaluate(base_apsimx_file_path='Plot3_7493.apsimx', plot_no='3')
    grain_m3, carbon_m3 = evaluate(base_apsimx_file_path='Plot3_7493_mick_uncalibrated.apsimx', plot_no='m_3')
    grain_m4, carbon_m4 = evaluate(base_apsimx_file_path='Plot4_7493_mick_uncalibrated.apsimx', plot_no='m_4')
    pp3 = plot_pred(carbon3, measured_col=measured_soc_col, predicted_col=predicted_soc_col)
    py3 = plot_pred(grain3, measured_col=measured_yield_col, predicted_col=predicted_yield_col)

    # ________________________
    # Plot 4
    # __________________________
    grain4, carbon4 = evaluate(base_apsimx_file_path='Plot4_7493.apsimx', plot_no='4')
    pp = plot_pred(carbon4, measured_col=measured_soc_col, predicted_col=predicted_soc_col)
    py = plot_pred(grain4, measured_col=measured_yield_col, predicted_col=predicted_yield_col)
    if METRICS:
        eval_data = DataFrame(METRICS)
        eval_data.to_csv(RESULTS / 'evaluation.csv', index=True)
        select = [i.upper() for i in ['rmse', 'rrmse', 'mae', 'ME', 'wia']]
        select.append('Pbias')
        df_metrics = eval_data.T[select]
    # storing all the input parameters used in the evaluation files
    #usedparams = collect_used_params(['Plot4_7493.apsimx', 'Plot3_7493.apsimx'], json_file='parameters.json')
    sowing_logic_nb_plots = tabulate_data([source_files / 'Plot3_7493.apsimx', source_files / 'Plot4_7493.apsimx'],
                                          paths=['.Simulations.contnous corn.3NB.Field.SowMaizeWithSwitch1',
                                                 '.Simulations.corn oats and soybean.4NB.Field11.SowMaizeWithSwitch1']).T
    # view(sowing_logic_nb_plots)
    sowing_logic_nc_plots = tabulate_data([source_files / 'Plot3_7493.apsimx', source_files / 'Plot4_7493.apsimx'],
                                          paths=['.Simulations.contnous corn.3NC.Field.SowMaizeWithSwitch1',
                                                 '.Simulations.corn oats and soybean.4NC.Field11.SowMaizeWithSwitch1']).T
    # view(sowing_logic_nc_plots, table=False)
    carbon3['Plotid'] = carbon3['Plotid'].map({
        '3NC': '3NC',
        '3NB': '3NB'
    })
    gd = concat([carbon3, carbon])
    # import seaborn as sns
    # gd  = gd.melt(id_vars=['year', 'Plotid'])
    #
    # sns.relplot(data=gd, x='year', y='value', hue='variable', kind='line', errorbar=None, col='Plotid', col_wrap=2)
    # plt.savefig('Plot4_7493.png')
    #
    # open_file('Plot4_7493.png')
    import gc

    print(gc.collect())
    each_plot_pred_obs(carbon4, measured_col=measured_soc_col, predicted_col=predicted_soc_col)
    each_plot_pred_obs(carbon3, measured_col=measured_soc_col, predicted_col=predicted_soc_col)
    each_plot_pred_obs(grain4, measured_col=measured_yield_col, predicted_col=predicted_yield_col)
    each_plot_pred_obs(grain3, measured_col=measured_yield_col, predicted_col=predicted_yield_col)
    #
