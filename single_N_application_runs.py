import os
import shutil
import sys
from pathlib import Path
import pandas as pd
from apsimNGpy.core.apsim import ApsimModel

from settings import RESIDUE_RATES, RESULTS
from apsimNGpy.settings import logger
from mgt_reporter import make_report, make_mgt_text
import matplotlib.pyplot as plt
from utils import create_experiment, merge_tables, N_RATES
from utils import open_file
from xlwings import view
from apsimNGpy.manager.weathermanager import read_apsim_met
from settings import Central as South

path_to_MP_data = Path.cwd() / 'Data-analysis-Morrow-Plots/APSIMX FILES'
scratch_DIR = Path.cwd() / 'OutPuts'
from mgt_reporter import summarize_outputs
import seaborn as sns

TILLAGE_DEPTH = 150  # mm
DPI = 600
scratch_DIR.mkdir(parents=True, exist_ok=True)


figures = Path(__file__).parent / 'Figures'
figures.mkdir(parents=True, exist_ok=True)
Y_FONTSIZE = 18
X_FONTSIZE = 18
SINGLE_METHOD_STR = 'Single'
SPLIT_METHOD_STR = 'Split'
n_rates = N_RATES.split(',')

palette = (
    "tan",  # blue
    "orange",  # orange
    "green",  # green
    "red",  # red
    'darkorchid'
)

COLOR_PALETTE = 'deep'
previously_tested = ((-88.226111, 40.104167),
                     (-95.114313, 42.131437), (-89.999216, 43.762696))
lonlatTest = -88.226111, 40.104167
# ________________________ single experiment___________________________________________________
logger.info(f"running no single fertilizer model under \n {RESIDUE_RATES} residue  rates, {N_RATES} N rates")
single_model = create_experiment(base_file='base_single.apsimx', lonlat=lonlatTest)
# model.edit_model(model_type='Models.Soils.Organic', model_name='Organic', FInert=[0.65,0.85], FBiom =[0.035,
# 0.040], Carbon = [1.8, 1.2]) add nitrogen levels 0, 165, 244, 326
single_model.add_factor(specification=f"[single_N_at_sowing].Script.Amount = {N_RATES}", factor_name='Nitrogen')
# add residue removal levels, 0, 0.5, 0.75, 1 as fraction
single_model.add_factor(specification=f"[Tillage].Script.Fraction = {RESIDUE_RATES}", factor_name='Residue')
# single_model.add_factor(specification=f"[Tillage].Script.Depth = 100, 150, 200, 250", factor_name='Depth')
single_model.edit_model(model_type='Models.Manager', model_name='Tillage', Depth=TILLAGE_DEPTH)
# model.add_factor(specification=f"[Tillage].Script.TillageDate = {TILLAGE_DATES}", factor_name='TillageDate')
# single_model.get_soil_from_web(lonlat=South, thinnest_layer=150)
# single_model.get_weather_from_web(lonlat=South, start=1981, end=2022, source='daymet')
single_model.adjust_dul()
# single_model.edit_model(model_name='Clock', model_type='Models.Clock',Start='1982-01-01', End='2005-12-31')
single_model.run('carbon', verbose=True)
single_model.results.to_csv(scratch_DIR / 'single_N_at_sowing.csv', index=False)

reports = single_model.inspect_model(model_type='Models.Report', fullpath=False)
#
# df_soc_single = single_model.get_simulated_output('mineralization')
# df_soc_single['timing'] = SINGLE_METHOD_STR
# df_biomass_single = single_model.get_simulated_output('yield')
# df_biomass_single['timing'] = SINGLE_METHOD_STR
shutil.copy(single_model.datastore, RESULTS / 'single_model.db')
logger.info(f'Saved data to: {RESULTS / 'single_model.db'}\n====================================')
#
# # _____________________________________ split experiment__________________________________________________
#
# logger.info(f"running no split fertilizer model under:\n {RESIDUE_RATES} residue  rates and {N_RATES} N rates")
#
# # ________________before comparisons _____________
# # make_report(single_model, model2)
# # ______________converting nitrogen to str_______________________
# df_soc_single['Nitrogen'] = df_soc_single['Nitrogen'].astype('str')
# # df_soc_split['Nitrogen'] = df_soc_split['Nitrogen'].astype('str')
# df_soc_single['Nitrogen'] = df_soc_single['Nitrogen'].astype('str')
# df_soc_single['tillage'] = TILLAGE_DEPTH
# s_carbon = single_model.get_simulated_output('carbon')
# s_mineral = single_model.get_simulated_output('mineralization')
#
# df_soc_single.to_csv(f'single_{TILLAGE_DEPTH}.csv')
# df_150mm = pd.read_csv(f'single_150.csv')
# df_300mm = pd.read_csv(f'single_300.csv')
# depth = pd.concat([df_150mm, df_300mm])
#
#
# def plot(model_obj, column, table='carbon', style=None, hue='Nitrogen', ):
#     import seaborn as sns
#     if isinstance(model_obj, pd.DataFrame):
#         data = model_obj
#     else:
#         data = model_obj.get_simulated_output(table)
#     sns.relplot(data=data, x='year', y=column,
#                 kind='line',
#                 errorbar=None,
#                 style=style,
#                 palette=COLOR_PALETTE,
#                 hue=hue)
#     plt.xlabel('Time (years)', fontsize=18)
#     ylabel = fr"SOC ($\mathrm{{Mg\,ha^{{-1}}}}$; 0–{TILLAGE_DEPTH} mm; BD = 1.45 $\mathrm{{g\,cm^{{-3}}}}$), 5-year rolling mean"
#     plt.ylabel(ylabel, fontsize=18)
#     # _________________soc plot for single____________________________
#     file_name = figures / f'{TILLAGE_DEPTH}_{column}-from-{table}.png'
#
#     plt.savefig(file_name, dpi=DPI)
#     open_file(file_name)
#     plt.close()
#
#
# # _________________soc plot for single____________________________
# file_name = figures / f'{TILLAGE_DEPTH}single_carbon.png'
# mva_single = mva(df_soc_single, window=5, col='SOC_0_15CM')
# plot_mva(mva_single, 'SOC_0_15CM_roll_mean', color_palette='tab10',
#          ylabel=fr"SOC ($\mathrm{{Mg\,ha^{{-1}}}}$; 0–{TILLAGE_DEPTH} mm; BD = 1.45 $\mathrm{{g\,cm^{{-3}}}}$), 5-year rolling mean",
#          xlabel='Time (years)', xtick_size=14)
#
# plt.savefig(file_name, dpi=DPI)
# # open_file(file_name)
# plt.close()
# # # ___________ SOC plot split________________
# # file_name = figures / f'{TILLAGE_DEPTH}_split_carbon.png'
# # mineral_ratio_single = mva(df_soc_split, window=5, col='SOC_0_15CM')
# # g = plot_mva(mineral_ratio_single, 'SOC_0_15CM_roll_mean', color_palette='tab10',
# #              ylabel=r"SOC ($\mathrm{Mg\,ha^{-1}}$; 0–150 mm; BD = 1.45 $\mathrm{g\,cm^{-3}}$), 5-year rolling mean",
# #              xlabel=r"Time (years)")
#
# plt.title('split')
# plt.savefig(file_name, dpi=DPI)
# # open_file(file_name)
# plt.close()
#
# mineralized_yl = fr"Mineralized nitrogen ($\mathrm{{kg\,ha^{{-1}}}}$), 5-year rolling mean"
#
# soc_mg_yl = fr"SOC ($\mathrm{{Mg\,ha^{{-1}}}}$; 0–{TILLAGE_DEPTH} mm; BD = 1.45 $\mathrm{{g\,cm^{{-3}}}}$), 5-year rolling mean"
# som_kg_yl = r"SOM ($\mathrm{kg\,ha^{-1}}$; 0–150 mm; BD = 1.45 $\mathrm{g\,cm^{-3}}$), 5-year rolling mean"
#
#
# def plot_trend_mva(model, response, table='carbon', filename=None, ylabel=None, grouping=('Nitrogen', 'Residue')):
#     filename = filename or f"mva_{response}-from-{table}-table.png"
#     df = model.get_simulated_output(table)
#     if response not in df:
#         raise ValueError(f"{response} is not in {table} table data frame")
#     file_nam = figures / f'{filename}_all_soc.png'
#     _df = mva(df, window=5, col=response, grouping=grouping)
#     plot_mva(_df, f'{response}_roll_mean', color_palette='tab10', style=None,
#              ylabel=ylabel,
#              xlabel='Time (Years)')
#     # plt.tight_layout()
#     plt.savefig(file_nam, dpi=DPI)
#     open_file(file_nam)
#     plt.close()
#
#
# import seaborn as sns
#
# g = sns.relplot(
#     data=single_model.get_simulated_output('yield'), x='year', y='grainwt',
#     hue="Nitrogen",
#     kind="line",
#     errorbar=None,
#     # showfliers=False,
#     linewidth=1,
#     palette='tab10',
#     height=8, aspect=1.4)
# plt.savefig(figures / f'__test.png', dpi=DPI)
#
# # open_file(figures / f'__test.png')
# print('mean surface carbon\n==================')
# sc = single_model.get_simulated_output('yield').groupby('Nitrogen')['grainwt'].mean()
# print(sc)
# print('leaf area index sum\n==================')
# lai = single_model.get_simulated_output('daily').groupby('Nitrogen')['lai'].mean()
# print(lai)
# print('mean total surface carbon (0-15cm)\n==================')
# tc = single_model.get_simulated_output('carbon').groupby(['Nitrogen', ])['SOC_0_15CM'].mean()
# print(tc)
# print('soil water top layer (0-15cm)\n==================')
# wm = single_model.get_simulated_output('water').groupby(['Nitrogen', ])['InCropMeanSoilWaterTopFirstLayer'].mean()
# print(wm)
# print('soil water top layer (0-15cm)\n==================')
# som = single_model.get_simulated_output('daily').groupby(['Nitrogen', ])['SOM'].mean()
# print(som)
# # view(single_model.get_simulated_output('carbon'), table = False)
# print('Mineralized nitrogen in the first layer\n=====================')
# minN = single_model.get_simulated_output('daily').groupby(['Nitrogen', ])['mineralN_ly1'].mean()
# print(minN)
# print('Above ground biomass\n=================')
# abg = single_model.get_simulated_output('yield').groupby(['Nitrogen', ])['maizeyield'].mean()
# print(abg)
#
# print('Below ground biomass\n================================')
# bbg = single_model.get_simulated_output('yield').groupby(['Nitrogen', ])['BBiomass'].mean()
# print(bbg)
#
# # 1) By Nitrogen only, all years, write Excel
# tbl = summarize_outputs(single_model, by=("Nitrogen",), outfile="summary.xlsx")
#
# # 2) Filter to year 1904 (column is 'year'), write CSV
# tbl_start = summarize_outputs(single_model, year=1900, outfile="summary_1904.csv")
#
# # 3) Multi-key grouping (Nitrogen × Residue) for years 1901–1905
# tbl_win = summarize_outputs(single_model,
#                             by=("Nitrogen", "Residue"),
#                             year=slice(1901, 1906))
#
# # 4) Custom aggregation: use mean for LAI instead of the sum
# tbl_mean_lai = summarize_outputs(single_model, agg_overrides={'lai': 'mean'})
# # tbl_mean_lai['min to nitrogen'] = tbl_mean_lai['mean_mineralN_ly1'] / tbl_mean_lai['Nitrogen']
#
# plot_trend_mva(single_model, 'mineralN_ly1', table='daily', ylabel=mineralized_yl)
# # plot_trend_mva(single_model, 'InCropMeanSoilWaterTopFirstLayer', table='water')
# # plot_trend_mva(single_model, 'catm1', table='daily', ylabel=soc_mg_yl)
# # plot_trend_mva(single_model, 'SOC1', table='carbon', ylabel=som_kg_yl)
#
#
# maize_default = create_experiment('maize-default', lonlat=lonlatTest)
# maize_default.add_factor(specification=f"[single_N_at_sowing].Script.Amount = {N_RATES}", factor_name='Nitrogen')
# # add residue removal levels, 0, 0.5, 0.75, 1 as fraction
# maize_default.add_factor(specification=f"[Tillage].Script.Fraction = {RESIDUE_RATES}", factor_name='Residue')
# # single_model.add_factor(specification=f"[Tillage].Script.Depth = 100, 150, 200, 250", factor_name='Depth')
# maize_default.edit_model(model_type='Models.Manager', model_name='Tillage', Depth=TILLAGE_DEPTH)
# # maize_default.run()
# # plot_trend_mva(maize_default, 'SOC_0_15CM', table='carbon', ylabel=soc_mg_yl)
# #
# plot(single_model, 'surface_organic_matter')
#
# g = sns.catplot(
#     x='Nitrogen', y='top_mineralized_N', data=depth,
#     kind='bar', estimator='mean', hue='Residue',
#     height=6, aspect=1.5,
#     palette=COLOR_PALETTE, errorbar=None, sharey=True
# )
# xtick_size = 14
# ytick_size = 14
# ylabel = fr"Mineralized nitrogen ($\mathrm{{kg\,ha^{{-1}}}}$)"
# xlabel = r"Nitrogen ($\mathrm{kg\,ha^{-1}}$)"
# ylabel_size = 20
# xlabel_size = 20
#
# g.set_axis_labels("", "")
#
# # Enforce tick sizes for every facet (robustly)
# for ax in g.axes.flat:
#     ax.tick_params(axis='x', which='both', labelsize=xtick_size)
#     ax.tick_params(axis='y', which='both', labelsize=ytick_size)
#     # Fallback in case styles/backends override tick_params
#     plt.setp(ax.get_xticklabels(), fontsize=xtick_size)
#     plt.setp(ax.get_yticklabels(), fontsize=ytick_size)
#     # Also scale the scientific-notation offset text if present
#     ax.yaxis.get_offset_text().set_size(ytick_size)
#
# # Shared labels
# g.fig.supylabel(ylabel, x=0.002, fontsize=ylabel_size)
# g.fig.supxlabel(xlabel, y=0.002, fontsize=xlabel_size)
#
# # Legend cleanup + size
# leg = getattr(g, "_legend", None) or getattr(g, "legend", None)
# if leg is not None:
#     leg.set_title(None)
#     for txt in leg.get_texts():
#         txt.set_fontsize(min(xtick_size, ytick_size))
#
# fn = "depth_comp.png"
# g.fig.savefig(fn, bbox_inches="tight", dpi=600)
#
# plt.close(g.fig)
#
# from settings import path_to_MP_data
#
# gui = ApsimModel(path_to_MP_data / 'single_gui.apsimx')
# # gui.run()
# # plot_trend_mva(gui, 'SOC_0_15CM', table='carbon', ylabel=soc_mg_yl)
# from mgt_reporter import make_mgt_text
#
# tx = make_mgt_text(single_model)
#
# single_model.add_memo(tx)
# from apsimNGpy.core.config import apsim_version, stamp_name_with_version
#
#
# def cat_plot(model, table, response, x='Residue', hue='Nitrogen', kind='bar', **kwargs):
#     df = model.get_simulated_output(table)
#     if response not in df:
#         raise ValueError(f'{response} is not found in {df.columns.tolist()}')
#     df.sort_values(by=['Nitrogen', 'Residue'], inplace=True)
#     sns.catplot(data=df, x=x, y=response, hue=hue, kind=kind, palette=COLOR_PALETTE, **kwargs)
#     fn = f"{x}-{response}-from{table}-{kind}.png"
#
#     plt.ylabel(r"Soil temperature ($^\circ$C)")
#     plt.savefig(fn, dpi=600)
#     open_file(fn)
#     plt.close()
#
#
# import numpy as np
#
# nitrogen = ",".join(map(str, np.arange(0, 326, 5)))
#
# # sg = create_experiment('base_single.apsimx', lonlatTest)
# # sg.add_factor(specification=f"[single_N_at_sowing].Script.Amount = {nitrogen}", factor_name='Nitrogen')
# # # add residue removal levels, 0, 0.5, 0.75, 1 as fraction
# # sg.add_factor(specification=f"[Tillage].Script.Fraction = {RESIDUE_RATES}", factor_name='Residue')
# # # single_model.add_factor(specification=f"[Tillage].Script.Depth = 100, 150, 200, 250", factor_name='Depth')
# # sg.edit_model(model_type='Models.Manager', model_name='Tillage', Depth=TILLAGE_DEPTH)
# # model.add_factor(specification=f"[Tillage].Script.TillageDate = {TILLAGE_DATES}", factor_name='TillageDate')
# # sg.run()
# # from settings import RESULTS
# # shutil.copy(sg.datastore, RESULTS/'gd.db')
# # sg.results.to_csv(RESULTS/'gd.csv')
if __name__ == '__main__':

    path = Path(single_model.path)
    # file_name = stamp_name_with_version(path)
    # sm= single_model.save(file_name=file_name)
    single_model.inspect_model_parameters('Weather', 'Weather')
    if not single_model.ran_ok:
        single_model.run()


    def test_mva(instance, title='', **kwargs):
        instance.plot_mva(**kwargs)
        plt.title(title)
        x, y, tab = kwargs.get('time_col'), kwargs.get('response'), kwargs.get('table')
        if not isinstance(tab, str):
            tab = ""
        name = f"mva_single_{x}-{y}-{tab}{TILLAGE_DEPTH}.png"
        plt.tight_layout()
        plt.savefig(name, dpi=600)
        os.startfile(name)
        plt.close()


    def cat_plot(instance, title="", **kwargs):
        instance.cat_plot(**kwargs)
        x, y, tab = kwargs.get('time_col'), kwargs.get('response'), kwargs.get('table')
        if not isinstance(tab, str):
            tab = ""
        name = f"series_single_{x}-{y}-{tab}{TILLAGE_DEPTH}.png"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(name, dpi=600)
        os.startfile(name)
        plt.close()


    def series(instance, title="", **kwargs):
        instance.series_plot(**kwargs)
        plt.title(title)
        plt.tight_layout()
        x, y, tab = kwargs.get('time_col'), kwargs.get('y'), kwargs.get('table')
        if not isinstance(tab, str):
            tab = ""
        name = f"series_single_{x}-{y}-{tab}{TILLAGE_DEPTH}.png"
        plt.savefig(name, dpi=600)
        os.startfile(name)
        plt.close()


    # single_model.preview_simulation()
    data = merge_tables(single_model, ['carbon', 'yield'])
    data.eval("cnr=SurfaceOrganicMatter_Carbon/SurfaceOrganicMatter_Nitrogen", inplace=True)

    # test_mva(single_model, table=data, response='cnr', time_col='year', window=5,
    #          grouping=("Residue", 'Nitrogen'), estimator='mean',
    #          errorbar=None, col='Residue', col_wrap=2, hue="Nitrogen")

    # single_model.preview_simulation()
    test_mva(single_model, table='carbon', response='SOC_0_15CM', time_col='year', hue="Nitrogen",
             grouping=("Residue", "Nitrogen"), estimator='mean',
             errorbar=None, col='Residue', col_wrap=2, )
    data['N'] = pd.Categorical(data['Nitrogen'], ordered=True)
    data['R'] = pd.Categorical(data['Residue'], ordered=True)

    from utils import calculate_soc_changes

    calculate_soc_changes(data, col="SOC_0_15CM")
    single_soc_changes = calculate_soc_changes(data, col="SOC_0_15CM")
    single_soc_changes['timing'] = 'Single'

    carbon = single_model.get_simulated_output('carbon')
    # carbon.sort_values(by='')
    carbon.sort_values(by=['Residue', 'Nitrogen'], inplace=True)

    # cat_plot(single_model, table=carbon, expression=None, y='SOC_0_15CM', x='Residue',
    #          estimator='mean', kind='box', palette='tab10', showfliers=False,
    #          errorbar=None, hue="Nitrogen")
    #
    # cat_plot(single_model, table=data, title='SOC', expression=None, y='cnr', x='Nitrogen',
    #          estimator='mean', kind='box', palette='tab10', showfliers=False,
    #          errorbar=None, hue="Residue")
    #
    # cat_plot(single_model, table=data, title='SOC', expression=None, y='cnr', x='Residue',
    #          estimator='mean', kind='box', palette='tab10', showfliers=False,
    #          errorbar=None, hue="Nitrogen")

# carbon mineralisation
# soil carbon
# grain yield
#
calculate_soc_changes(data, col='SOC_0_15CM', grouping=('Nitrogen', "year", "Residue"))
