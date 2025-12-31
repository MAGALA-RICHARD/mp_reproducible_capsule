"""
Quick wrapper around apsimNGpy's PlotManager to produce and save
moving-window average (MVA) plots from APSIM simulation outputs.

- Defines a lightweight Plot subclass that holds a copy of the input data
  and exposes matplotlib via `self.plot`.
- `mva_plot(...)` configures labels, palette, and typography; calls
  `Plot.plot_mva` to draw an MVA time-series by group; then applies
  consistent tick/font sizing across facets, adds shared axis labels,
  and writes the figure to `RESULTS` with a generated filename.
- The script section (`__main__`) pulls a table from the APSIM datastore
  and renders several MVA views for SOC (0–15 cm), faceting by Residue
  or Nitrogen and toggling the line hue.

Args (mva_plot):
    data (pd.DataFrame): Tidy table containing APSIM outputs.
    time_col (str): Time column for the MVA (default: 'year').
    table (Any): Passed through to PlotManager (defaults to `data`).
    response (str): Column to smooth/plot (e.g., 'SOC_0_15cm_Mg').
    hue/col/col_wrap: Faceting and color encoding options.
    height/palette/errorbar/window: Visual and smoothing controls.

Side effects:
    - Saves a high-resolution PNG to `RESULTS`.
    - Optionally opens the saved file via `open_file(...)`.
"""
import pandas as pd
from apsimNGpy.core.plotmanager import PlotManager
import matplotlib.pyplot as plt
from plotting import PlotConstants
from labels import LABELS
from settings import RESULTS
from utils import open_file
from simulate_scenario_data import datastore, table_name
import seaborn as sns
from plotting import cat_plot
from data_manager import read_db
from simulate_quadratic_fit_data import datastore as qdatastore
from simulate_quadratic_fit_data import table_name as qtable
from plotting import relplot
from pathlib import Path
import numpy as np

mODULENAME = Path(__file__).stem


class Plot(PlotManager):
    def __init__(self, data):
        super().__init__()
        self.data = data.copy()
        self.plot = plt

    @property
    def results(self):
        return self.data

    def add_report_variable(self):
        pass

    def run(self, **kwargs):
        pass

    def get_simulated_output(self, report_name: str):
        pass


def mva_plot(data, **kwargs):
    kwargs.setdefault("height", PlotConstants.HEIGHT)
    kwargs.setdefault("time_col", "year")
    kwargs.setdefault("table", data)
    kwargs.setdefault("response", kwargs.get("y"))

    base = Plot(data=data)

    # y label text default
    kwargs.setdefault("ylabel", LABELS.get("response"))

    kwargs.setdefault("palette", PlotConstants.PALETTE)
    g = base.plot_mva(**kwargs)

    y = kwargs.get("response")
    g.set_axis_labels("", "")

    xtick_size = PlotConstants.xTICK_SIZE + 2
    ytick_size = PlotConstants.yTICK_SIZE + 2
    ylabel_size = PlotConstants.yLABEL_SIZE + 10  # bumped up more
    xlabel_size = PlotConstants.xLABEL_SIZE + 6

    facet_title_size = 24  # <- requested
    ylabel = LABELS.get(y)
    xlabel = LABELS.get(kwargs.get("time_col", kwargs.get("x")))

    # --- Apply formatting for FacetGrid / single-axes cases ---
    axes_flat = getattr(g, "axes", [getattr(g, "ax", None)])
    axes_flat = np.ravel(axes_flat)

    for ax in axes_flat:
        if ax is None:
            continue

        # tick sizes
        ax.tick_params(axis="x", which="both", labelsize=xtick_size)
        ax.tick_params(axis="y", which="both", labelsize=ytick_size)

        # fallback in case style overrides tick_params
        plt.setp(ax.get_xticklabels(), fontsize=xtick_size)
        plt.setp(ax.get_yticklabels(), fontsize=ytick_size)

        # scale offset text if scientific notation appears
        ax.yaxis.get_offset_text().set_size(ytick_size)


        # facet title size (each small subplot title)
        if ax.get_title():
            ax.set_title(ax.get_title(), fontsize=facet_title_size)

    # shared / global labels when multiple facets
    if len(axes_flat) > 1:
        g.fig.supylabel(ylabel, x=0.002, fontsize=ylabel_size)
        g.fig.supxlabel(xlabel, y=0.002, fontsize=xlabel_size)

    else:
        # single-axes case
        single_ax = axes_flat[0]
        single_ax.set_xlabel(xlabel, fontsize=xlabel_size)
        single_ax.set_ylabel(ylabel, fontsize=ylabel_size)
        #single_ax.legend(loc="best", fontsize=16)


    # legend cleanup + font size
    leg = getattr(g, "_legend", None) or getattr(g, "legend", None)
    if leg is not None:
        leg.set_title(None)
        for txt in leg.get_texts():
            txt.set_fontsize(min(xtick_size, ytick_size))

    # move legend
    sns.move_legend(g, loc="upper right", bbox_to_anchor=(0.98, 0.95), fontsize=20)

    # tighten layout
    base.plot.tight_layout()

    # save figure
    fileName = RESULTS / (
        f"_one_file{kwargs.get('time_col')}_"
        f"{kwargs.get('y')}-"
        f"{kwargs.get('hue')}-"
        f"{kwargs.get('response')}-"
        f"{kwargs.get('height')}"
        f"{PlotConstants.FILE_EXTENSION}"
    )
    fileName = RESULTS / f'{mODULENAME}-mva-{kwargs.get('y')}{kwargs.get('hue')}.svg'
    base.plot.savefig(fileName, dpi=PlotConstants.DPI, bbox_inches='tight')
    open_file(fileName)

    # cleanup
    plt.close()
    base.plot.close()


if __name__ == "__main__":
    sns.set_style("darkgrid", rc={"font.family": "DejaVu Sans"})
    from logger import get_logger
    from simulate_scenario_data import table_name

    logger = get_logger()
    logger.info(f"plotting: moving averages from scenario simulated data")
    data = read_db(datastore, table_name)
    rdata = read_db(datastore, table_name)
    data.eval('Residue = R * 100', inplace=True)
    data['Nitrogen'] = pd.Categorical(data['Nitrogen'], ordered=True)
    data['Residue'] = pd.Categorical(data['Residue'], ordered=True)
    data.eval('R = Residue', inplace=True)
    rdata["ResiduePct"] = (rdata["R"].astype(float) * 100).round().astype(int)
    rdata["Residue incorporation"] = rdata["ResiduePct"].astype(str) + "%"
    mva_plot(data=rdata, response='SOC_0_15cm_Mg', grouping=("Nitrogen", "Residue incorporation",), hue='Nitrogen',
             aspect=1.2,
             errorbar=None, window=7,
             col="Residue incorporation", col_wrap=2)
    mva_plot(data=rdata, response='SOC_0_15cm_Mg', grouping=("Nitrogen", 'Residue',), hue='Residue',
             errorbar=None, window=7,
             col="Nitrogen", col_wrap=2)
    mva_plot(data=data, response='corn_yield_Mg', grouping=("Nitrogen", 'Residue',), hue='Nitrogen',
             errorbar=None, window=14,
             col="Residue", col_wrap=2)
    dg = rdata.groupby(['Nitrogen', 'year'], observed=True)['SOC_0_15cm_Mg'].mean().reset_index()
    mva_plot(data=rdata, response='SOC_0_15cm_Mg', grouping= ("Residue incorporation",), hue="Residue incorporation",
             aspect=1.5,
             errorbar=None, window=7)
    # mva_plot(data=rdata, response='SOC_0_15cm_Mg', grouping=("Nitrogen", "Residue incorporation",), hue="Nitrogen",
    #          aspect=1.5,
    #          errorbar=None, window=7)
    data.eval('total_residue_biomass= (total_biomass_Mg - corn_yield_Mg)', inplace=True)
    carbon_frac = 0.4
    data['R'] = data['R'].astype(float)
    data.eval('residue_incorporated = R * total_residue_biomass', inplace=True)
    data.eval('res_ci =residue_incorporated * total_residue_biomass * @carbon_frac', inplace=True)
    # below is cumulative soc_balance
    data.eval('soc_balance = SOC_0_15cm_Mg- 52.260000', inplace=True)  # 52.260000 is the initial SOC
    data.eval('cfe=(soc_balance/res_ci * soc_balance)', inplace=True)

    # mva_plot(data=data, response='SOC_0_15cm_Mg', grouping=("Nitrogen", 'Residue',), hue='Residue',)
    mva_plot(data=data, response='cfe', grouping=("Nitrogen", 'Residue',), hue='Nitrogen', window=10, errorbar=None)
    mva_plot(data=data, response='cfe', grouping=("Nitrogen", 'Residue',), hue='Residue', window=10, errorbar=None)

    cdata = data.copy()
    cdata['Residue'] = pd.Categorical(cdata['Residue'].astype(float), ordered=True)

    cat_plot(data=cdata, show=True, x='Nitrogen', y='soc_balance', kind='box', hue='Residue')
    cat_plot(data=cdata, show=True, x='Nitrogen', y='cnr', kind='box', hue='Residue', legend_loc='upper right',
             bbox_to_anchor=(0.98, 0.99))
    cat_plot(data=data, show=True, y='top_mineralized_N', x='Nitrogen', kind='box', hue='Residue')
    ac = data[['cnr', 'microbial_carbon', 'top_mineralized_N', 'Nitrogen']].groupby('Nitrogen', observed=True).mean()

    qdata = read_db(qdatastore, qtable)
    qdata['Nitrogen'] = pd.Categorical(qdata['Nitrogen'].astype(float), ordered=True)
    qd = qdata[(qdata['Nitrogen'].astype(float) >= 130) & (qdata['Nitrogen'].astype(float) < 244)].copy()

    # qd["Nitrogen"] = qd["Nitrogen"].replace(qd["Nitrogen"].min(), 137)
    qd.eval('Residue =R * 100', inplace=True)
    qd['Nitrogen'] = qd.Nitrogen.cat.remove_unused_categories()
    relplot(data=qd, show=True, annotation_text='', x='Nitrogen', y='corn_yield_Mg', hue="Residue", errorbar=None,
            add_scatter=False)
    #biomass - minus yield
    qd.eval('other_biomass =total_biomass_Mg -corn_yield_Mg', inplace=True)
    LABELS['other_biomass'] = '101 years average residue biomass (Mg ha⁻¹)'
    relplot(data=qd, show=True, annotation_text='', x='Nitrogen', y='other_biomass', hue="Residue", errorbar=None,
            add_scatter=False)
    relplot(data=qd, show=True, x='Nitrogen', y='SOC_0_15cm_Mg', hue="Residue", errorbar=None, add_scatter=False)

    from change_metrics import find_equilibrium

    # --- apply per treatment ---
    results = []
    from data_manager import read_db
    from simulate_scenario_data import datastore, table_name
    from change_metrics import find_equilibrium
    df = read_db(datastore, table_name)
    df['SOC_balance'] = df['SOC_0_15cm_Mg'] - 52.26
    for (residue_level, n_rate), g in df.groupby(["R", "N"]):
        eq = find_equilibrium(g,
                              soc_col="SOC_balance",
                              year_col="year",
                              w=7,
                              eps=0.05,  # Mg C ha-1 per year tolerance
                              k=5)  # must stay stable 5 yrs
        eq["Residue"] = residue_level
        eq["Nrate"] = n_rate
        results.append(eq)

    eq_df = pd.DataFrame(results)
    eq_df['n_years'] = eq_df['equilibrium_year'] - df['year'].min()
    eq_df.eval('rate = equilibrium_SOC/n_years', inplace=True)
    orig = df[df['year'].isin(eq_df['equilibrium_year'])][['year', "SOC_balance", 'Residue','Nitrogen']].copy()
    orig.rename(columns={"year": "equilibrium_year", "Nitrogen": "Nrate"}, inplace=True)
    orig['Nrate'] = orig['Nrate'].astype(float)
    orig['Residue'] = orig['Residue'].astype(float)
    adf= orig.merge(eq_df, on =['equilibrium_year', 'Nrate', 'Residue'], how='inner')
    df['SOC_balance'].max()
    print(eq_df)
    from xlwings import view
    # view(single_model.get_simulated_output('carbon_change'))
    from apsimNGpy.core.config import apsim_version

    logger.info(f"Simulated using APSIM version: {apsim_version()}")





