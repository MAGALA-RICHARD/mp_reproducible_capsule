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
    kwargs.setdefault('time_col', 'year')
    kwargs.setdefault('table', data)
    kwargs.setdefault('response', kwargs.get('y'))

    base = Plot(data=data)
    kwargs.setdefault('ylabel', LABELS.get('response'))
    kwargs.setdefault('palette', PlotConstants.PALETTE)
    g = base.plot_mva(**kwargs)
    y = kwargs.get('response')
    g.set_axis_labels("", "")
    xtick_size = PlotConstants.xTICK_SIZE + 1
    ytick_size = PlotConstants.yTICK_SIZE + 2
    ylabel_size = PlotConstants.yLABEL_SIZE + 8
    xlabel_size = PlotConstants.xLABEL_SIZE + 4
    ylabel = LABELS.get(y)
    xlabel = LABELS.get(kwargs.get('time_col', kwargs.get('x')))
    # Enforce tick sizes for every facet (robustly)
    if len(g.axes.flat) > 1:
        for ax in g.axes.flat:
            ax.tick_params(axis='x', which='both', labelsize=xtick_size)
            ax.tick_params(axis='y', which='both', labelsize=ytick_size)
            # Fallback in case styles/backends override tick_params
            plt.setp(ax.get_xticklabels(), fontsize=xtick_size)
            plt.setp(ax.get_yticklabels(), fontsize=ytick_size)
            # Also scale the scientific-notation offset text if present
            ax.yaxis.get_offset_text().set_size(ytick_size)

        # Shared labels
        g.fig.supylabel(ylabel, x=0.002, fontsize=ylabel_size)
        g.fig.supxlabel(xlabel, y=0.002, fontsize=xlabel_size)

        # Legend cleanup + size
        leg = getattr(g, "_legend", None) or getattr(g, "legend", None)
        if leg is not None:
            leg.set_title(None)
            for txt in leg.get_texts():
                txt.set_fontsize(min(xtick_size, ytick_size))
    else:
        plt.xlabel(LABELS.get(kwargs.get('time_col', kwargs.get('x'))), fontsize=PlotConstants.xLABEL_SIZE)
        plt.ylabel(LABELS.get(kwargs.get('response'), kwargs.get('y')), fontsize=PlotConstants.yLABEL_SIZE)
    sns.move_legend(g, loc='upper right', bbox_to_anchor=(0.98, 0.95))
    base.plot.tight_layout()

    fileName = RESULTS / f"_one_file{kwargs.get("time_col")}_{kwargs.get("y")}-{kwargs.get('hue')}-{kwargs.get('response')}-{kwargs.get('height')}{PlotConstants.FILE_EXTENSION}"
    base.plot.savefig(fileName, dpi=PlotConstants.DPI)
    open_file(fileName)
    plt.close()
    base.plot.close()


if __name__ == "__main__":
    from logger import get_logger
    from simulate_scenario_data import table_name
    from change_metrics import compute_last_minus_first_change
    logger = get_logger()
    logger.info(f"plotting: moving averages from scenario simulated data")
    data = read_db(datastore, table_name)

    rdata =data.copy()
    ch = compute_last_minus_first_change(data, grouping=['Nitrogen', "Residue"])

    data['Nitrogen'] = pd.Categorical(data['Nitrogen'], ordered=True)


    data.eval('total_residue_biomass= (total_biomass_Mg - corn_yield_Mg)', inplace=True)
    carbon_frac = 0.4
    data.eval('residue_incorporated = R * total_residue_biomass', inplace=True)
    data.eval('res_ci =residue_incorporated * total_residue_biomass * @carbon_frac', inplace=True)
    # below is cumulative soc_balance
    data.eval('soc_balance = SOC_0_15cm_Mg- 52.260000', inplace=True)  # 52.260000 is the initial SOC
    data.eval('cfe=(soc_balance/res_ci * soc_balance)', inplace=True)


    def optim_soc_biomass(data,opt=180):
        data = data.copy()
        md = data.groupby('Nitrogen', as_index=False)[['total_biomass_Mg', 'corn_yield_Mg']].mean()
        ref = md.loc[md['Nitrogen'].astype(str) == str(opt), ['total_biomass_Mg', 'corn_yield_Mg']].squeeze()
        print(ref)
        return md.assign(
            biomass_diff=md['total_biomass_Mg'] - ref['total_biomass_Mg'],
            yield_diff=md['corn_yield_Mg'] - ref['corn_yield_Mg']
        )
    print(optim_soc_biomass(data))



