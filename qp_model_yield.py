"""
Fits a quadratic model to the average corn grain yield
Author: Richard Magala
Created on Sunday Oct 26 2024:
"""
import math
from pathlib import Path
import pandas as pd
import seaborn as sns
from data_manager import read_db
from labels import LABELS
from logger import get_logger
from simulate_quadratic_fit_data import datastore, table_name
from plotting import relplot
from qp_model_utils import fit_quadratic_plateau, consistent_optimum, x_zero, plot_fitted_data
import numpy as np

sns.set_style("darkgrid", rc={"font.family": "DejaVu Sans"})

MODULE_NAME = Path(__file__).stem
script_name = Path(__file__).stem
logger = get_logger(name=f'{script_name}')

ylabel = "101 years average corn grain yield (Mg ha⁻¹)"
logger.info(
    "finding the optimum N fertilizer for SOC sequestration\n ===========================================================================")

if __name__ == '__main__':
    from utils import RESULTS, open_file
    from change_metrics import compute_last_minus_first_change, mean_diff_between_windows

    RESPONSE = 'Δcorn_yield_Mg'
    opt_params = []
    dfm = read_db(datastore, table_name)
    dy = dfm.copy()
    dfm['Residue'] = pd.Categorical(dfm['Residue'], ordered=True)
    y_min = dfm[dfm['year'] == dfm['year'].min()]['SOC1'].iloc[0]
    dfm['SOC1'] = dfm['SOC_0_15CM'] - float(y_min)
    predicted_observed = []
    mean_yield = dfm.groupby(["Residue", 'N', 'Nitrogen'])['corn_yield_Mg'].mean().reset_index()
    mean_yield[RESPONSE] = mean_yield['corn_yield_Mg']
    # changes in yield
    #############################################################
    yield_dif = mean_diff_between_windows(dfm, grouping=['Residue', "N", 'Nitrogen'], col="corn_yield_Mg", sample=False,
                                          sample_window=50, sample_size=25,
                                          first_window=(1904, 1911), last_window=(1997, 2005))
    yield_dif['Nitrogen'] = yield_dif['N'].astype(float)
    yield_dif['Residue'] = (yield_dif['Residue'].astype(float) * 100).astype(int)
    # plot for yield difference
    # ++++++++++++++++++++++++++++++++++++++
    pdata_y = yield_dif[yield_dif['Nitrogen'] > 139]
    relplot(data=pdata_y, show=True, filename=f"{RESULTS}/{MODULE_NAME}_average_corn_grain_yield_change.svg",
            x='Nitrogen', y='%Δcorn_yield_Mg', hue='Residue', kind='line',
            add_scatter=False)

    biomass_dif = mean_diff_between_windows(dfm, grouping=['Residue', "N", 'Nitrogen'], col="total_biomass_Mg",
                                            sample=False,
                                            sample_window=50, sample_size=25,
                                            first_window=(1904, 1911), last_window=(1997, 2005))
    biomass_dif['Nitrogen'] = biomass_dif['N'].astype(float)
    biomass_dif['Residue'] = (biomass_dif['Residue'].astype(float) * 100).astype(int)
    # plot for yield difference
    # ++++++++++++++++++++++++++++++++++++++
    pdata_b = biomass_dif[biomass_dif['Nitrogen'] > 139]
    relplot(data=pdata_b, filename=f"{RESULTS}/{MODULE_NAME}average_biomass_change.svg", show=True, x='Nitrogen',
            y='%Δtotal_biomass_Mg', hue='Residue', kind='line',
            add_scatter=False)

    # Loop to fit a quadratic/plateau model to each residue retention level
    #####################################################################
    for residue in sorted(mean_yield.Residue.unique()):
        df = mean_yield[mean_yield['Residue'] == residue]  # .groupby('N')['SOC1'].mean().reset_index()
        res = fit_quadratic_plateau(x=df['N'], y=df[RESPONSE])

        print(res['params'])  # a, b, c, x_break, plateau
        print(res['metrics'])  # R2, R2_adj, AIC, BIC, RMSE, etc.

        # Predict and (optional) plot
        xg = np.linspace(df['N'].min(), df['N'].max(), 200)
        yg = res['predict'](xg)

        import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

        x_obs = df['N'].to_numpy(float)
        y_obs = df[RESPONSE].to_numpy(float)
        y_hat = res['predict'](x_obs)
        # y_hat = df['SOC1'].to_numpy(float)

        d = pd.DataFrame({'x': x_obs, 'Actual': y_obs, 'Predicted': y_hat})
        d['Residue'] = residue

        ax = sns.scatterplot(data=d, x='x', y='Actual', s=18, alpha=0.6, label='APSIM', color='red')
        sns.lineplot(data=d.sort_values('x'), x='x', y='Predicted', linewidth=2, label='Fitted line', ax=ax)

        ax.set_xlabel(LABELS.get('Nitrogen'))
        ax.set_ylabel(LABELS.get('SOC1'))
        sns.despine()
        # plt.title(residue)
        # --- Add QP formula & fit stats (ASCII-safe, no LaTeX needed) ---
        a = float(res["params"]["a"])
        b = float(res["params"]["b"])
        c = float(res["params"]["c"])
        x0 = float(res["params"]["x_break"])
        plat = float(res["params"]["plateau"])
        r2 = float(res["metrics"]["R2"])
        rmse = float(res["metrics"]["RMSE"])
        print(f"Residue_Nthrehold `{residue}`:{x_zero(a, b, c)}")


        # Snap tiny values to zero to avoid ugly -0.000 strings
        def z(v, eps=1e-12):
            return 0.0 if abs(v) < eps else v


        a, b, c = z(a), z(b), z(c)

        try:
            test = "x\u00b2"
            assert ax.figure.canvas.get_renderer()  # forces font load in some backends
            use_sup2 = True
        except Exception:
            use_sup2 = False

        line1 = (
                f"y = {a:.3g}{b:+.3g}N{c:+.3g}" + ("N\u00b2" if use_sup2 else "N^2")
        )
        rsq = ("R\u00b2" if use_sup2 else "R^2")
        use_sub = True
        xo = ("N\u2080" if use_sub else "N_0")
        line2 = f"{xo} = {x0:.3g}; plateau = {plat:.3g} \n{rsq} = {r2:.3f}; RMSE = {rmse:.3g}"
        txt = line1 + "\n" + line2 + "\n"  # + f"for {int(float(residue) * 100)}% residue incorporation"
        plt.legend(frameon=False)
        bg = ax.get_facecolor()
        ax.text(
            0.1, 0.6, txt,
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(facecolor=bg, alpha=0.75, edgecolor="none", pad=6),
            fontsize=12, family="DejaVu Sans Mono"  # mono keeps signs/spacing tidy
        )

        plt.tight_layout()

        fname = RESULTS / f'{residue}-qp.png'
        plt.savefig(fname, dpi=600)
        # open_file(fname)
        plt.close()
        argmax = int(np.argmax(y_hat))
        print(f"{residue}: {x_obs[argmax]}")
        qop = consistent_optimum(res, x_obs, model='qp')
        qop['Residue'] = residue
        qop['N Threshold'] = x_zero(a, b, c)
        logger.info(qop)
        opt_params.append(qop)
        d['txt'] = txt
        predicted_observed.append(d)
    dp = pd.DataFrame(opt_params)
    plt.close()
    logger.info("\n%s", dp.to_csv(index=False))
    print(dp)

    mean_yield['Nitrogen'] = mean_yield['Nitrogen'].astype('float')
    yield_change = compute_last_minus_first_change(data=dy, col='corn_yield_Mg', grouping=['Residue', 'Nitrogen'])
    mean_yield["Residue"] = (100 * mean_yield["Residue"].astype(float)).astype(int)
    # soc_balance['Residue'] = (soc_balance['Residue'] * 100).astype(float).astype(int)
    relplot(data=mean_yield, show=True, x='Nitrogen', y=RESPONSE, hue='Residue', kind='line',
            add_scatter=False)
    yield_change['Nitrogen'] = yield_change['Nitrogen'].astype('float')
    dy['Nitrogen'] = dy['Nitrogen'].astype('float')

    cn = dy.groupby(['Residue', "R", 'Nitrogen'])['cnr'].mean().reset_index()
    cn['Residue'] = (cn['R'].astype(float) * 100).astype(int)

    mean_yield[['Nitrogen', 'Residue']] = mean_yield[['Nitrogen', 'Residue']].astype(float)
    mean_yield.eval('nr =Nitrogen * Residue', inplace=True)

    ##########################################################################################################
    # trying to plot it on the same plane
    plt.close()
    # --- prep data ---
    # === prep data ===
    po = pd.concat(predicted_observed, ignore_index=True).copy()

    # create clean residue % label
    po.eval('R = Residue', inplace=True)
    po["ResiduePct"] = (po["R"].astype(float) * 100).round().astype(int)
    po["Residue incorporation"] = po["ResiduePct"].astype(str) + "%"
    fg_name = RESULTS / f"{MODULE_NAME}_fitted_avg_yield.svg"
    plot_fitted_data(po, ylabel="101 years average corn grain yield (Mg ha⁻¹)", filename=fg_name)
    open_file(fg_name)
    ydf = dfm.copy()


    def plot_yield():
        ydf = read_db(datastore, table_name)
        window = 5
        fist_window = 1904, 1904 + window
        last_window = 2005 - window, 2005
        print(ydf.Residue.unique())
        ydf['Nitrogen'] = ydf['N'].astype(float)
        ydf = ydf[ydf['Nitrogen']>=0]
        ydf['Residue'] = (ydf['R'] * 100).astype(int)
        relplot(data=ydf, filename=f"{RESULTS}/{MODULE_NAME}_average_yield.png", show=True, x='Nitrogen', errorbar=None,
                y='corn_yield_Mg', hue='Residue', kind='line',
                add_scatter=False)
        relplot(data=ydf, filename=f"{RESULTS}/{MODULE_NAME}_average_biomass.png", show=True, x='Nitrogen',  errorbar=None,
                y='total_biomass_Mg', hue='Residue', kind='line',
                add_scatter=False)

        biomass_dif = mean_diff_between_windows(ydf, grouping=['Residue', "N", 'Nitrogen'], col="total_biomass_Mg",
                                                sample=False,
                                                sample_window=20, sample_size=15,
                                                first_window=fist_window, last_window=last_window)
        biomass_dif['Nitrogen'] = biomass_dif['N'].astype(float)
        adf = biomass_dif#biomass_dif[biomass_dif['%Δtotal_biomass_Mg']>= 0]
        relplot(adf, filename=f"{RESULTS}/{MODULE_NAME}average_biomass_change.png", show=True, x='Nitrogen',
                y='%Δtotal_biomass_Mg', hue='Residue', kind='line',
                add_scatter=False)
        yield_dif = mean_diff_between_windows(ydf, grouping=['Residue', "N", 'Nitrogen'], col="corn_yield_Mg",
                                                sample=False,
                                                sample_window=20, sample_size=15,
                                                first_window=(1904, 1911), last_window=(1997, 2005))
        yield_dif['Nitrogen'] = biomass_dif['N'].astype(float)
        relplot(data=yield_dif, filename=f"{RESULTS}/{MODULE_NAME}average_corn_yield_change.png", show=True,
                x='Nitrogen',
                y='%Δcorn_yield_Mg', hue='Residue', kind='line',
                add_scatter=False)
    plot_yield()
