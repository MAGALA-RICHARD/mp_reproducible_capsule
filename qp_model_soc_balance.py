"""
Fits a quadratic model to the average corn grain yield
Author: Richard Magala
Created on monday Oct 27 2025:
"""
import math
from pathlib import Path
import pandas as pd
import seaborn as sns
from data_manager import read_db
from labels import LABELS
from logger import get_logger
from simulate_quadratic_fit_data import datastore, table_name
from utils import  load_manifest
cfg = load_manifest()

sns.set_style("darkgrid", rc={"font.family": "DejaVu Sans"})
# pd.plotting.register_matplotlib_converters()
MODULE_NAME = Path(__file__).stem
from qp_model_utils import fit_quadratic_plateau, consistent_optimum, x_zero, plot_fitted_data
from scipy.optimize import least_squares
from numpy.linalg import inv
from scipy.stats import t as tdist
import numpy as np

script_name = Path(__file__).stem
logger = get_logger(name=f'{script_name}')

logger.info(
    "\n finding the optimum N fertilizer for SOC sequestration please wait \n ===========================================================================")

if __name__ == '__main__':
    from utils import RESULTS, open_file
    from change_metrics import compute_last_minus_first_change

    RESPONSE = 'ΔSOC_0_15cm_Mg'
    opt_params = []
    dfm = read_db(datastore, table_name)
    dy = dfm.copy()
    dfm['Residue'] = pd.Categorical(dfm['Residue'], ordered=True)
    y_min = dfm[dfm['year'] == dfm['year'].min()]['SOC1'].iloc[0]
    dfm['SOC1'] = dfm['SOC_0_15CM'] - float(y_min)
    soc_balance = compute_last_minus_first_change(dfm, grouping=['Residue', 'Nitrogen']).sort_values(
        by='ΔSOC_0_15cm_Mg', ascending=False)
    soc_balance['N'] = soc_balance['Nitrogen'].astype('float')
    soc_balance['SOC1'] = soc_balance['ΔSOC_0_15cm_Mg']
    # x = Nitrogen (kg/ha), y = SOC (e.g., Mg C/ha)
    predicted_observed = []

    for residue in sorted(soc_balance.Residue.unique()):
        df = soc_balance[soc_balance['Residue'] == residue]  # .groupby('N')['SOC1'].mean().reset_index()
        res = fit_quadratic_plateau(x=df['N'], y=df[RESPONSE])

        print(res['params'])  # a, b, c, x_break, plateau
        print(res['metrics'])  # R2, R2_adj, AIC, BIC, RMSE, etc.

        # Predict and (optional) plot
        xg = np.linspace(df['N'].min(), df['N'].max(), 250)
        yg = res['predict'](xg)

        import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

        x_obs = df['N'].to_numpy(float)
        y_obs = df['SOC1'].to_numpy(float)
        y_hat = res['predict'](x_obs)
        # y_hat = df['SOC1'].to_numpy(float)

        d = pd.DataFrame({'x': x_obs, 'Actual': y_obs, 'Predicted': y_hat})
        d['Residue'] = residue

        ax = sns.scatterplot(data=d, x='x', y='Actual', s=18, alpha=0.6, label='APSIM', color='red')
        sns.lineplot(data=d.sort_values('x'), x='x', y='Predicted', linewidth=2, label='Fitted line', ax=ax)

        ax.set_xlabel(LABELS.get('Nitrogen'), fontsize=cfg['plots']['x_fontsize']),
        ax.set_ylabel(LABELS.get('SOC1'), fontsize=cfg['plots']['y_fontsize'])
        ax.tick_params(axis='y', labelsize=cfg['plots']['x_ticks_fontsize'])
        ax.tick_params(axis='y', labelsize=cfg['plots']['x_ticks_fontsize'])
        ax.legend(loc='best', fontsize=16)
        plt.ylabel(LABELS.get('SOC1'), fontsize=22)
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
            0.2, 0.45, txt,
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
    logger.info("\n%s", dp.to_csv(index=False))
    print(dp)
    print("fitted values summary")
    print(dp.mean(numeric_only=True))
    from plotting import relplot

    soc_balance['Nitrogen'] = soc_balance['Nitrogen'].astype('float')
    yield_change = compute_last_minus_first_change(data=dy, col='corn_yield_Mg', grouping=['Residue', 'Nitrogen'])
    soc_balance["Residue"] = (100 * soc_balance["Residue"].astype(float)).astype(int)
    # soc_balance['Residue'] = (soc_balance['Residue'] * 100).astype(float).astype(int)
    relplot(data=soc_balance, show=True, x='Nitrogen', y='ΔSOC_0_15cm_Mg', hue='Residue', kind='line',
            add_scatter=False)
    yield_change['Nitrogen'] = yield_change['Nitrogen'].astype('float')
    dy['Nitrogen'] = dy['Nitrogen'].astype('float')
    relplot(data=dy, show=True, filename=RESULTS / f'{MODULE_NAME}_total_biomass_Mg.svg', x='Nitrogen',
            y='total_biomass_Mg', hue='Residue', kind='line', errorbar=None,
            add_scatter=False)
    cn = dy.groupby(['Residue', "R", 'Nitrogen'])['cnr'].mean().reset_index()
    cn['Residue'] = (cn['R'].astype(float) * 100).astype(int)
    relplot(data=cn, show=True, x='Nitrogen', y='cnr', hue='Residue', kind='line', errorbar=None,
            add_scatter=False, estimator='sum')

    soc_balance[['Nitrogen', 'Residue']] = soc_balance[['Nitrogen', 'Residue']].astype(float)
    soc_balance.eval('nr =Nitrogen * Residue', inplace=True)
    soc_balance[['Nitrogen', 'Residue', 'nr', 'ΔSOC_0_15cm_Mg']].corr(method='pearson')
    dy.eval('nr =R *N', inplace=True)
    from scipy.stats import pearsonr

    df = dy[['R', 'N', 'nr', 'cnr']]

    results = []
    cols = df.columns

    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            if i < j:
                r, p = pearsonr(df[col1], df[col2])
                results.append({'var1': col1, 'var2': col2, 'r': r, 'p': p})

    corr_table = pd.DataFrame(results)
    # print(corr_table)
    ##########################################################################################################
    # trying to plot it on the same plane

    # ------------------ prep data ---------------------------------------------
    po = pd.concat(predicted_observed, ignore_index=True).copy()

    # create clean residue % label
    po.eval('R = Residue', inplace=True)
    po["ResiduePct"] = (po["R"].astype(float) * 100).round().astype(int)
    po["Residue incorporation"] = po["ResiduePct"].astype(str) + "%"
    fg_name = RESULTS / f"{MODULE_NAME}_fitted_soc_balance.svg"
    plot_fitted_data(po, ylabel="Soil organic carbon balance (Mg ha⁻¹) after 101 years", filename=fg_name)
    open_file(fg_name)
    average_n_threshold= dp.mean(numeric_only=True)

