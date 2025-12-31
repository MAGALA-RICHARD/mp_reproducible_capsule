import re
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2


def interaction_effect(
        df: pd.DataFrame,
        y: str,
        x1: str,
        x2: str,
        covariates: Optional[List[str]] = None,
        categorical: Optional[List[str]] = None,
        model: str = "ols",
) -> Dict[str, Any]:
    """
    Test the interaction between x1 and x2 on y (with optional covariates).
    Works for OLS and Logit. Returns interaction coefficients and a global test.
    """
    covariates = covariates or []
    categorical = set(categorical or [])

    def term(v: str) -> str:
        return f"C({v})" if v in categorical else v

    x1_t, x2_t = term(x1), term(x2)
    cov_t = [term(c) for c in covariates]

    rhs_reduced = f"{x1_t} + {x2_t}" + ((" + " + " + ".join(cov_t)) if cov_t else "")
    rhs_full = f"{x1_t}*{x2_t}" + ((" + " + " + ".join(cov_t)) if cov_t else "")

    formula_reduced = f"{y} ~ {rhs_reduced}"
    formula_full = f"{y} ~ {rhs_full}"

    # Drop rows with any NA in the variables we actually use
    cols_needed = {y, x1, x2, *covariates}
    d = df.dropna(subset=cols_needed)

    # Helper: pick params that belong to the x1:x2 interaction block
    def is_interaction_param(name: str) -> bool:
        # e.g., 'x1:x2', 'C(x1)[T.a]:x2', 'C(x1)[T.a]:C(x2)[T.b]', etc.
        return (":" in name) and (x1 in name) and (x2 in name)

    if model.lower() == "ols":
        fit_full = smf.ols(formula_full, data=d).fit()
        fit_red = smf.ols(formula_reduced, data=d).fit()

        # Global F-test (full vs reduced)
        F, p, df_num = fit_full.compare_f_test(fit_red)
        df_den = int(fit_full.df_resid)

        # Partial R^2 for the interaction block (based on SSR)
        ssr_full, ssr_red = float(fit_full.ssr), float(fit_red.ssr)
        partial_r2 = max(0.0, (ssr_red - ssr_full) / ssr_red) if ssr_red > 0 else 0.0

        interaction_params = {
            k: (float(fit_full.params[k]), float(fit_full.pvalues[k]))
            for k in fit_full.params.index if is_interaction_param(k)
        }

        return {
            "model": "ols",
            "formula_full": formula_full,
            "formula_reduced": formula_reduced,
            "interaction_params": interaction_params,
            "global_test": {
                "type": "F-test (interaction block)",
                "F": float(F),
                "df_num": int(df_num),
                "df_den": df_den,
                "pvalue": float(p),
            },
            "partial_R2": float(partial_r2),
            "nobs": int(fit_full.nobs),
            "summary": fit_full.summary().as_text(),
        }

    elif model.lower() == "logit":
        fit_full = smf.logit(formula_full, data=d).fit(disp=False)
        fit_red = smf.logit(formula_reduced, data=d).fit(disp=False)

        # Likelihood-ratio test
        ll_full, ll_red = float(fit_full.llf), float(fit_red.llf)
        df_num = int(fit_full.df_model - fit_red.df_model)
        df_num = abs(df_num)
        lr_stat = 2.0 * (ll_full - ll_red)
        p = float(chi2.sf(lr_stat, df_num)) if df_num > 0 else np.nan

        interaction_params = {
            k: (float(fit_full.params[k]), float(fit_full.pvalues[k]))
            for k in fit_full.params.index if is_interaction_param(k)
        }

        # McFadden pseudo-R^2 change (informal effect size)
        r2_full = 1.0 - ll_full / float(fit_full.llnull)
        r2_red = 1.0 - ll_red / float(fit_red.llnull)
        delta_pseudo_r2 = float(r2_full - r2_red)

        return {
            "model": "logit",
            "formula_full": formula_full,
            "formula_reduced": formula_reduced,
            "interaction_params": interaction_params,
            "global_test": {
                "type": "LR-test (interaction block)",
                "chi2": float(lr_stat),
                "df_num": df_num,
                "pvalue": p,
            },
            "delta_pseudo_R2": delta_pseudo_r2,
            "nobs": int(fit_full.nobs),
            "summary": fit_full.summary().as_text(),
        }

    else:
        raise ValueError("model must be 'ols' or 'logit'")


from scipy.stats import ttest_rel
import pandas as pd


def dynamic_paired_ttest(data1, data2, alpha=0.05):
    """
    Performs a paired t-test between two related datasets.

    Parameters
    ----------
    data1, data2 : array-like
        Paired samples (e.g., same plots at two time periods).
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    dict
        Summary including t-statistic, p-value, significance, and interpretation.
    """
    # Drop missing values pairwise
    df = pd.DataFrame({'x': data1, 'y': data2}).dropna()
    t_stat, p_val = ttest_rel(df['x'], df['y'])

    return {
        't_statistic': t_stat,
        'p_value': p_val,
        'significant': p_val < alpha,
        'interpretation': (
            f"Difference is {'significant' if p_val < alpha else 'not significant'} "
            f"(p = {p_val:.4f}, t = {t_stat:.3f})."
        )
    }

from scipy.stats import ttest_ind
import pandas as pd

def dynamic_independent_ttest(group1, group2, alpha=0.05, equal_var=False):
    """
    Performs an independent two-sample t-test between two groups.

    Parameters
    ----------
    group1, group2 : array-like
        Independent samples.
    alpha : float, default 0.05
        Significance level.
    equal_var : bool, default False
        Assume equal variance? (Welch’s correction if False)

    Returns
    -------
    dict
        Summary including t-statistic, p-value, significance, and interpretation.
    """
    # Clean data
    g1, g2 = pd.Series(group1).dropna(), pd.Series(group2).dropna()
    t_stat, p_val = ttest_ind(g1, g2, equal_var=equal_var)

    return {
        't_statistic': t_stat,
        'p_value': p_val,
        'significant': p_val < alpha,
        'interpretation': (
            f"Difference between groups is {'significant' if p_val < alpha else 'not significant'} "
            f"(p = {p_val:.4f}, t = {t_stat:.3f})."
        )
    }

if __name__ == "__main__":
    df = pd.read_csv("ccg.csv")
    for i in ['Single', "Split"]:
        dfs = df[df['timing'] == i]
        ap = interaction_effect(df=dfs, x1='Residue', x2='Nitrogen', y='SOC_0_15CM', )
        print(ap['summary'])
