import pandas as pd
from correlation import corr_stats
from settings import RESULTS
from apsimNGpy.core_utils.database_utils import read_db_table


def get_data(db, table):
    da = read_db_table(RESULTS / db, table)
    da['N'] = da['Nitrogen'].astype(float)
    da['R'] = da['Residue'].astype(float)

    return da


if __name__ == '__main__':
    df = get_data('single_model.db', 'carbon')
    df.eval('cnr = SurfaceOrganicMatter_Carbon/SurfaceOrganicMatter_Nitrogen', inplace=True)
    out = corr_stats(df, "top_mineralized_N", "SOC_0_15CM", method="pearson", alpha=0.05)
    yd = get_data('single_model.db', 'yield')
    yd.eval('RBiomass =(BBiomass + (total_biomass - grainwt)) * R', inplace=True)
    yd.eval('RB =BBiomass + ((total_biomass - grainwt))', inplace=True)
    ndf = yd.merge(df, how='inner', on = ['N', 'R', 'year', 'SimulationID'])

    def corr_row(g, x, y):
        out = corr_stats(g, x, y, method="pearson", dropna="pairwise")
        return pd.Series({k: out[k] for k in ("coef", "p_value", "n", "ci_low", "ci_high", "r_squared")})


    res = df.groupby(["N"]).apply(corr_row, *("top_mineralized_N", "SOC_0_15CM",)).reset_index()
    res2 = df.groupby(["N"]).apply(corr_row, *('top_carbon_mineralization', "SOC_0_15CM",)).reset_index()
    res3 = ndf.groupby(['N', 'year']).apply(corr_row, *('RBiomass', "cnr",)).reset_index()
    pairs =['top_mineralized_N', "top_carbon_mineralization"]
    gp =ndf.groupby(['R', "N"])[pairs].mean().reset_index().drop_duplicates(['R', 'N'], inplace=False)
    print(res3)
    gp.eval('RN = R*N', inplace=True)
    print(corr_row(gp, *pairs))
    print('Interactions++++++++')
    print(corr_row(gp, 'RB', "top_mineralized_N"))
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    mn= model.fit(gp["top_carbon_mineralization"].to_numpy().reshape(-1,1), gp['top_mineralized_N'].to_numpy().reshape(-1,1))

