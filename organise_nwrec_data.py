from pathlib import Path
import pandas as pd
from apsimNGpy.core.apsim import ApsimModel
from xlwings import view
from sqlalchemy import create_engine
from apsimNGpy.exceptions import ApsimRuntimeError

SITE = 'NWREC'
dir_data = Path('./yield_DATA')
lonlat = '(-90.7272222, 40.9305556)'
lonlat_sable = eval(lonlat)
lonlat_muskatine_and_osco = (-90.7281, 40.9317)
datastore = str(dir_data / "yield_DATA.db")

con = create_engine(f'sqlite:///{datastore}')
out_apsimx = dir_data / 'nwrec.apsimx'
yield_table = 'nwrec'
carbon_table = 'carbon'
from apsimNGpy.core_utils.database_utils import read_db_table,get_db_table_names
print(get_db_table_names(datastore))
import sys
sys.exit()
if __name__ == '__main__':
    from nwrec_soil_tools import exp_scale, exp_bridge, exp_scale_df

    d_yield = pd.read_csv(dir_data / 'nwrecoObservedYieldData.csv')
    col = "Corn grain yield at 15.5% MB"

    d_yield[col] = pd.to_numeric(d_yield[col], errors="coerce")
    d_yield = d_yield.dropna(subset=[col])
    d_yield.rename(columns={col: "corn_grain_yield_kg"}, inplace=True)
    d_yield.eval('grain_yield_Mg =corn_grain_yield_kg/1000', inplace=True)
    d_yield.eval('lon_lat = @lonlat', inplace=True)
    dtypes = d_yield.dtypes
    d_yield.to_sql(yield_table, con, if_exists='replace', index=False)

    soils = pd.read_csv(dir_data / 'nwerecSoilProperties.csv')
    soils.rename(columns={'Water retention at 0.33 bar': 'DUL', 'Water retention at 0 bar': 'SAT',
                          'Water retention at 15 bar': 'LL15'}, inplace=True)
    # water related data
    ############################################
    water_columns = ['SAT', "DUL", "LL15"]
    water = soils.copy()
    water[water_columns] = soils[water_columns].apply(pd.to_numeric, errors='coerce')
    water = water.dropna(subset=water_columns)
    assert water['Soil sampling depth'].unique().shape[0] == 2, "is this a new data? sampling depth exceeds 2"
    dg = water.groupby(['uniqueid', 'plotid'])[['SAT', 'DUL', 'LL15']].mean()
    scaled = exp_scale_df(dg[['SAT', 'DUL', 'LL15']], steps=10, rate=-0.06, )
    scaled['AirDry'] = scaled['LL15'] * 0.5
    water_scaled = scaled  # .reset_index().set_index(['uniqueid', 'plotid'])
    # Bulk density
    #######################################
    bd_cols = 'Bulk density'
    bd = soils.copy()
    bd[bd_cols] = soils[bd_cols].apply(pd.to_numeric, errors='coerce')
    bd = bd.dropna(subset=bd_cols)
    bd = bd[bd['year'] == 2011]
    bd['Soil sampling depth'] = bd['Soil sampling depth'].replace({'10 to 20': '0 to 10'})
    bd = bd[bd['Soil sampling depth'] == '0 to 10']
    bg = bd.groupby(['uniqueid', 'plotid'])[[bd_cols]].mean()
    bg.rename(columns={'Bulk density': 'BD'}, inplace=True)
    bg = exp_scale_df(bg[['BD']], steps=10, rate=-0.06, )
    bg_scaled = bg  # .reset_index().set_index(['uniqueid', 'plotid'])
    # Ph
    #####################################################
    ph_cols = "Soil pH"
    ph = soils.copy()
    ph[ph_cols] = soils[ph_cols].apply(pd.to_numeric, errors='coerce')
    ph = ph.dropna(subset=ph_cols)
    ph = ph[(ph['year'] == 2011) & (ph['Soil sampling depth'] == '0 to 10')]
    ph = ph.groupby(['uniqueid', 'plotid'])[[ph_cols]].mean()
    ph_scaled = exp_scale_df(ph[ph_cols], steps=10, rate=-0.03, )
    # texture related
    ############################################
    texture_cols = ['Percent sand', 'Percent silt', 'Percent clay']

    texture = soils.copy()
    texture[texture_cols] = soils[texture_cols].apply(pd.to_numeric, errors='coerce')
    td = texture.dropna(subset=texture_cols)
    td = td[td['year'] == 2011]
    td['Soil sampling depth'] = td['Soil sampling depth'].replace({'10 to 20': '0 to 10'})
    td = td[td['Soil sampling depth'] == '0 to 10']
    td = td.groupby(['uniqueid', 'plotid'])[texture_cols].mean()
    td_scaled = exp_scale_df(td[texture_cols], steps=10, rate=-0.03, )

    td_scaled.rename(columns={'Percent sand': 'ParticleSizeSand',
                              'Percent silt': 'ParticleSizeSilt', 'Percent clay': 'ParticleSizeClay'}, inplace=True)
    txd_scaled = td_scaled  # .reset_index().set_index(['uniqueid', 'plotid'])

    # model.get_soil_from_web(lonlat=(-90.7281, 40.9317), soil_series='Osco')
    plotids = pd.read_csv(dir_data / 'plotids.csv')
    pp = plotids[plotids['tillage']=='TIL2'].copy()
    crops = pp[["2011crop", "2012crop", "2013crop", "2014crop", "2015crop"]]
    pp["crop_seq"] = crops.apply(tuple, axis=1)  # tuple is hashable than a list
    cont_corn = pp[pp["crop_seq"] == ('Corn', 'Corn', 'Corn', 'Corn', 'Corn')].copy()
    cont_corn['unique_plots'] = cont_corn[['uniqueid', 'plotid']].apply(tuple, axis=1)
    if bg_scaled.index.names == water_scaled.index.names == txd_scaled.index.names:
        df = bg_scaled.join(water_scaled).join(txd_scaled)
    else:
        raise ValueError('data do not have the same index, did you somehow change the code above')
    rdf = df.reset_index().set_index(['uniqueid', 'plotid'])
    cont_corn.set_index(['uniqueid', 'plotid'], inplace=True)
    wk = list(set(cont_corn['unique_plots'].index.tolist()))
    ser = list(set(cont_corn['soilseriesname1']))
    rdf = rdf.sort_index(level=['uniqueid', 'plotid'])
    base = ApsimModel('Maize', out_path='b.apsimx')
    comps = []
    for uin_plot in wk:
        mod = ApsimModel(dir_data / "base.apsimx", dir_data / 'base_copy.apsimx')
        comps.append(uin_plot)
        for serie in ser:
            print('Working on: ', uin_plot)
            try:
                plot_data = rdf.loc[uin_plot, :]

            except KeyError as ke:
                print('Skipping: ', uin_plot)
                continue
            phys = {c: plot_data[c].tolist() for c in
                    ['BD', 'SAT', 'DUL', 'LL15', 'AirDry', 'ParticleSizeSand', 'ParticleSizeClay', 'ParticleSizeSilt']}
            if serie.strip().lower() == 'muscatune' or serie.strip().lower() == 'osco':
                lonlat_a = lonlat_muskatine_and_osco

            else:
                lonlat_a = lonlat_sable
            mod.get_soil_from_web(lonlat=lonlat_a, soil_series=serie.strip(), thinnest_layer=200)
            sat = phys['SAT']
            DUL = phys['DUL']
            for enum, (s, d) in enumerate(zip(sat, DUL)):
                if s > 0.506:
                    st = 0.504
                else:
                    st = s
                if st < d:
                    dif = d - s
                    ss = s * 1.05
                    print(s, ss)
                    phys['SAT'][enum] = ss
            mod.edit_model(model_type='Physical', model_name='Physical', **phys)
            mod.edit_model(model_type="Models.Soils.Water", model_name='Water', InitialValues=phys['DUL'])

            mod.adjust_dul()
            edited = dir_data / 'edited'
            edited.mkdir(exist_ok=True)
            plotid = str(uin_plot[1])
            sim_names = [b.Name for b in base.simulations]
            if not plotid in sim_names:
                mod.simulations[0].Name = plotid
                base.Simulations.Children.Add(mod.simulations[0])
                mod.save(file_name=edited / f'{serie.strip()}_{uin_plot[0]}_{uin_plot[1]}.apsimx')
        base.save(dir_data / 'nwrec1.apsimx', reload=True)
