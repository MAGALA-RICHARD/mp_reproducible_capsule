import os
import numpy as np
from matplotlib import pyplot as plt
from seaborn import relplot
from nrates import quad_plateau, fit_quadratic_plateau, fit_quadratic, eonr_quadratic, eonr_quadratic_plateau
import pandas as pd
from settings import RESULTS

db = RESULTS / 'gd.db'
from apsimNGpy.core_utils.database_utils import read_db_table

dff = read_db_table(db, report_name='yield')
dff['Yield'] = dff['grainwt']
df = dff[dff['Residue'] == '0.25'].copy()

df['Yield'] = df['grainwt']
df['Nitrogen'] = df['Nitrogen'].astype(float)
df = df.groupby(['Nitrogen'])['Yield'].mean().reset_index()

# Fit both models
q = fit_quadratic(df, "Nitrogen", "Yield")
qp = fit_quadratic_plateau(df, "Nitrogen", "Yield")

print("Quadratic AONR:", q["AONR"])
print("Quadratic-Plateau AONR:", qp["AONR"])

# Prices for EONR (edit to your market; units must match your dataset)
price_grain = 0.18  # $ per kg grain (≈ $4/bu corn -> adjust as needed)
price_N = 2.10  # $ per kg N

N_min, N_max = df["Nitrogen"].min(), df["Nitrogen"].max()

print("Quadratic EONR:",
      eonr_quadratic(q["params"], price_N, price_grain, N_min, N_max))

print("Quadratic-Plateau EONR:",
      eonr_quadratic_plateau(qp["params"], price_N, price_grain, N_min, N_max))

relplot(x='Nitrogen', y='grainwt', data=read_db_table(db, 'yield'), kind='line', col_wrap=2,
        height=8, errorbar=None, col='Residue')
plt.savefig('Nitrogen.png')
os.startfile('Nitrogen.png')

# ______________ carbon______________________
dff = read_db_table(db, report_name='carbon')

df = dff[dff['Residue'] == '0.25'].copy()

df['Nitrogen'] = df['Nitrogen'].astype(float)
df = df.groupby(['Nitrogen'])['SOC_0_15CM'].mean().reset_index()

# Fit both models
q = fit_quadratic(df, "Nitrogen", 'SOC_0_15CM')
qp = fit_quadratic_plateau(df, "Nitrogen", 'SOC_0_15CM')

print("Quadratic AONR:", q["AONR"])
print("Quadratic-Plateau AONR:", qp["AONR"])

# Prices for EONR (edit to your market; units must match your dataset)
price_grain = 0.18  # $ per kg grain (≈ $4/bu corn -> adjust as needed)
price_N = 2.10  # $ per kg N

N_min, N_max = df["Nitrogen"].min(), df["Nitrogen"].max()

print("Quadratic EONR:",
      eonr_quadratic(q["params"], price_N, price_grain, N_min, N_max))

print("Quadratic-Plateau EONR:",
      eonr_quadratic_plateau(qp["params"], price_N, price_grain, N_min, N_max))

relplot(x='Nitrogen', y='grainwt', data=read_db_table(db, 'yield'), kind='line', col_wrap=2,
        height=8, errorbar=None, col='Residue')
plt.savefig('Nitrogen.png')
os.startfile('Nitrogen.png')
