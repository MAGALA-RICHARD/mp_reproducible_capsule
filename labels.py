"""
labels are created based on the data columns as in merge_tables from data_manager.py.
- If it is rolling mean, then variable followed by the `_roll_mean`.
- If it is changes in the variable from end and start of the simulation label key is
  represented as %Δ and Δ before real column name in the data frame for percentages and absolute values, respectively,
Created on 22/102025
"""

LABELS = {
    'Surf_Org_Carbon_Mg': 'Surface organic carbon (Mg ha⁻¹)',
    'cnr': 'C:N ratio ',
    'SurfaceOrganicMatter_Nitrogen': 'Surface organic matter N (kg ha⁻¹)',
    'top_mineralized_N': 'Mineralized nitrogen (0-15cm, kg ha⁻¹)',
    'Residue': 'Incorporated residue fraction',
    'Nitrogen': 'Nitrogen fertilizer (kg ha⁻¹)',
    'corn_yield_Mg': '101 years average corn grain yield (Mg ha⁻¹)',
    'corn_yield_Mg_roll_mean': 'Corn grain yield (Mg ha⁻¹)',
    'SOC_0_15cm_Mg': 'Soil organic carbon (0-15cm,Mg ha⁻¹)',
    'soc_balance': 'Soil organic carbon balance (0-15cm,Mg ha⁻¹)',
    'SOC_0_15cm_Mg_roll_mean': 'Soil organic carbon (0-15cm,Mg ha⁻¹)',
    'Residue_Biomass_Mg': 'Total residue biomass (Mg ha⁻¹)',
    'Incorporated_Biomass_Mg': 'Incorporated Residue biomass (Mg ha⁻¹)',
    'Below_ground_biomass_Mg': 'Below-ground biomass (Mg ha⁻¹)',
    'total_biomass_Mg': '101 years average biomass (Mg ha⁻¹)',
    'total_biomass': '101 yars average biomass (Mg ha⁻¹)',
    'total_biomass_roll_mean': '101 years average biomass (kg ha⁻¹)',
    'total_biomass_Mg_roll_mean': 'Average biomass (Mg ha⁻¹)',
    'year': 'Time (Years)',
    'top_carbon_mineralization': 'Soil carbon mineralization (Mg ha⁻¹)',
    'ΔSOC_0_15CM': 'Changes in soil carbon (Mg ha⁻¹) after 101 years',
    'ΔSOC_0_15cm_Mg': 'Changes in soil carbon (Mg ha⁻¹) after 101 years',
    'SOC1': 'Changes in soil carbon (Mg ha⁻¹)  after 101 years ',
    'Δcorn_yield_Mg': 'Changes in corn grain yield (Mg ha⁻¹)  after 101 years ',
    '%Δcorn_yield_Mg': "Percentage change in corn grain yield",
    '%ΔResidue_Biomass_Mg': "Percentage change in residue biomass",
    'microbial_carbon': "Microbial biomass carbon (Mg ha⁻¹)",
    'cfe': 'Carbon formation efficiency',
    '%Δtotal_biomass_Mg': "Percentage change in average biomass yield",

}
