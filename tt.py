from apsimNGpy.core import config
print(config.get_apsim_bin_path())

from apsimNGpy.core.apsim import ApsimModel, Models
# Option 1: Load default maize simulation
model = ApsimModel('Maize') # case sensitive
# Run the simulation
model.run(report_name='Report')

# Retrieve and save the results
df = model.results
df.to_csv('apsim_df_res.csv')  # Save the results to a CSV file
print(model.results)  # Print all DataFrames in the storage domain

model.save('./edited_maize_model.apsimx')
model.inspect_model('Models.Manager', fullpath=True)
model.inspect_model(Models.Clock)
model.inspect_model(Models.Core.IPlant)

model.inspect_model_parameters_by_path('.Simulations.Simulation.Field.Sow using a variable rule')