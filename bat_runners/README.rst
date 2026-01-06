This directory contains Windows .bat runner scripts. Simply double-click on any of the .bat files to start the execution automatically.

OR

From a command-line interface (CMD or PowerShell), navigate to the **this directory** and run the following commands:

.. code-block:: bat

   start evaluation.bat
   start simulate_qp.bat
   start fit_qp_model.bat
   start scenarios.bat

Tasks in .bat script:
======================

- ``evaluation.bat``
  Generates results for APSIM model evaluation and validation.

- ``simulate_qp.bat``
  Generates large simulation datasets used for fitting quadratic response models.

- ``fit_qp_model.bat``
  Fits quadratic models for soil organic carbon and corn grain yield and calculates the economic optimum nitrogen rate (EONR).

- ``scenarios.bat``
  Runs scenario analyses comparing four nitrogen application rates and residue management levels.
  The four nitrogen rates are determined in the previous steps; see the manuscript for details.

The first script will create a virtual environment, and also install all the requirements from the requirements.txt.
That being said, the preceding script will execute right away.

At the end of the simulation, a dialog prompt appears asking whether you want to close the window. This is intended to give you sufficient time to read the results printed in the console.

Running the scripts in this folder has additional setup requirements: you must provide the accompanying
`.env` file, manifest.yml and the `requirements.txt` included with this directory. If execution fails or
the environment cannot be resolved, revert to running the scripts from the project root, where the default
configuration is expected to work.

.. leave_comment
  passed all unit tests