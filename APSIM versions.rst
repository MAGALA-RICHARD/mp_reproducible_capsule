APSIM Bin Path Management
==========================

APSIM simulations in this project rely on explicit management of the APSIM binary path to ensure
reproducibility across analyses. The default APSIM version used throughout the project, and
included in the project root, is **APSIM2025.8.7844.0**. However, selected components of the
workflow require the use of additional APSIM versions, specifically **APSIM2024.5.7493.0**
and **APSIM2025.12.7939.0**, which are pinned and managed through the ``apsimNGpy`` version manager.

The use of multiple APSIM versions is necessary to maintain backward compatibility with earlier
simulation configurations and model components that are not fully supported under a single APSIM release.
As a result, reproducing all results reported in this study requires that **all three APSIM versions
be installed and accessible** on the host system.

The ``apsimNGpy`` version manager provides programmatic control over APSIM binary paths, allowing
individual simulations or workflow stages to be executed under the appropriate APSIM version without
manual intervention. This approach ensures consistent model execution while accommodating version-specific
behavior across the simulation pipeline.

Environment File Configuration
------------------------------

In the ``.env`` file, each APSIM version is represented by a numeric key that maps to
a specific APSIM installation. For example:

::

   7844 = APSIM2025.8.7844.0
   7493 = APSIM2024.5.7493.0
   7939 = APSIM2025.12.7939.0

The values on the right-hand side may be specified either as version identifiers or
as **absolute paths** pointing directly to the corresponding APSIM installation directories.

The ``.env`` file is expected to reside in the **root directory of the project** so that it
can be automatically discovered and loaded by ``apsimNGpy`` during execution.


Prerequisites
-------------
- Windows 10 or 11 (recommended windows 11)
- Python 3.11+ (recommended:  3.12.7)
- Git (to clone this repository)
- .NET 8.0 (install from https://dotnet.microsoft.com/en-us/download/dotnet/8.0)  see the requirements for each platform https://apsimnextgeneration.netlify.app/install/macos/

Accessing the project
-----------------------
Open **Command Prompt** and run:

   .. code-block:: bat

      git clone https://github.com/MAGALA-RICHARD/mp_reproducible_capsule.git
      cd mp_reproducible_capsule

Populate or create .env file with the three APSIM versions specified above.

Code Execution
==============
1. The quickest and ingenious way is to run .bat objects Windows(only)

**Operating System:** Windows 10 or later

To execute the workflows in this project on Windows, a set of provided ``.bat`` scripts can be used. From a command-line interface (CMD or PowerShell), navigate to the **root directory of the project** and run the following commands:

.. code-block:: bat

   start evaluation.bat
   start simulate_qp.bat
   start fit_qp_model.bat
   start scenarios.bat

The scripts perform the following tasks:

- ``evaluation.bat``
  Generates results for APSIM model evaluation and validation.

- ``simulate_qp.bat``
  Generates large simulation datasets used for fitting quadratic response models.

- ``fit_qp_model.bat``
  Fits quadratic models for soil organic carbon and corn grain yield and calculates the economic optimum nitrogen rate (EONR).

- ``scenarios.bat``
  Runs scenario analyses comparing four nitrogen application rates and residue management levels.
  The four nitrogen rates are determined in the previous step; see the manuscript for details.

Alternatively, each ``.bat`` script can be executed by **double-clicking** it from the project root directory.


Other Operating Systems
-----------------------

For non-Windows operating systems (Linux or macOS),
the underlying Python scripts can be executed directly from the command line following
the same workflow order described above.

Manual Setup (command line)
---------------------------

2. Create and activate a virtual environment:

   .. code-block:: bat

      python -m venv .venv
      REM Activate it:
      REM Windows (PowerShell):
      .\.venv\Scripts\Activate.ps1
      REM Windows (CMD):
      call .venv\Scripts\activate.bat

3. (Optional) Upgrade packaging tools:

   .. code-block:: bat

      python -m pip install --upgrade pip setuptools wheel

4. Install pinned dependencies:

   .. code-block:: bat

      pip install -r requirements.txt

Using uv (optional)
-------------------
If you prefer the faster ``uv`` installer:

.. code-block:: bat

   python -m pip install -U uv
   uv pip install -r requirements.txt

5. Run the example listings (all scripts live in the ``reproducible`` folder):

   .. code-block:: bat

      python model_evaluation.py
      python apsim_validation_nwrec.py
      python simulate_quadratic_fit_data.py
      python qp_model_soc_balance.py
      python qp_model_yield.py
      python qp_model_EONR.py
      python simulate_scenario_data.py

.. note::

   Some version of python may require specifying python3 instead of python but even py is adequate


Outputs
-------

All numerical outputs are written to the ``results/`` directory located in the project root.
Figures are saved to either the ``plots/`` or ``figures/`` directories. In most cases, figures
are also displayed automatically for immediate visualization when the scripts are executed.
