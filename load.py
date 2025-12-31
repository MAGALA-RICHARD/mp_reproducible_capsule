import pythonnet

# Load .NET runtime (CoreCLR). On success, this returns immediately on later calls.
pythonnet.load("coreclr")

# Sanity checks
info = pythonnet.get_runtime_info()
print(info)  # e.g., RuntimeInfo(kind='CoreCLR', version='<undefined>', initialized=True, shutdown=False)

# Get actual .NET version from the runtime
import System
from System.Runtime.InteropServices import RuntimeInformation
print(System.Environment.Version)              # e.g. 8.0.x
print(RuntimeInformation.FrameworkDescription) # e.g. ".NET 8.0.x"

