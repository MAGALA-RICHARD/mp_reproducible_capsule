from apsimNGpy.core.apsim import ApsimModel
from pathlib import Path
with ApsimModel('Maize', out_path =Path('my_model.apsimx').resolve()) as model:
        # current out path
        print(Path(model.path).name)
        'my_model.apsimx'
        model.save('./edited_maize_model.apsimx', reload=True)
        print(Path(model.path).name)
        'my_model.apsimx'