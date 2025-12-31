import gc

import pandas as pd
from apsimNGpy.core.apsim import ApsimModel
from apsimNGpy.core.mult_cores import MultiCoreManager
from utils import create_experiment, BASE_DIR
from apsimNGpy.parallel.process import custom_parallel
from apsimNGpy.core.pythonet_config import is_file_format_modified
from settings import path_to_MP_data, scratch_DIR
from itertools import product

prod = list(product((100, 165, 244, 326), (0.25, 0.5, 0.75, 1)))


# _____________create a function that edits_________________

def editor(nitrogen_residue, method):
    nitrogen, residue = nitrogen_residue
    if nitrogen < residue:
        if nitrogen != 0:
            raise ValueError("No way that nitrogen can be greater than the residue unless is is involved")
    match method.lower():
        case 'single':
            base_file = 'base_single.apsimx'

        case 'split':
            base_file = 'base_split.apsimx'

        case 'no_till':
            base_file = 'no_till.apsimx'

        case 'auto' | 'automatic':
            base_file = 'base_auto.apsimx'

        case 'other':
            base_file = f'plot_1C.apsimx'
        case _:
            raise ValueError(f"Invalid `{method}` method not supported/implemented.")
    model_path = path_to_MP_data / base_file

    if not model_path.exists():

        raise FileNotFoundError(f"{model_path} is not found at{path_to_MP_data}. Perhaps was deleted")
    else:
        _model = ApsimModel(model_path)
        # ___________________________________________edits________________________________________________
        _model.edit_model(model_type='Models.Manager', model_name='Tillage', Fraction=residue)
        if method == 'single':

            fertilizer_manager = "single_N_at_sowing"

        elif method == 'split':

            fertilizer_manager = "fertilize in phases"
        else:

            raise ValueError(f"Invalid `{method}` method not implemented to edit fertilizer manager")

        _model.edit_model(model_type='Models.Manager', model_name=fertilizer_manager, Amount=nitrogen)

        out_file_name = scratch_DIR / f"{method}_{nitrogen}_{int(residue * 100)}.apsimx"

        try:
            out_file_name.unlink(missing_ok=True)
        except PermissionError:
            pass
        _model.run()
        # _model.save(out_file_name)
        df = _model.results
        df['Nitrogen'] = nitrogen
        df['Residue'] = residue
        return df


if __name__ == "__main__":
    #
    data = list(custom_parallel(editor, prod, 'single', ncores=6))
    df = pd.concat(data)
    carbon = df[df['source_table'] == 'carbon']
    from utils import mva, plot_mva, open_file
    import matplotlib.pyplot as plt

    # ___________________all experimental data plot___________________

    file_name = 'single_soc.png'
    all_df = mva(df, window=7, col='SOC_0_15CM').reset_index(drop=True)
    plot_mva(all_df, 'SOC_0_15CM_roll_mean', color_palette='tab10',
             ylabel=r"SOC ($\mathrm{Mg\,ha^{-1}}$; 0–150 mm; BD = 1.45 $\mathrm{g\,cm^{-3}}$), 5-year rolling mean",
             xlabel='Time (Years)')
    # plt.tight_layout()
    plt.savefig(file_name, dpi=600)
    open_file(file_name)
    plt.close()
