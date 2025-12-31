from pathlib import Path
import re
from apsimNGpy.core.config import set_apsim_bin_path, get_apsim_bin_path
import configparser

CURRENT_BIN = get_apsim_bin_path()
EXISTS = False
# check if the Current bin is not none
if CURRENT_BIN:
    EXISTS = True
# read in from the config_file
config_file = 'project_config.ini'
config = configparser.ConfigParser()
config.read(config_file)


def search_version_like(path):
    m = re.search(r'APSIM\s?\d{4}\.\d{1,2}\.\d+\.\d+', str(path))
    return m.group(0)
