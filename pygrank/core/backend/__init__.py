import importlib
import sys
import os
import json
from pygrank.core.backend.specification import *


_imported_mods = dict()


def load_backend(mod_name):
    if mod_name not in ['pytorch', 'numpy', 'tensorflow']:
        raise Exception("Unsupported backend "+mod_name)
    if mod_name in _imported_mods:
        mod = _imported_mods[mod_name]
    else:
        mod = importlib.import_module('.%s' % mod_name, __name__)
        _imported_mods[mod_name] = mod
    mod_name = ""
    for mod_name_part in __name__.split("."):
        if mod_name:
            mod_name += "."
        mod_name += mod_name_part
        if mod_name in sys.modules:
            thismod = sys.modules[mod_name]
            for api in specification.__dict__.keys():
                if api.startswith('__') or api in ["Iterable", "Optional", "Tuple", "BackendGraph", "BackendPrimitive"]:
                    continue
                if api in mod.__dict__:
                    setattr(thismod, api, mod.__dict__[api])
                else:  # pragma: no cover
                    raise Exception("Missing implementation for "+str(api))
    mod.backend_init()


def get_backend_preference():  # pragma: no cover
    config_path = os.path.join(os.path.expanduser('~'), '.pygrank', 'config.json')
    mod_name = None
    remind_where_to_find = False
    if "pygrankBackend" in os.environ:
        mod_name = os.getenv('pygrankBackend')
    elif os.path.exists(config_path):
        with open(config_path, "r") as config_file:
            config_dict = json.load(config_file)
            mod_name = config_dict.get('backend', '').lower()
            remind_where_to_find = (config_dict.get('reminder', 'true').lower() == 'true')

    if mod_name not in ['tensorflow', 'numpy', 'pytorch']:
        print("pygrank backend "
              + ("not found." if mod_name is not None or mod_name == "None" else str(mod_name)+" is not valid. "
                + "Automatically setting \"numpy\" as the backend of choice."),
              file=sys.stderr)
        set_backend_preference('numpy')
        return 'numpy'

    if remind_where_to_find:
        _notify_load(mod_name)
    return mod_name


def set_backend_preference(mod_name, remind_where_to_find=True):  # pragma: no cover
    default_dir = os.path.join(os.path.expanduser('~'), '.pygrank')
    if not os.path.exists(default_dir):
        os.makedirs(default_dir)
    config_path = os.path.join(default_dir, 'config.json')
    with open(config_path, "w") as config_file:
        json.dump({'backend': mod_name.lower(), 'reminder': str(remind_where_to_find).lower()}, config_file)
    if remind_where_to_find:
        _notify_load(mod_name)
    load_backend(mod_name)


def _notify_load(mod_name):
    print(f'The default pygrank backend has been set to "{mod_name}" ' +
          'by the file '
          + os.path.join(os.path.expanduser('~'), '.pygrank', 'config.json')
          + '\nSet your preferred backend as one of ["numpy", "pytorch", "tensorflow"] '
            'and "reminder": false in that file to remove this message from future runs.',
          file=sys.stderr)


load_backend(get_backend_preference())
