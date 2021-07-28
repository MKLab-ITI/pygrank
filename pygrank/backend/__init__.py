import importlib
import sys
import os
import json
from pygrank.backend.specification import *


def load_backend(mod_name):
    if mod_name == 'pytorch':
        pass
    elif mod_name == 'numpy':
        pass
    elif mod_name == 'tensorflow':
        pass
    else:
        raise Exception("Unsupported backend "+mod_name)
    importlib.import_module('.%s' % mod_name, __name__)
    mod = importlib.import_module('.%s' % mod_name, __name__)
    thismod = sys.modules[__name__]
    for api in specification.__dict__.keys():
        if api.startswith('__'):
            continue
        if api in mod.__dict__:
            setattr(thismod, api, mod.__dict__[api])
        else:
            raise Exception("Missing implementation for "+str(api))


def get_backend_preference():
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
        print("pygrank backend "+("not found." if mod_name is not None or mod_name=="None" else str(mod_name)+" is not valid. " +
              "Automatically setting \"numpy\" as the backend of choice."),
              file=sys.stderr)
        set_backend_preference('numpy')
        return 'numpy'

    if remind_where_to_find:
        _notify_load(mod_name)
    return mod_name


def set_backend_preference(mod_name, remind_where_to_find=True):
    default_dir = os.path.join(os.path.expanduser('~'), '.pygrank')
    if not os.path.exists(default_dir):
        os.makedirs(default_dir)
    config_path = os.path.join(default_dir, 'config.json')
    with open(config_path, "w") as config_file:
        json.dump({'backend': mod_name.lower(), 'reminder': str(remind_where_to_find).lower()}, config_file)
    _notify_load(mod_name)


def _notify_load(mod_name):
    print(f'The default pygrank backend has been set to "{mod_name}" ' +
          'by the file '
          + os.path.join(os.path.expanduser('~'), '.pygrank', 'config.json')
          + '\nSet your preferred backend as one of ["numpy", "pytorch", "tensorflow"] '
            'and "reminder": false in that file to remove this message from future runs.',
          file=sys.stderr)


load_backend(get_backend_preference())
