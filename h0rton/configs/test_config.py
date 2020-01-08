import os, sys
import importlib
import json
from addict import Dict

class TestConfig:
    """Nested dictionary representing the configuration for h0_inference

    """
    def __init__(self, user_cfg):
        """
        Parameters
        ----------
        user_cfg : dict or Dict
            user-defined configuration
        
        """
        self.__dict__ = Dict(user_cfg)

    @classmethod
    def from_file(cls, user_cfg_path):
        """Alternative constructor that accepts the path to the user-defined configuration python file

        Parameters
        ----------
        user_cfg_path : str or os.path object
            path to the user-defined configuration python file

        """
        dirname, filename = os.path.split(os.path.abspath(user_cfg_path))
        module_name, ext = os.path.splitext(filename)
        sys.path.append(dirname)
        if ext == '.py':
            #user_cfg_file = map(__import__, module_name)
            #user_cfg = getattr(user_cfg_file, 'cfg')
            user_cfg_script = importlib.import_module(module_name)
            user_cfg = getattr(user_cfg_script, 'cfg')
            return cls(user_cfg)
        elif ext == '.json':
            with open(user_cfg_path, 'r') as f:
                user_cfg_str = f.read()
            user_cfg = Dict(json.loads(user_cfg_str))
            return cls(user_cfg)
        else:
            raise NotImplementedError("This extension is not supported.")