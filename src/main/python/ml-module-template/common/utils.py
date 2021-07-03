"""General utilities to train a network."""
import os
import importlib
import sys
import collections
import pkgutil
import logging
from functools import wraps  
import numpy as np

logger = logging.getLogger(__name__)


class AttrDict(dict):
    """ 
    AttrDict: save the configuration files as default nest dictionary
    recursive save the config parameter
    e.g. d = {'a': a, 'b': {'c': c }}
    d.b.c equal the value of the inner dictionary
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def make_recursive(obj):
        """Finds all dictionaries and converts them to AttrDict"""
        if isinstance(obj, list):
            for i, l in enumerate(obj):
                obj[i] = AttrDict.make_recursive(l)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                obj[k] = AttrDict.make_recursive(v)
            return AttrDict(obj)
        return obj


def get_config(path):
    """
    get the config file and return as the self define AttDict
    """
    if path.endswith('.py'):
        # This is a bit dangerous if path would be needed
        # afterwards as it might be overwritten
        # content must be standard python code (no imports)
        content = open(path).read()
        # https://stackoverflow.com/questions/1463306/how-does-exec-work-with-locals
        # exec save the input as dict format
        ldict = {}
        exec(content, globals(), ldict)
        cfg = ldict['cfg']
    elif path.endswith('.json') or path.endswith('.cfg'):
        import json
        with open(path, 'r') as f:
            cfg = json.load(f)
    else: # path.endswith('.yaml'):
        raise NotImplementedError

    cfg = AttrDict.make_recursive(cfg)
    return cfg


def make_module_name(path):
    module_name = path[:-len(".py")].replace(os.path.sep, '.')
    logger.debug('Config from module %s', module_name)
    return module_name


def import_submodules(package_name):
    """
    [summary] import all the lib from thepackage folders
    Need this function for function registration plugin
    """

    importlib.import_module(package_name)
    package = sys.modules[package_name]
    for importer, name, is_package in pkgutil.walk_packages(package.__path__):
        # not sure why this check is necessary...
        if not importer.path.startswith(package.__path__[0]):
            continue
        name_with_package = package_name + "." + name
        importlib.import_module(name_with_package)
        if is_package:
            import_submodules(name_with_package)


class Register():
    """
    Register class for getting all the build function from 
    loss, modles, dataset submodules
    """
    __registered_fn__ = collections.defaultdict(dict)
    
    @classmethod
    def register(cls, typ, name):
        logger.debug("register type: %s, name: %s", typ, name)
        name = name.lower()

        def _register(build_fn):
            if name in cls.__registered_fn__[typ]:
                logger.debug("Name %s already chosen, will be overwritten in %s.", name, typ)
            cls.__registered_fn__[typ][name] = build_fn
            return build_fn
        return _register
    
    @classmethod
    def _get_all_registered_fn(cls):
        return cls.__registered_fn__
    
    @classmethod
    def _get_registered(cls, typ, name):

        def list_to_text(l):
            return (', ').join(l)

        logger.debug("get_registered type: %s, name: %s", typ, name)
        lower_name = name.lower()
        if typ not in cls.__registered_fn__:
            raise ValueError("Unknown type {}".format(typ))

        if lower_name in cls.__registered_fn__[typ]:
            return cls.__registered_fn__[typ][lower_name]
        else:
            raise ValueError("{} not found. Choose from {}."
                             .format(typ, list_to_text(get_all(typ))))

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


class CaptureStdout(object):
    def __init__(self):
        self.captured = ""

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = StringIO()
        return self

    def __exit__(self, *args, **kwargs):
        self.captured = sys.stdout.getvalue()
        sys.stdout = self._stdout


def onehot_initialization(a):
    """
    https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy/36960495
    """
    ncols = a.max()+1
    out = np.zeros(a.shape + (ncols,), dtype=np.bool)
    out[all_idx(a, axis=2)] = 1
    return out


def all_idx(idx, axis):
    grid = np.ogrid[tuple(map(slice, idx.shape))]
    grid.insert(axis, idx)
    return tuple(grid)


