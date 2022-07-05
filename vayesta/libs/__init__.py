import logging
import os.path
import numpy as np


log = logging.getLogger(__name__)

def load_library(libname, required=False):
    try:
        if not libname.startswith('lib'):
            libname = 'lib' + libname
        path = os.path.dirname(__file__)
        return np.ctypeslib.load_library(libname, path)
    except OSError as e:
        if required:
            log.critical("Required library %s not found!", libname)
            raise e
        log.debug("Library %s not found", libname)
    return None

libcore = load_library('core')
