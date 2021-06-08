import os.path
import numpy as np

def load_library(libname):
    if not libname.startswith('lib'):
        libname = 'lib' + libname
    path = os.path.dirname(__file__)
    return np.ctypeslib.load_library(libname, path)

libcore = load_library('core')
