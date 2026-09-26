import importlib.resources
import logging
import os.path
import numpy as np


log = logging.getLogger(__name__)


def _get_library_dirs(libname):
    """Directories which may contain the compiled library `libname`.

    For editable installs, the compiled libraries and the Python sources can reside in different directories.
    `importlib.resources` is aware of both locations.
    """
    dirs = [os.path.dirname(__file__)]
    try:
        for entry in importlib.resources.files(__name__).iterdir():
            if entry.name.startswith(libname + ".") and os.path.isfile(str(entry)):
                dirs.append(os.path.dirname(str(entry)))
    except (OSError, TypeError, ValueError) as e:
        log.debug("Could not search package resources for %s: %s", libname, e)
    return list(dict.fromkeys(dirs))


def load_library(libname, required=False):
    if not libname.startswith("lib"):
        libname = "lib" + libname
    error = None
    for path in _get_library_dirs(libname):
        try:
            return np.ctypeslib.load_library(libname, path)
        except OSError as e:
            error = e
    if required:
        log.critical("Required library %s not found!", libname)
        raise error
    log.debug("Library %s not found", libname)
    return None


libcore = load_library("core")
