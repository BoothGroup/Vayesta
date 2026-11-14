import importlib.metadata
import importlib.resources
import logging
import os.path
import numpy as np


log = logging.getLogger(__name__)


def _get_library_dirs(libname):
    """Directories which may contain the compiled library `libname`.

    The compiled libraries and the imported Python sources can reside in different directories:
    - For editable installs; `importlib.resources` is aware of both locations.
    - When the sources are imported from a source checkout (e.g. when running the tests from the
      repository root), but Vayesta was installed non-editable; the library is then taken from the
      installed distribution.
    """
    dirs = [os.path.dirname(__file__)]
    try:
        for entry in importlib.resources.files(__name__).iterdir():
            if entry.name.startswith(libname + ".") and os.path.isfile(str(entry)):
                dirs.append(os.path.dirname(str(entry)))
    except (OSError, TypeError, ValueError) as e:
        log.debug("Could not search package resources for %s: %s", libname, e)
    try:
        dist = importlib.metadata.distribution("vayesta")
        for file in dist.files or []:
            if file.parent.as_posix() == "vayesta/libs" and file.name.startswith(libname + "."):
                dirs.append(os.path.dirname(str(dist.locate_file(file))))
    except importlib.metadata.PackageNotFoundError:
        pass
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
