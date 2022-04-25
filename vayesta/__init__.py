
__version__ = '0.0.0'

logo = """\
__    __ ___ __    __ ___ ____ _______ ___
\ \  / // _ ## \  / // __/ __//__  __// _ #
 \ \/ // /_\ ## \/ // __/\__ \  / /  / /_\ #
  \__//_/   \_##  //____//___/ /_/  /_/   \_#
   ************/ /****************************
              /_/""".replace('#', '\\')

import sys
import os.path
import logging
import subprocess
import platform
import importlib

from .core import cmdargs
from .core import vlog
from .core.mpi import mpi


# Command line arguments
args = cmdargs.parse_cmd_args()

# Logging
vlog.init_logging()
log = logging.getLogger(__name__)
log.setLevel(args.loglevel)

fmt = vlog.VFormatter(indent=True)
# Log to stream
if (args.output is None) and (mpi.is_master):
    # Everything except logging.OUTPUT goes to stderr:
    errh = vlog.VStreamHandler(sys.stderr, formatter=fmt)
    errh.addFilter(vlog.LevelExcludeFilter(exclude=[logging.OUTPUT]))
    log.addHandler(errh)
    # Log level logging.OUTPUT to stdout
    outh = vlog.VStreamHandler(sys.stdout, formatter=fmt)
    outh.addFilter(vlog.LevelIncludeFilter(include=[logging.OUTPUT]))
    log.addHandler(outh)
# Log to file
if (args.output or args.log or (not mpi.is_master)):
    logname = (args.output or args.log) or "vayesta"
    log.addHandler(vlog.VFileHandler(logname, formatter=fmt))
# Error handler
errlog = args.errlog
if errlog:
    errfmt = vlog.VFormatter(show_mpi_rank=True, indent=False)
    errhandler = vlog.VFileHandler(errlog, formatter=errfmt, add_mpi_rank=False)
    errhandler.setLevel(args.errlog_level)
    log.addHandler(errhandler)

# Print Logo
log.info(logo + (' Version %s' % __version__) + '\n')

# --- Required modules

def import_package(name, required=True):
    fmt = '  * %-10s  v%-8s  location: %s'
    try:
        package = importlib.import_module(name.lower())
        log.debug(fmt, name, package.__version__, os.path.dirname(package.__file__))
        return package
    except ImportError:
        if required:
            log.critical("%s not found.", name)
            raise
        log.warning("%s not found.", name)
        return None

log.debug("Required packages:")
numpy = import_package('NumPy')
import_package('SciPy')
import_package('h5py')
pyscf = import_package('PySCF')
import_package('mpi4py', False)
import_package('cvxpy', False)
ebcc = import_package('ebcc', False)


# --- Git hashes

def get_git_hash(dir):
    git_dir = os.path.join(dir, '.git')
    cmd = ['git', '--git-dir=%s' % git_dir, 'rev-parse', '--short', 'HEAD']
    try:
        githash = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.STDOUT).rstrip()
    except subprocess.CalledProcessError:
        githash = "<Not Found>"
    return githash

log.debug("Git hashes:")
vdir = os.path.dirname(os.path.dirname(__file__))
vhash = get_git_hash(vdir)
log.debug("  * Vayesta:  %s", vhash)
pdir = os.path.dirname(os.path.dirname(pyscf.__file__))
phash = get_git_hash(pdir)
log.debug("  * PySCF:    %s", phash)

# --- MPI
if mpi:
    log.debug("MPI:  rank= %d  size= %d  node= %s", mpi.rank, mpi.size, platform.node())

# --- Environment
log.debug("Environment variables:")
omp_num_threads = os.getenv('OMP_NUM_THREADS')
if omp_num_threads is not None:
    omp_num_threads = int(omp_num_threads)
log.debug("  OMP_NUM_THREADS= %s", omp_num_threads)

log.debug("")

# ---

def new_log(logname, fmt=None, remove_existing=True):
    if fmt is None:
        fmt = vlog.VFormatter(indent=True)
    if remove_existing:
        for hdl in log.handlers[:]:
            # Do not remove error handler
            if hdl is errhandler:
                continue
            log.removeHandler(hdl)
    log.addHandler(vlog.VFileHandler(logname, formatter=fmt))

# --- NumPy
numpy.set_printoptions(linewidth=120)
