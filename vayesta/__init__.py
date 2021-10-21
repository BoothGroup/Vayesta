
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

from .core import cmdargs
from .core import vlog

# Command line arguments
args = cmdargs.parse_cmd_args()

# Logging
vlog.init_logging()
log = logging.getLogger(__name__)
log.setLevel(args.loglevel)

fmt = vlog.VFormatter(indent=True)
# Log to stream
if args.output is None:
    # Everything except logging.OUTPUT goes to stderr:
    errh = vlog.VStreamHandler(sys.stderr, formatter=fmt)
    errh.addFilter(vlog.LevelExcludeFilter(exclude=[logging.OUTPUT]))
    log.addHandler(errh)
    # Log level logging.OUTPUT to stdout
    outh = vlog.VStreamHandler(sys.stdout, formatter=fmt)
    outh.addFilter(vlog.LevelIncludeFilter(include=[logging.OUTPUT]))
    log.addHandler(outh)
# Log to file
if (args.output or args.log):
    logname = args.output or args.log
    log.addHandler(vlog.VFileHandler(logname, formatter=fmt))

# Print Logo
log.info(logo + (' Version %s' % __version__) + '\n')

# Required modules
log.debug("Required modules:")

# NumPy
fmt = '  * %-10s  v%-8s  location: %s'
try:
    import numpy
    log.debug(fmt, 'NumPy', numpy.__version__, os.path.dirname(numpy.__file__))
except ImportError:
    log.critical("NumPy not found.")
    raise
# SciPy
try:
    import scipy
    log.debug(fmt, 'SciPy', scipy.__version__, os.path.dirname(scipy.__file__))
except ImportError:
    log.critical("SciPy not found.")
    raise
# h5py
try:
    import h5py
    log.debug(fmt, 'h5py', h5py.__version__, os.path.dirname(h5py.__file__))
except ImportError:
    log.critical("h5py not found.")
    raise
# PySCF
try:
    import pyscf
    log.debug(fmt, 'PySCF', pyscf.__version__, os.path.dirname(pyscf.__file__))
except ImportError:
    log.critical("PySCF not found.")
    raise
# mpi4py
try:
    import mpi4py
    log.debug(fmt, 'mpi4py', mpi4py.__version__, os.path.dirname(mpi4py.__file__))
    from mpi4py import MPI
except ImportError:
    MPI = False
    log.debug("mpi4py not found.")

#log.debug("")

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

# --- MPI info
# TODO Add MPI implementation info
if MPI:
    MPI_comm = MPI.COMM_WORLD
    MPI_rank = MPI_comm.Get_rank()
    MPI_size = MPI_comm.Get_size()
    log.debug("MPI info:")
    log.debug("  * rank %d / %d", MPI_rank, MPI_size)
log.debug("")

# ---

def new_log(logname, fmt=None, remove_existing=True):
    if fmt is None:
        fmt = vlog.VFormatter(indent=True)
    if remove_existing:
        for hdl in log.handlers[:]:
            log.removeHandler(hdl)
    log.addHandler(vlog.VFileHandler(logname, formatter=fmt))

# --- NumPy
numpy.set_printoptions(linewidth=120)
