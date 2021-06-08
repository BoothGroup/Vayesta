"""Vayesta"""

import os.path
import logging
import subprocess

from .core import cmdargs
from .core import vlog

__version__ = 'v0.0.0'

# Command line arguments
args = cmdargs.parse_cmd_args()
strict = args.strict

# Logging
vlog.init_logging()
log = logging.getLogger(__name__)
log.setLevel(args.loglevel)

# Default log
fmt = vlog.VFormatter(indent=True)
log.addHandler(vlog.VFileHandler(args.logname, formatter=fmt))
# Warning log (for WARNING and above)
wh = vlog.VFileHandler("vayesta-warnings")
wh.setLevel(logging.WARNING)
log.addHandler(wh)

log.info("+%s+", (len(__version__)+10)*'-')
log.info("| Vayesta %s |", __version__)
log.info("+%s+", (len(__version__)+10)*'-')

def get_git_hash(dir):
    git_dir = os.path.join(dir, '.git')
    cmd = ['git', '--git-dir=%s' % git_dir, 'rev-parse', '--short', 'HEAD']
    try:
        githash = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.STDOUT).rstrip()
    except subprocess.CalledProcessError:
        githash = "<Not Found>"
    return githash

# Print git commit hash
vdir = os.path.dirname(os.path.dirname(__file__))
vhash = get_git_hash(vdir)
log.info("Git hash: %s", vhash)
log.info("")

# Required modules
log.debug("Required modules:")
log.changeIndentLevel(1)

# NumPy
fmt = '%-10s  v%-8s  found at  %s'
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
    pyscf_dir = os.path.dirname(os.path.dirname(pyscf.__file__))
    pyscf_hash = get_git_hash(pyscf_dir)
    log.info("PySCF Git hash: %s", pyscf_hash)
except ImportError:
    log.critical("PySCF not found.")
    raise
# mpi4py
try:
    import mpi4py
    log.debug(fmt, 'mpi4py', mpi4py.__version__, os.path.dirname(mpi4py.__file__))
    from mpi4py import MPI
    MPI_comm = MPI.COMM_WORLD
    MPI_rank = MPI_comm.Get_rank()
    MPI_size = MPI_comm.Get_size()
    log.debug("MPI(rank= %d , size= %d)", MPI_rank, MPI_size)
except ImportError:
    log.debug("mpi4py not found.")

log.changeIndentLevel(-1)
log.debug("")
