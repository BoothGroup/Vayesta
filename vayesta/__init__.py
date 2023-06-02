import sys
import os.path
import logging
import subprocess
import platform
import importlib

from vayesta.core import cmdargs
from vayesta.mpi import init_mpi


__version__ = '1.0.1'

# Command line arguments
args = cmdargs.parse_cmd_args()

# Initialization of MPI
if args.mpi is None:
    init_mpi(True, required=False)
else:
    init_mpi(args.mpi)
from vayesta.mpi import mpi

# Logging
from vayesta.core import vlog
if args.output_dir:
    os.makedirs(args.output_dir, exist_ok=True)

vlog.init_logging()
log = logging.getLogger(__name__)
log.setLevel(args.log_level)

fmt = vlog.VFormatter(indent=True)
# Log to stream
if (not args.quiet and mpi.is_master):
    # Everything except logging.OUTPUT goes to stderr:
    errh = vlog.VStreamHandler(sys.stderr, formatter=fmt)
    errh.addFilter(vlog.LevelExcludeFilter(exclude=[logging.OUTPUT]))
    log.addHandler(errh)
    # Log level logging.OUTPUT to stdout
    outh = vlog.VStreamHandler(sys.stdout, formatter=fmt)
    outh.addFilter(vlog.LevelIncludeFilter(include=[logging.OUTPUT]))
    log.addHandler(outh)
# Log to file
if (args.log or not mpi.is_master):
    logname = (args.log or "output")
    if args.output_dir:
        logname = os.path.join(args.output_dir, logname)
    log.addHandler(vlog.VFileHandler(logname, formatter=fmt))
# Error handler
errlog = args.errlog
if errlog:
    errfmt = vlog.VFormatter(show_mpi_rank=True, indent=False)
    if args.output_dir:
        errlog = os.path.join(args.output_dir, errlog)
    errhandler = vlog.VFileHandler(errlog, formatter=errfmt, add_mpi_rank=False)
    errhandler.setLevel(args.errlog_level)
    log.addHandler(errhandler)

# Print Logo, unless interactive Python interpreter
is_interactive = hasattr(sys, 'ps1')
if not is_interactive:
    logofile = os.path.abspath(os.path.join(os.path.dirname(__file__), 'logo.txt'))
    try:
        with open(logofile, 'r') as f:
            logo = f.read()
        logo = (logo.rstrip() + ' ')
    except FileNotFoundError:
        log.error("%s not found.", logofile)
        logo = ''
    log.info(logo + '\n', "version " + __version__)

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
        log.debug("%s not found.", name)
        return None

log.debug("Required packages:")
numpy = import_package('NumPy')
import_package('SciPy')
import_package('h5py')
pyscf = import_package('PySCF')
# Optional
import_package('mpi4py', False)
import_package('cvxpy', False)
dyson = import_package('dyson', False)
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
ddir = os.path.dirname(os.path.dirname(dyson.__file__))
dhash = get_git_hash(ddir)
log.debug("  * Dyson:    %s", dhash)
edir = os.path.dirname(os.path.dirname(ebcc.__file__))
ehash = get_git_hash(edir)
log.debug("  * EBCC:    %s", ehash)

# --- System information
log.debug('System:  node= %s  processor= %s' % (platform.node(), platform.processor()))

# --- MPI
if mpi:
    log.debug("MPI:  rank= %d  size= %d  node= %s", mpi.rank, mpi.size, platform.node())

# --- Environment
log.debug("Environment variables:")
omp_num_threads = os.getenv('OMP_NUM_THREADS')
if omp_num_threads is not None:
    omp_num_threads = int(omp_num_threads)
log.debug("  OMP_NUM_THREADS= %s", omp_num_threads)

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
