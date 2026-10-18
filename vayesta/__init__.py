import sys
import os.path
import logging
import subprocess
import platform
import importlib

from vayesta.core import cmdargs
from vayesta.mpi import init_mpi


__version__ = "1.0.1"

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

vlog.init_logging()
log = vlog.getLogger(__name__)
# Avoids Python's last-resort handler printing to stderr if setup adds no handlers
log.addHandler(logging.NullHandler())

# --- Required and optional packages


import numpy
import scipy
import h5py
import pyscf


def import_package(name, required=True):
    try:
        return importlib.import_module(name.lower())
    except ImportError:
        if required:
            raise
        return None


# Name -> module, or None if an optional package is not installed. Logged by setup
_packages = {"NumPy": numpy, "SciPy": scipy, "h5py": h5py, "PySCF": pyscf}
_packages.update({name: import_package(name, False) for name in ("mpi4py", "cvxpy", "dyson", "ebcc", "pygnme")})
dyson = _packages["dyson"]
ebcc = _packages["ebcc"]


def get_git_hash(dir):
    git_dir = os.path.join(dir, ".git")
    cmd = ["git", "--git-dir=%s" % git_dir, "rev-parse", "--short", "HEAD"]
    try:
        githash = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.STDOUT).rstrip()
    except (subprocess.CalledProcessError, OSError):
        # OSError: git is not installed
        githash = "<Not Found>"
    return githash


# --- Setup

# Handlers added by setup and new_log
_handlers = []
errhandler = None


def setup(
    log_level=None, logfile=None, errfile=None, errfile_level=None, quiet=None, output_dir=None, banner=True
) -> None:
    """Set up the output of Vayesta: the handlers of the Vayesta logger, and NumPy's print options.

    This is called on import of Vayesta, using the command line arguments. It can be called again to reconfigure
    the output, in which case the handlers added by the previous call are removed.

    Parameters
    ----------
    log_level : int, optional
        Log level of the Vayesta logger. Default: command line argument `--log-level` (or `-v`, `-vv`, `-vvv`).
    logfile : str or False, optional
        Name of the log file, or False for no log file. On MPI ranks other than the master rank, a log file is always
        written. Default: command line argument `--log`.
    errfile : str or False, optional
        Name of the error log file, or False for no error log file. Default: command line argument `--errlog`.
    errfile_level : int, optional
        Log level of the error log file. Default: command line argument `--errlog-level`.
    quiet : bool, optional
        If True, do not log to stdout and stderr. Default: command line argument `-q`.
    output_dir : str, optional
        Directory of the log files. Default: command line argument `-o`.
    banner : bool, optional
        If True, log the Vayesta logo (unless in an interactive interpreter) and, at debug level, the versions of the
        dependencies and system information. Default: True.
    """
    log_level = args.log_level if log_level is None else log_level
    logfile = args.log if logfile is None else logfile
    errfile = args.errlog if errfile is None else errfile
    errfile_level = args.errlog_level if errfile_level is None else errfile_level
    quiet = args.quiet if quiet is None else quiet
    output_dir = args.output_dir if output_dir is None else output_dir

    _setup_handlers(log_level, logfile, errfile, errfile_level, quiet, output_dir)
    if banner:
        _log_banner()
    numpy.set_printoptions(linewidth=120)


def _setup_handlers(log_level, logfile, errfile, errfile_level, quiet, output_dir):
    global errhandler
    for hdl in _handlers:
        log.removeHandler(hdl)
        hdl.close()
    _handlers.clear()
    errhandler = None

    def add_handler(hdl):
        log.addHandler(hdl)
        _handlers.append(hdl)

    log.setLevel(log_level)
    fmt = vlog.VFormatter(indent=True)
    # Log to stream
    if not quiet and mpi.is_master:
        # Everything except logging.OUTPUT goes to stderr:
        errh = vlog.VStreamHandler(sys.stderr, formatter=fmt)
        errh.addFilter(vlog.LevelExcludeFilter(exclude=[logging.OUTPUT]))
        add_handler(errh)
        # Log level logging.OUTPUT to stdout
        outh = vlog.VStreamHandler(sys.stdout, formatter=fmt)
        outh.addFilter(vlog.LevelIncludeFilter(include=[logging.OUTPUT]))
        add_handler(outh)
    if (logfile or errfile or not mpi.is_master) and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    # Log to file
    if logfile or not mpi.is_master:
        logname = logfile or "output"
        if output_dir:
            logname = os.path.join(output_dir, logname)
        add_handler(vlog.VFileHandler(logname, formatter=fmt))
    # Error handler
    if errfile:
        errfmt = vlog.VFormatter(show_mpi_rank=True, indent=False)
        if output_dir:
            errfile = os.path.join(output_dir, errfile)
        errhandler = vlog.VFileHandler(errfile, formatter=errfmt, add_mpi_rank=False)
        errhandler.setLevel(errfile_level)
        add_handler(errhandler)


def _log_banner():
    # Logo, unless interactive Python interpreter
    if not hasattr(sys, "ps1"):
        logofile = os.path.abspath(os.path.join(os.path.dirname(__file__), "logo.txt"))
        try:
            with open(logofile, "r") as f:
                logo = f.read()
            logo = logo.rstrip() + " "
        except FileNotFoundError:
            log.error("%s not found.", logofile)
            logo = ""
        log.info(logo + "\n", "version " + __version__)

    fmt = "  * %-10s  v%-8s  location: %s"
    log.debug("Required packages:")
    for name, package in _packages.items():
        if package is None:
            log.debug("%s not found.", name)
        else:
            log.debug(fmt, name, package.__version__, os.path.dirname(package.__file__))

    log.debug("Git hashes:")
    log.debug("  * Vayesta:  %s", get_git_hash(os.path.dirname(os.path.dirname(__file__))))
    log.debug("  * PySCF:    %s", get_git_hash(os.path.dirname(os.path.dirname(pyscf.__file__))))
    if dyson is not None:
        log.debug("  * Dyson:    %s", get_git_hash(os.path.dirname(os.path.dirname(dyson.__file__))))
    if ebcc is not None:
        log.debug("  * EBCC:     %s", get_git_hash(os.path.dirname(os.path.dirname(ebcc.__file__))))

    log.debug("System:  node= %s  processor= %s" % (platform.node(), platform.processor()))
    if mpi:
        log.debug("MPI:  rank= %d  size= %d  node= %s", mpi.rank, mpi.size, platform.node())
    log.debug("Environment variables:")
    omp_num_threads = os.getenv("OMP_NUM_THREADS")
    if omp_num_threads is not None:
        omp_num_threads = int(omp_num_threads)
    log.debug("  OMP_NUM_THREADS= %s", omp_num_threads)


def new_log(logname, fmt=None, remove_existing=True):
    if fmt is None:
        fmt = vlog.VFormatter(indent=True)
    if remove_existing:
        for hdl in log.handlers[:]:
            # Do not remove error handler and NullHandler
            if hdl is errhandler or isinstance(hdl, logging.NullHandler):
                continue
            log.removeHandler(hdl)
            if hdl in _handlers:
                _handlers.remove(hdl)
    hdl = vlog.VFileHandler(logname, formatter=fmt)
    log.addHandler(hdl)
    _handlers.append(hdl)


setup()
