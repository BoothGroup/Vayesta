import sys
import os
import logging
import subprocess
import platform
import importlib
from pathlib import Path
from types import ModuleType
from typing import Literal

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


def import_optional(name: str) -> ModuleType | None:
    """Import an optional package, or return None if it is not installed."""
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


_required_packages: dict[str, ModuleType] = {"NumPy": numpy, "SciPy": scipy, "h5py": h5py, "PySCF": pyscf}
_optional_packages: dict[str, ModuleType | None] = {
    name: import_optional(name.lower()) for name in ("mpi4py", "cvxpy", "dyson", "ebcc", "pygnme")
}
dyson = _optional_packages["dyson"]
ebcc = _optional_packages["ebcc"]


def get_git_hash(path: str | os.PathLike[str]) -> str:
    """Short git hash of the repository at `path`, or "<Not Found>"."""
    cmd = ["git", f"--git-dir={Path(path) / '.git'}", "rev-parse", "--short", "HEAD"]
    try:
        return subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.STDOUT).rstrip()
    except (subprocess.CalledProcessError, OSError):
        # OSError: git is not installed
        return "<Not Found>"


def _system_info() -> str:
    def package_line(name: str, package: ModuleType | None) -> str:
        if package is None:
            return f"  * {name:<10}  not found"
        return f"  * {name:<10}  v{package.__version__:<8}  location: {Path(package.__file__ or '').parent}"

    lines = ["Required packages:"]
    lines += [package_line(name, package) for name, package in _required_packages.items()]
    lines += ["Optional packages:"]
    lines += [package_line(name, package) for name, package in _optional_packages.items()]
    lines += ["Git hashes:"]
    repos = {"Vayesta": Path(__file__).parent.parent}
    for name, package in (("PySCF", pyscf), ("Dyson", dyson), ("EBCC", ebcc)):
        repos[name] = Path(package.__file__ or "").parent.parent if package is not None else None
    for name, repo in repos.items():
        lines.append(f"  * {name + ':':<9} {get_git_hash(repo) if repo is not None else '<Not Found>'}")
    lines.append(f"System:  node= {platform.node()}  processor= {platform.processor()}")
    if mpi:
        lines.append(f"MPI:  rank= {mpi.rank}  size= {mpi.size}  node= {platform.node()}")
    lines.append("Environment variables:")
    lines.append(f"  OMP_NUM_THREADS= {os.getenv('OMP_NUM_THREADS')}")
    return "\n".join(lines)


def info() -> str:
    """Return the versions of Vayesta and its dependencies, and information about the system.

    Useful for bug reports: `print(vayesta.info())`.
    """
    return f"Vayesta version {__version__}\n{_system_info()}"


# --- Setup

# Handlers added by setup and new_log
_handlers: list[logging.Handler] = []
errhandler: logging.Handler | None = None


def setup(
    log_level: int | None = None,
    logfile: str | Literal[False] | None = None,
    errfile: str | Literal[False] | None = None,
    errfile_level: int | None = None,
    quiet: bool | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    log_info: bool = True,
) -> None:
    """Set up the output of Vayesta, i.e. the handlers of the Vayesta logger.

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
    output_dir : str or os.PathLike, optional
        Directory of the log files. Default: command line argument `-o`.
    log_info : bool, optional
        If True, log the Vayesta version and, at debug level, the information returned by :func:`info`.
        Default: True.
    """
    _setup_handlers(
        log_level=args.log_level if log_level is None else log_level,
        logfile=args.log if logfile is None else logfile,
        errfile=args.errlog if errfile is None else errfile,
        errfile_level=args.errlog_level if errfile_level is None else errfile_level,
        quiet=args.quiet if quiet is None else quiet,
        output_dir=args.output_dir if output_dir is None else output_dir,
    )
    if log_info:
        log.info("Vayesta version %s", __version__)
        log.debug(_system_info())


def _setup_handlers(
    log_level: int,
    logfile: str | Literal[False],
    errfile: str | Literal[False],
    errfile_level: int,
    quiet: bool,
    output_dir: str | os.PathLike[str],
) -> None:
    global errhandler
    for hdl in _handlers:
        log.removeHandler(hdl)
        hdl.close()
    _handlers.clear()
    errhandler = None

    def add_handler(hdl: logging.Handler) -> None:
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
    outdir = Path(output_dir) if output_dir else Path()
    if logfile or errfile or not mpi.is_master:
        outdir.mkdir(parents=True, exist_ok=True)
    # Log to file
    if logfile or not mpi.is_master:
        add_handler(vlog.VFileHandler(str(outdir / (logfile or "output")), formatter=fmt))
    # Error handler
    if errfile:
        errfmt = vlog.VFormatter(show_mpi_rank=True, indent=False)
        errhandler = vlog.VFileHandler(str(outdir / errfile), formatter=errfmt, add_mpi_rank=False)
        errhandler.setLevel(errfile_level)
        add_handler(errhandler)


def new_log(
    logname: str | os.PathLike[str], fmt: logging.Formatter | None = None, remove_existing: bool = True
) -> None:
    """Log to a new file `logname`, by default replacing all handlers except the error log file."""
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
    hdl = vlog.VFileHandler(str(logname), formatter=fmt)
    log.addHandler(hdl)
    _handlers.append(hdl)


setup()
