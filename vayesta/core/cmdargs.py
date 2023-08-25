import argparse
import sys

DEFAULT_LOG = "vlog.txt"
DEFAULT_ERR = "verr.txt"
DEFAULT_LOGLVL = 20
DEFAULT_ERRLVL = 30


def parse_cmd_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    # Log files
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="directory for Vayesta output files [default: current working directory]",
    )
    parser.add_argument("--log", default="vlog.txt", help="name of the log file [default: %s]" % DEFAULT_LOG)
    parser.add_argument("--errlog", default="verr.txt", help="name of the error log file [default: %s]" % DEFAULT_ERR)
    # Log level
    parser.add_argument(
        "--log-level",
        type=int,
        default=DEFAULT_LOGLVL,
        help="logging level for the log file [default: %d]" % DEFAULT_LOGLVL,
    )
    parser.add_argument(
        "--errlog-level",
        type=int,
        default=DEFAULT_ERRLVL,
        help="logging level for the error log file [default: %d]" % DEFAULT_ERRLVL,
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Do not print to terminal")
    # Enables infov:
    parser.add_argument("-v", action="store_const", dest="log_level", const=15, help="Enables verbose logging output.")
    # Enables debug, timing
    parser.add_argument(
        "-vv", action="store_const", dest="log_level", const=10, help="Enables very verbose logging output."
    )
    # Enables debugv, timingv, trace
    parser.add_argument(
        "-vvv", action="store_const", dest="log_level", const=1, help="Enables complete logging output."
    )
    # MPI
    parser.add_argument(
        "--mpi", action="store_true", dest="mpi", default=None, help="Import mpi4py [default: attempt import]"
    )
    parser.add_argument("--no-mpi", action="store_false", dest="mpi", default=None, help="Do not import mpi4py")
    args, unknown_args = parser.parse_known_args()

    # Remove known arguments:
    sys.argv = [sys.argv[0], *unknown_args]
    return args
