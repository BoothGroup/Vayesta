import argparse
import sys

DEFAULT_LOGLVL = 10
# In future, INFO level will be default:
#DEFAULT_LOGLVL = 20


def parse_cmd_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    #parser.add_argument('-o', '--output', help="If set, redirect all logging to this file.")
    parser.add_argument(
            '-o',
            '--output-dir',
            default='vayesta_output',
            help="Directory for Vayesta output files.",
    )
    parser.add_argument('--log', default='log.txt', help="Log to this file in addition to stdout and stderr.")
    parser.add_argument('--errlog', default='errors.txt', help="Warnings and errors will be written to this file.")
    parser.add_argument(
            '--errlog-level',
            type=int,
            default=30,
            help="Determines the default logging level for the error log.",
    )
    parser.add_argument('-q', '--quiet', action='store_true', help='Do not print to terminal')
    # Enables infov:
    parser.add_argument('-v',   action='store_const', dest='loglevel', const=15, default=DEFAULT_LOGLVL)
    # Enables timing:
    parser.add_argument('-vv',  action='store_const', dest='loglevel', const=10)
    # Enables debugv, timingv, trace:
    parser.add_argument('-vvv', action='store_const', dest='loglevel', const=1)
    parser.add_argument('--mpi', action='store_true', dest='mpi', default='auto')
    parser.add_argument('--no-mpi', action='store_false', dest='mpi', default='auto')
    args, unknown_args = parser.parse_known_args()

    # Remove known arguments:
    sys.argv = [sys.argv[0], *unknown_args]

    return args
