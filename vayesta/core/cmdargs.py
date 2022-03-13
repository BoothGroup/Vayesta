import argparse
import sys

DEFAULT_LOGLVL = 10
# In future, INFO level will be default:
#DEFAULT_LOGLVL = 20

def parse_cmd_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('-o', '--output', help="If set, redirect all logging to this file.")
    parser.add_argument('--output-dir', help="Directory for Vayesta output files.", default='vayesta_files')
    parser.add_argument('--log', help="If set, log to this file in addition to stdout and stderr.")
    parser.add_argument('--errlog', default='vayesta-errors', help="Warnings and errors will be written to this file.")
    parser.add_argument('--errlog-level', type=int, default=30, help="Determines the default logging level for the error log.")
    parser.add_argument('-v',   action='store_const', dest='loglevel', const=15, default=DEFAULT_LOGLVL)    # Enables infov
    parser.add_argument('-vv',  action='store_const', dest='loglevel', const=10)                            # Enables timing
    parser.add_argument('-vvv', action='store_const', dest='loglevel', const=1)                             # Enables debugv, timingv, trace
    args, unknown_args = parser.parse_known_args()

    # Remove known arguments:
    sys.argv = [sys.argv[0], *unknown_args]

    return args
