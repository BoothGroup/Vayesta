'''
Runs unit tests, and do the following:
 * Produce a coverage report (.coverage)
 * Produce a profile report (.profile)
 * Produce a codestyle report (.codestyle)
For post-processing:
 * `firefox htmlcov/index.html`
 * `snakeviz .profile`
 * `vi .codestyle`
'''

# Imports:
import os
import sys
import argparse
import importlib
import unittest
import pytest
import coverage
import cProfile
import logging
import io
import contextlib
import subprocess
from timeit import default_timer as timer

parser = argparse.ArgumentParser()
parser.add_argument('--max-test-level', type=int, default=2)
parser.add_argument('--test-levels', type=int, nargs='*', default=None)
parser.add_argument('--coverage', dest='coverage', action='store_true', default=True)
parser.add_argument('--no-coverage', dest='coverage', action='store_false')
parser.add_argument('--codestyle', dest='codestyle', action='store_true', default=True)
parser.add_argument('--no-codestyle', dest='codestyle', action='store_false')

args = parser.parse_args()
if args.test_levels is None:
    args.test_levels = list(range(args.max_test_level+1))
if args.coverage:
    import coverage
    os.system('rm -f .coverage')
if args.codestyle:
    import pycodestyle
    os.system('rm -f .codestyle')

# Clean up:
os.system('rm -f .profile')
os.system('rm -rf htmlcov')

# If only clean required, exit here:
if len(sys.argv) > 1 and sys.argv[1] == 'clean':
    sys.exit(0)

# Produce the codestyle report:
if args.codestyle:
    style = pycodestyle.StyleGuide(paths=['vayesta/'], report=pycodestyle.FileReport)
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        report = style.check_files()
    with open('.codestyle', 'w') as file:
        file.write('\n'.join(report.get_statistics()))
        file.write('\n\n')
        file.write(f.getvalue())

# Get a list of untracked files to ignore:
untracked = subprocess.run(['git', 'ls-files', '--others', '--exclude-standard', 'vayesta/'], capture_output=True, text=True).stdout
untracked = [x.strip() for x in untracked.split('\n')]
untracked += [
        '*/libs/*',
        '*/test_*.py',
        '*/dmet/pdmet.py',
        '*/misc/brueckner.py',
        '*/misc/counterpoise.py',
]

# Start the coverage report:
if args.coverage:
    print("Coverage omitting files:")
    [print(f) for f in untracked]
    cov = coverage.Coverage(omit=untracked)
    cov.start()

# Import Vayesta (inside coverage scope) and restrict logging to WARNING:
import vayesta
vayesta.log.setLevel(logging.WARNING)

# Start the profiler:
prof = cProfile.Profile()
prof.enable()

# Perform the tests:
lvls = ["fast", "not (fast or slow or veryslow)", "slow", "veryslow"]
for lvl in args.test_levels:
    t0 = timer()
    pytest.main(["vayesta/tests", "-m %s" % (lvls[int(lvl)])])
    print("Finished level %d tests in %.0f s" % (lvl, timer()-t0))

# End the profiler:
prof.disable()
prof.dump_stats('.profile')

if args.coverage:
    # End the coverage report:
    cov.stop()
    cov.save()

    # Generate HTML report for coverage:
    cov.html_report()
