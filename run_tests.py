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
import coverage
import cProfile
import logging
import pycodestyle
import io
import contextlib
import subprocess
from timeit import default_timer as timer

parser = argparse.ArgumentParser()
parser.add_argument('--max-test-level', type=int, default=1)
parser.add_argument('--test-levels', type=int, nargs='*', default=None)
args = parser.parse_args()
if args.test_levels is None:
    args.test_levels = list(range(args.max_test_level+1))

# Clean up:
os.system('rm -f .coverage')
os.system('rm -f .profile')
os.system('rm -f .codestyle')
os.system('rm -rf htmlcov')

# If only clean required, exit here:
if len(sys.argv) > 1 and sys.argv[1] == 'clean':
    sys.exit(0)

# Produce the codestyle report:
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
        '*/core/k2gamma.py',
        '*/dmet/pdmet.py',
        '*/misc/brueckner.py',
        '*/misc/counterpoise.py',
]

# Start the coverage report:
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
testdirs = [d.name for d in os.scandir('vayesta/tests') if (d.is_dir() and not d.name.startswith('_'))]
t0 = timer()
ncount_tot = 0
for lvl in args.test_levels:
    ncount_lvl = 0
    pattern = ('test%d_*.py' % lvl) if (lvl > 0) else 'test_*.py'
    t0_lvl = timer()
    for d in testdirs:
        loader = unittest.TestLoader()
        suite = loader.discover('vayesta/tests/%s/' % d, pattern=pattern)
        ncount = suite.countTestCases()
        if (ncount == 0):
            continue
        ncount_tot += ncount
        ncount_lvl += ncount
        runner = unittest.TextTestRunner(verbosity=2)
        t0_dir = timer()
        res = runner.run(suite)
        print("Finished %3d level %d tests in %-10s in %.0f s" % (ncount, lvl, "'"+d+"'", timer()-t0_dir))
    print("Finished %3d level %d tests in %.0f s" % (ncount_lvl, lvl, timer()-t0_lvl))
print("Finished %3d tests in %.0f s" %  (ncount_tot, timer()-t0))

# End the profiler:
prof.disable()
prof.dump_stats('.profile')

# End the coverage report:
cov.stop()
cov.save()

# Generate HTML report for coverage:
cov.html_report()
