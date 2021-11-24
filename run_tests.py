'''
Runs unit tests, and do the following:
 ⋅ Produce a coverage report (.coverage)
 ⋅ Produce a profile report (.profile)
 ⋅ Produce a codestyle report (.codestyle)
For post-processing:
 ⋅ `firefox htmlcov/index.html`
 ⋅ `snakeviz .profile`
 ⋅ `vi .codestyle`
'''

# Imports:
import os
import sys
import importlib
import unittest
import coverage
import cProfile
import logging
import pycodestyle
import io
import contextlib

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

# Start the coverage report:
cov = coverage.Coverage()
cov.start()

# Import Vayesta (inside coverage scope) and restrict logging to WARNING:
import vayesta
vayesta.log.setLevel(logging.WARNING)

# Start the profiler:
prof = cProfile.Profile()
prof.enable()

# Perform the tests:
loader = unittest.TestLoader()
runner = unittest.TextTestRunner(verbosity=2)
suite = loader.discover('vayesta/tests/')
runner.run(suite)

# End the profiler:
prof.disable()
prof.dump_stats('.profile')

# End the coverage report:
cov.stop()
cov.save()

# Generate HTML report for coverage:
cov.html_report()
