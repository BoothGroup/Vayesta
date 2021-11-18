'''
Run tests an produce reports on timings and coverage.
    - To view the coverage report: `firefox htmlcov/index.html`
    - To view the profile report: `snakeviz .profile`
'''

# Imports:
import os
import sys
import importlib
import unittest
import coverage
import cProfile
import logging

# Clean up:
os.system('rm -f .coverage')
os.system('rm -f .profile')
os.system('rm -rf htmlcov')

# If only clean required, exit here:
if len(sys.argv) > 1 and sys.argv[1] == 'clean':
    sys.exit(0)

# Start the coverage report:
cov = coverage.Coverage()
cov.start()

# Import Vayesta and restrict logging to WARNING:
import vayesta
vayesta.log.setLevel(logging.WARNING)

# Start the profiler:
prof = cProfile.Profile()
prof.enable()

# Perform the tests:
loader = unittest.TestLoader()
runner = unittest.TextTestRunner(verbosity=2)
suite = loader.discover('tests/')
runner.run(suite)

# End the profiler:
prof.disable()
prof.dump_stats('.profile')

# End the coverage report:
cov.stop()
cov.save()

# Generate HTML report for coverage:
cov.html_report()
