import os
import sys
import subprocess

if len(sys.argv) > 1:
    directory = sys.argv[1]
else:
    directory = os.path.abspath(os.path.dirname(__file__))

examples = os.popen("find . | grep \.py$").readlines()
assert len(examples) > 0

N = len(examples)
errs = []
for eg in examples[1:]:
    print(eg)
    if 'run_examples.py' in eg: continue
    print("Running %s" % eg)
    errno = subprocess.call("python " + eg[:-1] + " -q", shell=True)
    if errno != 0:
        print("\033[91mException in %s \033[0m" % eg)
        errs.append(eg)

if len(errs) == 0:
    print("\033[92mAll examples passed \033[0m")
else:
    print("\033[91m Exceptions found: %d/%d examples failed \033[0m" % (len(errs), len(examples)))
    for eg in errs:
        print("Execption in %s" % eg)
