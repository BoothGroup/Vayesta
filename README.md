# Vayesta

Authors
-------

OJ Backhouse, M Nusspickel, CJC Scott, GH Booth

Installation
------------

* Requirements
    - NumPy
    - SciPy
    - PySCF
    - CVXPY (for DMET functionality)

* Installation
    1. Clone from Git repository: `git clone git@github.com:BoothGroup/Vayesta.git`

    2. Build

        - `cd vayesta/libs`
        - `mkdir build && cd build`
        - `cmake ..`
        - `make`

    3. Add Vayesta to your PYTHONPATH environment variable, for example by adding this line to your .bashrc/profile:
        `export PYTHONPATH=<path to Vayesta>:$PYTHONPATH`
