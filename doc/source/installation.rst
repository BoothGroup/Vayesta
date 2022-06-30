Installation
------------

* Requirements
    - NumPy
    - SciPy
    - PySCF

* Installation
    1. Clone from Git repository: `git clone git@github.com:BoothGroup/Vayesta.git`

    2. Build

        .. code-block:: console

            cd vayesta/libs
            mkdir build && cd build
            cmake ..
            make

    3. Add Vayesta to your PYTHONPATH environment variable, for example by adding this line to your .bashrc/profile:

        .. code-block:: console

            export PYTHONPATH=<path to Vayesta>:$PYTHONPATH
