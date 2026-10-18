.. include:: /include/links.rst

Controlling the output
----------------------

On import, Vayesta logs to the terminal and to the files ``vlog.txt`` and ``verr.txt`` (warnings and errors only).
This can be changed with command line arguments of the job script:

.. code-block:: console

   [~]$ python job.py --log job.log -vv        # Log file job.log, debug level
   [~]$ python job.py -q --errlog verr-job.txt  # No terminal output, different error log

Run ``python -c "import vayesta" --help`` for all options.

The output can also be reconfigured from within Python, with arguments that are not given taking their values
from the command line:

.. code-block:: python

   import logging
   import vayesta

   vayesta.setup(log_level=logging.DEBUG, logfile="job.log", errfile=False)

Vayesta's terminal output and log files can be switched off with

.. code-block:: python

   vayesta.setup(quiet=True, logfile=False, errfile=False, log_info=False)

Its messages are then only handled by the logging configuration of the program, via the logger ``"vayesta"``.
Note that this only applies after the call: ``import vayesta`` currently still sets up the default output first.

To print the versions of Vayesta and its dependencies, e.g. for a bug report, use ``print(vayesta.info())``.
