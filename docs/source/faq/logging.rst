.. include:: /include/links.rst

Controlling the logging output
------------------------------

On import, Vayesta logs to the terminal and to the files ``vlog.txt`` and ``verr.txt`` (warnings and errors only).
This can be changed with command line arguments of the job script:

.. code-block:: console

   [~]$ python job.py --log job.log -vv        # Log file job.log, debug level
   [~]$ python job.py -q --errlog verr-job.txt  # No terminal output, different error log

Run ``python -c "import vayesta" --help`` for all options.

The logging can also be reconfigured from within Python, with arguments that are not given taking their values
from the command line:

.. code-block:: python

   import logging
   import vayesta

   vayesta.setup_logging(log_level=logging.DEBUG, logfile="job.log", errfile=False)

When Vayesta is used as part of another program, its terminal output and log files can be switched off with

.. code-block:: python

   vayesta.setup_logging(quiet=True, logfile=False, errfile=False)

Its messages are then only handled by the logging configuration of the program, via the logger ``"vayesta"``.
