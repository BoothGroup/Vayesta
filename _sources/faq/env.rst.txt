.. include:: /include/links.rst

Environment Variables
---------------------

An important point is to configure the ``pip`` to point to the directory ``$HOME/work``.
This can be done via the following lines of code:

.. code-block:: console

   [~]$ export PYTHONUSERBASE=${WORK}/.local
   [~]$ export PATH=$PYTHONUSERBASE/bin:$PATH
   [~]$ export PYTHONPATH=$PYTHONUSERBASE/lib/pythonX.X/site-packages:$PYTHONPATH

This ensures that the future installations will be stored in this directory.
