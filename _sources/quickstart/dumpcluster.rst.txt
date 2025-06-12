.. include:: /include/links.rst

.. _dumpcluster:

Dumping Cluster Hamiltonians
============================

Some users may want to utilize Vayesta_ to easily define fragments within the system
and obtain the corresponding cluster Hamiltonians, but solve the embedding problems externally and with their own solvers.
To accomodate for this, the ``EWF`` class allows setting ``solver='Dump'``, which will dump orbitals and integrals
of all fragments to an HDF5_ file and exit.

The name of the HDF5 file is ``clusters.h5`` by default, but can be adjusted via an additional solver option:

.. literalinclude:: /../../examples/ewf/molecules/20-dump-clusters.py
    :lines: 26-27

.. note::
    The full example can be found at ``examples/ewf/molecules/20-dump-clusters.py``

The dump file contains a separate group for each fragment which was defined for the embedding.
The content of each group can be best illustrated via this code snippet:

.. literalinclude:: /../../examples/ewf/molecules/20-dump-clusters.py
    :lines: 56-87

For a spin-unrestricted calculation, the shapes and dataset names are slighlty different:

.. literalinclude:: /../../examples/ewf/molecules/20-dump-clusters.py
    :lines: 92-122
