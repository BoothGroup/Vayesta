.. include:: /include/links.rst
.. _parallel:


Parallelism in Vayesta
===========================

Vayesta_ has been constructed to perform parallel calculations at different levels. A first layer of parallelisation is obtained by using the OpenBLAS_
library (via the NumPy_ package) which offers **openmp** parallelization for different mathematical operations.

In a second layer of optimization, Vayesta_ includes C-extended codes for providing a speed up in the calculations in critical operations employing
an **openmp** parallelization strategy.

Finally, Vayesta_ has implemented a third layer of parallelisation, where different operations between fragments are carried out using a **mpi** strategy
aiming for an enhacement in the scaling up of the calculations. The **mpi** implementation can be used by running any Vayesta_ script in the following
manner:

.. code-block:: console

   [~]$ mpirun -np #ncores  vayesta_script.py

However, it is relevant to use a correct number of cores to perform an efficient **mpi** calculation. To this aim, a new water molecule example is used
explicitly showing the use of the module `ref:vayesta.mpi`. Firstly, the pertinent Vayesta_ modules are loaded in the following way:

.. literalinclude:: parallel.py
   :lines: 6-8

The PySCF_ molecule object is correspondingly built:

.. literalinclude:: parallel.py
   :lines: 10-20

It is important to define an individual output file for every process as is declared in `ref:mol.output`. The initial mean-field method (in this case is
**HF**) used to compute the ground-state wavefunction is also used within the **mpi** methdology in this manner:

.. literalinclude:: parallel.py
   :lines: 23-25

The `ref:EWT` method is used in the following manner:

.. literalinclude:: parallel.py
   :lines: 27-29

In this **mpi** implementation all the results are gathered in the master process. This proccess can be also used for computing the total energies
provided at the different levels of theory as is shown in the following snippet:

.. literalinclude:: parallel.py
   :lines: 32-38

To provide a good computational scaling, it is strongly suggested that **N_frag % #ncores = 0**, where **N_frag** is the number of fragments in which the
system has been divided. For this specific case, the suggested command is:

.. code-block:: console

   [~]$ mpirun -np 3 python parallel.py

since the water molecule is fragmented into 3 different **IAO** components.
