.. parallel:


Parallelism in Vayesta
===========================

Vayesta_ has been constructed to perform parallel calculations at different levels. A first layer of parallelisation is obtained by using the OpenBLAS_ 
library (via the numpy_ package) which offers **openmp** parallelization for different mathematical operations.

In a second layer of optimization, Vayesta_ includes C-extended codes for providing a speed up in the calculations in critical operations employing 
an **openmp** parallelization strategy.

Finally, Vayesta_ has implemented a third layer of parallelisation, where different operations between fragments are carried out using a **mpi** strategy 
aiming for an enhacement in the scaling up of the calculations. The **mpi** implementation can be used by running any Vayesta_ script in the following 
manner:

.. code-block:: console

   [~]$ mpirun -np #cores  vayesta_script.py

In this tutorial, 







.. _OpenBLAS: https://github.com/xianyi/OpenBLAS
.. _PySCF: https://sunqm.github.io/pyscf/
.. _Vayesta: https://github.com/BoothGroup/Vayesta
.. _numpy: https://numpy.org/
