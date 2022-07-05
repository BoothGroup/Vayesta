.. _dumpcluster:


Dumping Cluster Hamiltonian
===============================

In this tutorial, the dumping cluster Hamiltonian functionality of Vayesta_ is introduced. To illustrate this capability, the pertinent modules of PySCF_ and Vayesta_ are loaded as shown in the following lines:

.. literalinclude:: dump_clusters.py
   :lines: 1-6

As an initial example, a finite system (water molecule) is used and all the relevant PySCF_ variables are declared as shown in the snippet below:

.. literalinclude:: dump_clusters.py
   :lines: 11-19

The atoms in the water molecule has been labeled as **O1**, **H2**, and **H3**. Then, a mean-field calculation (at the Hartree-Fock level of theory) is performed as shown in the following lines of code: 
	   
.. literalinclude:: dump_clusters.py
   :lines: 21-23

The Vayesta_ function `ref:ewf.EWF` is used as presented below: 
	   
.. literalinclude:: dump_clusters.py
   :lines: 25-28

The arguments includes required values such as the PySCF_ mean-field calculation (stored in **mf**) and the bath option. In this specific case, two new variables are used, namely, the solver **Dump** and the **solver_options** with the argument dumpfile where the clusters are stored.

The content of the file **clusters-rhf.h5** can be browsed, as illustrated in the following lines of code:

.. literalinclude:: dump_clusters.py
   :lines: 33-34

DESCRIPTION OF THE HDF5 FILE...

.. literalinclude:: dump_clusters.py
   :lines: 35-64




.. _PySCF: https://sunqm.github.io/pyscf/
.. _Vayesta: https://github.com/BoothGroup/Vayesta
