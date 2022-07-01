.. _molecules:

Using EWF for simulation of Molecules
=======================================

This tutorial introduces the use of the `ref:EWF` method to perform electronic structure of molecular simulations in Vayesta_. Firstly, Vayesta_ utilizes 
electronic structure properties computed with PySCF_. The needed modules can be imported as it is shown in the following snippet:

.. literalinclude:: fragmentation.py
   :lines: 1-4

The relevant Vayesta_ modules can be imported in a familiar Python way as shown in the following lines of code:
	   
.. literalinclude:: fragmentation.py
   :lines: 6-7

The computed mean-field electronic structure ground state calculations at the level of Restricted-Hartree-Fock (RHF) can be carried out as shown in the 
following snippet:
	   
.. literalinclude:: fragmentation.py
   :lines: 9-23

The PySCF_ results are subsequently parsed to the `ref:EWT` Vayesta_ method as depicted in the following snippet:

.. literalinclude:: fragmentation.py
   :lines: 25-27

The `ref:EWF1 method requires a fragmentation scheme. Vayesta_ has implemented different methods to perform fragmentations, whose description can be 
found in `REF:VAYESTA_MODULE`. The **IAO** fragmentation methodology is set as default and automatically used once the method **.kernel()** is invoked as 
is shown in these lines of code: 

.. literalinclude:: fragmentation.py
   :lines: 29-31

Alternatively, a second fragmentation procedure (the IAO+PAOs method) can be used as shown in the following lines:

.. literalinclude:: fragmentation.py
   :lines: 33-40


Important quantities, such as T2 amplitudes and 1-RDM from global EWF, can be computed using the following lines of code:

.. literalinclude:: fragmentation.py
   :lines: 42-46


Similarly, quantities such as spin-spin correlation functions and different observables can be computed using Vayesta_ ...... (TO BE COMPLETED)


	   
.. _PySCF: https://sunqm.github.io/pyscf/
.. _Vayesta: https://github.com/BoothGroup/Vayesta
