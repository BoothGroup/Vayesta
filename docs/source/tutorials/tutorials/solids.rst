.. _solids:

Using EWF for Periodic Systems
==================================================

This tutorial introduces the use of the EWF method to perform electronic structure simulations of periodic systems in Vayesta_. Additionally to the 
standard molecular quantum chemistry capabilities, PySCF_ enables the use of a variety of quantum chemistry methods for extended systems with Periodic 
Boundary Conditions (PBC). Vayesta_ utilizes these capabilities of performing ground state calculations as starting point. Initially, the relevant modules 
are imported as shown in the snipet below:


.. literalinclude:: 1d_hubbard.py
   :lines: 1-3



.. _PySCF: https://sunqm.github.io/pyscf/
.. _Vayesta: https://github.com/BoothGroup/Vayesta
