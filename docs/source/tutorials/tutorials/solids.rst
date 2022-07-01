.. _solids:

Using EWF for Periodic Systems
==================================================

This tutorial introduces the use of the EWF method to perform electronic structure simulations of periodic systems in Vayesta_. Additionally to the 
standard molecular quantum chemistry capabilities, PySCF_ enables the use of a variety of quantum chemistry methods for extended systems with Periodic 
Boundary Conditions (PBC). Vayesta_ utilizes these capabilities of performing ground state calculations as starting point. Initially, the relevant 
modules are imported as shown in the snipet below:

.. literalinclude:: diamond.py
   :lines: 1-6

Subsequently, the required Vayesta_ modules can be imported as indicated in the following lines of code:

.. literalinclude:: diamond.py
   :lines: 8-9

PySCF_ supplies the user with functions that are able to build a crystalline structure and a user-defined **k-mesh** as shown below:

.. literalinclude:: diamond.py
   :lines: 11-23

where a diamond crystalline structure has been created. To perform electronic structure calculations with **HF** and using **k-points**, the following 
lines of code are needed:

.. literalinclude:: diamond.py
   :lines: 26-28

Similarly, a full **CCSD** calculation is carried out as presented in the following snipet:

.. literalinclude:: diamond.py
   :lines: 31-32

As starting point, the ref:EWT* function uses the result of the previous calculation *kmf* to start the embedding procedure:

.. literalinclude:: diamond.py
   :lines: 35-38

important to notice is that the **IAO** fragmentation method has been used, together with a **sym_factor=2** variable, since the diamond unit cell there 
are two C atoms per unit cell. 




.. _PySCF: https://sunqm.github.io/pyscf/
.. _Vayesta: https://github.com/BoothGroup/Vayesta
