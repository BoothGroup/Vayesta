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

As starting point, the `ref:EWT` function uses the result of the previous **HF** calculation (denoted by **kmf**) to start the embedding procedure:

.. literalinclude:: diamond.py
   :lines: 35-38

important to notice is that the **IAO** fragmentation method has been used, together with a **sym_factor=2** variable, since the diamond unit cell has 
two carbon atoms per unit cell. The user-provided k-point mean-field calculation will be automatically folded to the supercell. 

Similarly, PySCF_ enables the use of explicit supercells for computing mean-field electronic ground-state as shown below:

.. literalinclude:: diamond.py
   :lines: 41-44

The `ref:EWF` embedding procedure can be also carried out using the supercell input as shown below:

.. literalinclude:: diamond.py
   :lines: 46-50

in which the **IAO** fragmentation has been used. The `ref:add_atomic_fragment` function is respectively changed by tuning the argument **sym_factor** 
provided by the number of images created to reproduce the kpoint mesh (i.e 2 as computed in the ncells variable). 

The results of these different calculation setups can be shown using the following lines of code:

.. literalinclude:: diamond.py
   :lines: 52-56



.. _PySCF: https://sunqm.github.io/pyscf/
.. _Vayesta: https://github.com/BoothGroup/Vayesta
