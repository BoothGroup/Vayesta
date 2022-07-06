.. include:: /include/links.rst
.. _ewf:

Embedded Wave Function method (EWF)
===================================

As presented in PRX_

This tutorial introduces the use of the `ref:ewf.EWF` method to perform electronic structure simulations of molecules
and periodic systems using the embedded wave function class (EWF) as implemnted in Vayesta_.

Finite Systems
^^^^^^^^^^^^^^^

As starting point, Vayesta_ utilizes mean-field computed properties provided by the PySCF_ code. All the relevant modules are imported as shown in the following snippet:

.. literalinclude:: fragmentation.py
   :lines: 1-4

The relevant Vayesta_ modules can be imported in a familiar Python way as displayed in the following lines of code:

.. literalinclude:: fragmentation.py
   :lines: 6-7

The mean-field ground state properties are compued at the level of Restricted-Hartree-Fock (RHF) as shown below:

.. literalinclude:: fragmentation.py
   :lines: 9-23

The PySCF_ results are subsequently parsed to the `ref:ewt.EWT` Vayesta_ method as depicted in the following snippet:

.. literalinclude:: fragmentation.py
   :lines: 25-27

The `ref:ewf.EWF` method requires the use of a fragmentation scheme. Vayesta_ has implemented different methods to perform it, whose full description can be found in `REF:VAYESTA_MODULE`. The **IAO** fragmentation methodology is set as default and automatically used once the method **.kernel()** is invoked as is shown in these lines of code:

.. literalinclude:: fragmentation.py
   :lines: 29-31

Alternatively, a second fragmentation procedure (the **IAO+PAOs** method) can be used as shown in the following lines:

.. literalinclude:: fragmentation.py
   :lines: 33-40

Important quantities, such as **T2** amplitudes and one-body reduced density matrix (**1-RDM**) can be obtained from the global **EWF**, and computed using the following lines of code:

.. literalinclude:: fragmentation.py
   :lines: 42-46

Similarly, quantities such as spin-spin correlation functions and different observables can be calculated using Vayesta_ ...... (TO BE COMPLETED)

Extended Systems
^^^^^^^^^^^^^^^^^^

Additionally to the standard molecular quantum chemistry capabilities, PySCF_ enables the use of a variety of quantum chemistry methods for extended systems with Periodic Boundary Conditions (PBC). Vayesta_ utilizes these capabilities of performing ground state calculations as starting point. Initially, the relevant modules are imported as shown in the snipet below:

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
