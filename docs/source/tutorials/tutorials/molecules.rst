.. _molecules:

Using EWF for simulation of Molecules
=======================================

Vayesta_ employs electronic structure properties computed using PySCF_. The needed modules can be imported as it is shown in the following snipet:

.. literalinclude:: fragmentation.py
   :lines: 1-4

The corresponding modules to perform calculations with Vayesta_ can be imported as modules in a Python script in the following way:
	   
.. literalinclude:: fragmentation.py
   :lines: 6-7

The required electronic properties (ground state calculations) at the level of Restricted-Hartree-Fock (RHF) can be carried out as shown in the following 
snipet:
	   
.. literalinclude:: fragmentation.py
   :lines: 9-23

The PySCF_ results are subsequently used for the *EWF* Vayesta_ method as depicted in the following snippet:

.. literalinclude:: fragmentation.py
   :lines: 25-27

The succesfully use the *EWF* method, a fragmentation scheme is needed. Vayesta_ enables many methods to perform fragmentations, which can be found in 
REF:VAYESTA_MODULE. As an example, the **IAO** fragmentation methodology is set as default if the kernel function has been invoked as is shown below:

.. literalinclude:: fragmentation.py
   :lines: 29-31

Alternatively, a second fragmentation procedure (i.e the IAO+PAOs method) can be carried out using the following lines:

.. literalinclude:: fragmentation.py
   :lines: 33-40



	   
.. _PySCF: https://sunqm.github.io/pyscf/
.. _Vayesta: https://github.com/BoothGroup/Vayesta
