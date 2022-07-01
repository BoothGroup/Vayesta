.. _1dhubbard:

Custom Hamiltonians 
==========================

Model Hamiltonians are commonly used for simulating systems of interest both in Quantum Chemistry calculations as well as in Condensed-matter studies. 
Vayesta_ enables the creation of customized Hamiltonians by employing the module 'ref:lattmod' and the functions `ref:Hubbard_1d` and `ref:LatticeMF`
contained within. 

To perform calculations using this feature, the required modules must be imported as shown in the following snippet:

.. literalinclude:: 1d_hubbard.py
   :lines: 1-3

The initial conditions for an user-defined Hubbard's model are declared as shown in the following lines of code:

.. literalinclude:: 1d_hubbard.py
   :lines: 5-7

The creation of the Lattice model is performed automatically by Vayesta_ using these commands:

.. literalinclude:: 1d_hubbard.py
   :lines: 5-7
   
The function `ref:Hubbard1D` is a specialized function used to create a Hubbard's model based on the user-defined variables such as *nsite*, *nelectron*, and *hubbard_u* as shown below:

.. literalinclude:: 1d_hubbard.py
   :lines: 5-7
   
An important feature of Vayesta_ is the way in which desired properties are printed and stored. 



.. _Vayesta: https://github.com/BoothGroup/




