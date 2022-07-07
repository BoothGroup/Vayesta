.. include:: /include/links.rst
.. _edmet:


Extended Density-matrix embedding theory EDMET:
======================================================

In the following tutorial, the extended density-matrix embbeding theory (EDMET) is introduced as implemented in Vayesta_. Two examples (Finite systems
and custom Hamiltonians) are used to illustrate the capabilities of this methodology.


Finite Systems
^^^^^^^^^^^^^^^^^^^^^

The Vayesta_ `ref:edmet.EDMET` module is introduced. As starting point, the relevant modules of PySCF_ and Vayesta_ are loaded:

.. literalinclude:: edmetfinite.py
   :lines: 1-7

The pertinent variables to declare a finite system (water molecule) in PySCF_ are shown in the following snippet:

.. literalinclude:: edmetfinite.py
   :lines: 10-18

The `ref:EDMET` module enables the use of different mean-field approaches, like for instance, Density Functional Theory (DFT). As implemente in PySCF_, the relevant variables are called as shown in the following lines of code:

.. literalinclude:: edmetfinite.py
   :lines: 20-24

It is important to notice that the features of a DFT object in PySCF_ are not consistent with the characteristics of a Hartree-Fock object and therefore needs to be converted in the followin way:

.. literalinclude:: edmetfinite.py
   :lines: 26-27

To use the module, the function `ref:edmet` is declared and arguments are provided from previous steps as shown in the snippet below:

.. literalinclude:: edmetfinite.py
   :lines: 29-30

The arguments **dmet_threshold**, **oneshot**, and **make_dd_moments** are arguments employed to define specific strategies of the algortihm. The **solver** option offers the possibility to use the **EBFCI** and **EBCCSD**, which is the one selected in this example.

A fragmentation scheme is needed and, in this example, perfomed in the following manner:

.. literalinclude:: edmetfinite.py
   :lines: 32-34

To compare with the **CCSD** reference, the computation can be submitted as shown in the code below:

.. literalinclude:: edmetfinite.py
   :lines: 61-63

Relevant quantities, such as the total energy, can be printed as displayed in the following lines of code:

.. literalinclude:: edmetfinite.py
   :lines: 65-68
