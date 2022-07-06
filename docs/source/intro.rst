.. include:: /include/links.rst
.. _intro:

.. figure:: figures/graphene_titlepic.png
    :align: center
    :figclass: align-center

    Embedding of a carbon atom in graphene.


============
Introduction
============

Vayesta_ is a Python package for quantum embedding calculations in molecules and solids.

It builds on the functionality of the PySCF_ package to set up systems and find their mean-field solution.
With Vayesta_ one can define fragments within these systems in a highly flexible way
and add bath orbitals to represent the environment of each fragment.
The resulting quantum embedding problems can be solved with a variety of wave function based solvers,
making use of the efficient implementations within PySCF_. As a final step, expectation values,
such as the energy or reduced density-matrices, can be obtained from the collection of embedding problems.

Features
--------

- Fragments can be defined in terms of:
    - `Intrinsic atomic orbitals <REF_IAO_>`_ (IAO) [1]_
    - `Intrinsic atomic orbitals <REF_IAO_>`_ + orthogonalized `projected atomic orbitals <REF_PAO_>`_ (IAO+PAO)
    - Symmetrically (Löwdin) orthogonalized atomic orbitals (SAO)
    - Site basis (for lattice models)

- Bath orbitals:
    - `Density-matrix embedding theory (DMET) <REF_DMET_>`_ bath orbitals [2]_
    - `MP2 bath natural orbitals <REF_PRX_>`_ [3]_
    - Spatially closest bath orbitals (:math:`R^2` bath)

- Quantum embedding problems can be solved with the following PySCF solvers:
    - Second-order Møller--Plesset perturbation theory (MP2)
    - Configuration interaction with single and double excitations (CISD)
    - Coupled-cluster with single and double excitations (CCSD)
    - Full configuration-interaction (FCI)
    - Dump orbitals and integrals to HDF5 file for external processing

..
    - Expectation values:
        - Projected wave function correlation energy
        - Democratically partitioned density-matrices
        - Partitioned


.. [1] Gerald Knizia, J. Chem. Theory Comput. 9, 11, 4834 (2013).
.. [2] Gerald Knizia and Garnet Kin-Lic Chan, Phys. Rev. Lett. 109, 186404 (2012).
.. [3] Max Nusspickel and George H. Booth, Phys. Rev. X 12, 011046 (2022).
