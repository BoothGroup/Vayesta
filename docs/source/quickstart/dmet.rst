.. include:: /include/links.rst
.. _dmet:


Density-Matrix Embbeding Theory (DMET)
======================================

In the following tutorial, the Density-matrix embbeding theory (DMET) as implemented in Vayesta_ is introduced.
Two examples (Finite systems and custom Hamiltonians) are used to illustrate the capabilities of this methodology.


Simple Molecule
---------------

A simple DMET calculation of a :math:`\textrm{H}_6` ring can be performed as follows (the example can also be found in ``examples/dmet/01-simple-dmet.py``):

.. literalinclude:: /../../../vayesta/examples/dmet/01-simple-dmet.py
    :linenos:


In **lines 9--17** the :math:`\textrm{H}_6` molecule is set up
and a restricted Hartree--Fock (RHF) calculation is performed using PySCF_.
We also perform an full system full configuration interaction (FCI) calculation (**lines 20--21**),
to obtain a reference value for the DMET results.

In **line 24**, a one-shot DMET calculation is instantiated with the Hartree--Fock object as first argument.
Additionally, the keyword argument ``solver='FCI'`` is used to select FCI
as the default solver and ``maxiter=1`` is used to skip the DMET self-consistency cycle.


In **lines 25--28** the fragments for the calculation are defined.
This is done with help of the context manager ``dmet.sao_fragmentation()``, which specifies that the
fragments added in the body of the context manager will refer to symmetrically orthogonalized atomic orbitals (SAOs).
The method ``f.add_atomic_fragment(atoms)`` adds a fragment comprised of
all orbitals corresponding to the atom ``atoms`` to the calculation.
In this case, we split the system into three fragments, containing two neighboring hydrogen atoms each.

.. note::

    ``add_atomic_fragment`` is only one of many ways to define fragments. See also TODO

Finally, the two embedding problems are solved by calling ``dmet.kernel()``
and the resulting total energy is stored in ``dmet.e_tot``.

In **lines 32--37** these steps are repeated for a self-consistent DMET calculation, in which case only
the keyword argument ``maxiter=1`` has to be dropped.


.. note::

    The fragments in the calculation above are symmetry-related by a rotation of 120° or 240° around the `z`-axis.
    This symmetry can be added to the embedding class, such that only a single fragment needs to be solved:

    .. literalinclude:: /../../../vayesta/examples/dmet/02-rotational-symmetry.py
        :lines: 25-30

    see also ``example/dmet/02-rotations-symmetry.py``


Custom Hamiltonians
-------------------

Costumized Hamiltonians can be also studied using the **DMET** methodology as implemented in Vayesta_. Initially, the required Vayesta_ modules are imported (and NumPy_ as an auxiliary library) as displayed in the following snippet:

.. literalinclude:: dmet1dhubbard.py
   :lines: 1-4

The most important parameters to set up the 1D Hubbard's model (as done in the module `ref:Lattmod`) are declared in the following lines of code:

.. literalinclude:: dmet1dhubbard.py
   :lines: 7-11

It is important to notice that function `ref:lattmod.Hubbard1D` contains different periodic boundary conditions, which in this case is the anti-periodic boundary condition. Using these variables as arguments, the corresponding `ref:lattmod.Hubbard1D` and `ref:lattmod.LatticeMF` are utilized to perform a mean-field calculation as displayed in this snippet:

.. literalinclude:: dmet1dhubbard.py
   :lines: 13-16

As in the finite case, a fragmentation scheme is needed to perform a **DMET** calculation. In the case of costumized Hamiltonians, the relevant the function is `ref:site_fragmentation`. The fragmentation procedure will be carried out using adjacent sites, as shown in the followin lines of code:

.. literalinclude:: dmet1dhubbard.py
   :lines: 18-23

The computation is carried out using the **FCI** solver. Alternatively, Vayesta_ can exploit the inherent translation symmetry displayed by the 1D-Hubbard's model. To use this option, one needs to declare a single fragment and then perform the translation over the desired direction. This can be done using the following lines of code:

.. literalinclude:: dmet1dhubbard.py
   :lines: 25-28

To specify translation vectors as parts of the full system lattice vectors by passing a list with three integers, **[n, m, l]**; the translation vectors will be set equal to the lattice vectors, divided by **n, m, l** in **a0, a1, and a2** direction, respectively. This is depicted schematically in **Figure(1)**.

.. figure:: figures/1dhbdtrsym.png
   :alt: aperiodic hubbard model
   :align: center
   :figclass: align-center

   **Figure(1)** Schematic depiction of the 1-D Hubbard model, half filling with double-site embedding fragmentation using the tsymmetric feature.

In this case, this is done with the following command:

.. literalinclude:: dmet1dhubbard.py
   :lines: 35-37


To confirm that this is correct, the number of cpmputed fragments can be counted and validate against the expected number in the following manner:

.. literalinclude:: dmet1dhubbard.py
   :lines: 38-40

Both methodologies can be compared, as shown in the following snippet:

.. literalinclude:: dmet1dhubbard.py
   :lines: 42-44
