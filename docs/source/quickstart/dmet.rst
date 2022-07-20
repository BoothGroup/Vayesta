.. include:: /include/links.rst

.. _dmet:

Density-Matrix Embbeding Theory (DMET)
======================================

Vayesta_ can be used for DMET calculations of molecules, solids, and lattice models.
In this section, we give two simple examples: the calculations of a :math:`\textrm{H}_6` `molecule <Simple Molecule_>`_
and a `1D Hubbard model <Hubbard Model in 1D_>`_.

Simple Molecule
---------------

A simple DMET calculation of a :math:`\textrm{H}_6` ring can be performed as follows (the example can also be found in ``examples/dmet/01-simple-dmet.py``):

.. literalinclude:: /../../examples/dmet/01-simple-dmet.py
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

    For more information on the fragmentation and the ``add_atomic_fragment`` method,
    see section :ref:`Defining Fragments <fragments>`.

Finally, the two embedding problems are solved by calling ``dmet.kernel()``
and the resulting total energy is stored in ``dmet.e_tot``.

In **lines 32--37** these steps are repeated for a self-consistent DMET calculation, in which case only
the keyword argument ``maxiter=1`` has to be dropped.


.. note::

    The fragments in the calculation above are symmetry-related by a rotation of 120° or 240° around the `z`-axis.
    This symmetry can be added to the embedding class, such that only a single fragment needs to be solved:

    .. literalinclude:: /../../examples/dmet/02-rotational-symmetry.py
        :lines: 25-30

    see also ``example/dmet/02-rotations-symmetry.py``

.. note::

    The self-consistent optimization of a correlation potential in DMET can be ill-defined and will sometimes fail.
    In this case, changing the tolerance or starting point can sometimes help.
    The underlying origin of these difficulties are described in Ref. [1]_.

    Our implementation of self-consistent DMET follows that of Ref. [2]_ and requires the Python package ``cxvpy``
    for use.

Density-Matrices
----------------

In DMET expectation values are defined in terms of democratically partitioned one- and two-body density-matices.
These can be obtained by calling ``dmet.make_rdm1()`` and ``dmet.make_rdm2()`` as shown in example
``examples/dmet/03-density-matrices.py``:

.. literalinclude:: /../../examples/dmet/03-density-matrices.py
    :linenos:

.. _dmet_hub1d:

Hubbard Model in 1D
-------------------

In order to simulate lattice model systems, such as the Hubbard model, Vayesta provides the classes
``Hubbard1D``, ``Hubbard2D`` and ``LatticeMF`` in the ``lattmod`` package.
In the following example the half-filled, ten-site Hubbard chain is calculated with :math:`U = 6t` (where :math:`t` is the hopping parameter, which is 1 by default):

.. literalinclude:: /../../examples/dmet/63-hubbard-1d.py
    :linenos:

For lattice model systems, fragments for quantum embedding calculations are usually defined in terms of lattice sites.
For this purpose, the embedding class has the fragmentation context manager ``dmet.site_fragmentation()``.
Within the body of this context manager, fragments can be added as before with the method
``add_atomic_fragment``---for the purpose fo defining fragments, the sites are considered as atoms.
In **lines 19--20** of this example, the lattice is divided into two-site fragments, as depicted in :numref:`fig_hub1d`.

.. _fig_hub1d:
.. figure:: figures/1dhubbfig.png
   :alt:  1D Hubbard model
   :align: center
   :figclass: align-center

   Schematic depiction of the 1-D Hubbard model with two-sites fragments.

Just as for the :math:`\mathrm{H}_6` ring molecule in the example `above <Simple Molecule_>`_,
this lattice model system has an inherent (in this case translational) symmetry between the fragments, which can
be exploited.
This is done in the second DMET calculation in **lines 24--32**, where the translational symmetry is specified
in the following lines:

.. literalinclude:: /../../examples/dmet/63-hubbard-1d.py
    :lines: 27-28

The three integers in ``nimages`` specify the number of symmetry related copies (including the original)
along each lattice vector.
For a 1D system, the first lattice vector corresponds to the periodic dimension and is thus the only dimension
along which there are more than one copies.

.. [1] Faulstich et al., J. Chem. Theor. Comput. 18, 851-864 (2022).
.. [2] Wu et al., Phys. Rev. B 102, 085123 (2020).
