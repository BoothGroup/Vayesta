.. include:: /include/links.rst

.. _ewf:

Wave Function Based Embedding (EWF)
===================================

This introduces the ``EWF`` class to perform a more generalized wave function based quantum embedding that
improves on DMET for *ab initio* systems, as presented in `Phys. Rev. X 12, 011046 <REF_PRX_>`_ [1]_.

Water Molecule
--------------

An embedded wave function calculation of a simple water molecule can be performed with the following code:

.. literalinclude:: /../../examples/ewf/molecules/01-simple-ccsd.py
    :linenos:

There are two main differences in setup compared to the :ref:`DMET embedding class <dmet>` in the definition of the ``EWF`` class (**lines 24--25**).

* The keyword argument :python:`bath_options=dict(threshold=1e-6)` defines the threshold for the MP2 bath natural orbitals
  (the variable :math:`\eta` in Ref. `1 <1_>`_) [2]_.

* No fragmentation scheme is explicitly provided, which by default results in the system being fully fragmented
  into simple atomic fragments as defined by `intrinsic atomic orbitals <REF_IAO_>`_ (IAOs).
  This is equivalent to adding the following lines before calling the embedding kernel method:

.. code-block:: python

    with emb.iao_fragmentation() as f:
        f.add_all_atomic_fragments()

which, in turn, is equivalent to

.. code-block:: python

    with emb.iao_fragmentation() as f:
        for atom in range(mol.natm):
            f.add_atomic_fragment(atom)

Other :ref:`fragmentations schemes <fragments>` can still be used, but have to be specified manually.


Cubic Boron Nitride (cBN)
-------------------------

In this example we calculate cubic boron nitride (Zinc Blende structure):

.. note::
    The basis set, auxiliary (density-fitting) basis set, and **k**-point sampling in this example are much too small
    for accurate results and only chosen for demonstration.


.. literalinclude:: /../../examples/ewf/solids/03-cubic-BN.py
    :linenos:

In **line 34** the setup of the embedding class is performed in the same way as for molecules.
Vayesta will detect if the mean field object ``mf`` has **k**-points defined. If these are found, then
the **k**-point sampled mean-field will automatically be folded to the :math:`\Gamma`-point of the equivalent
(in this case :math:`2\times2\times2`) Born--von Karman supercell.

.. note::

    Only Monkhorst-pack **k**-point meshes which include the :math:`\Gamma`-point are currently supported.

Note that instantiating the embedding class with a **k**-point sampled mean-field object
will automatically add the translational symmetry to the symmetry group stored in :python:`emb.symmetry`.
This assumes that the user will only define fragments within the original primitive unit cell,
which are then copied throughout the supercell using the translational symmetry (and this symmetry will be exploited
to avoid the cost of solving embedded fragments which are symmetrically equivalent to others).
For calculations of fragments across multiple primitive cells or where the primitive cell has not been explicitly
enlarged to encompass the full fragment space,
the translational symmetry should be removed by calling :python:`emb.symmetry.clear_translations()`
or overwritten via  :python:`emb.symmetry.set_translations(nimages)`, as demonstrated in
for the 1D Hubbard model :ref:`here <dmet_hub1d>`.

Performing the embedding in the supercell allows for optimal utilization of the locality of electron correlation,
as the embedding problems are only restricted to have the periodicity of the supercell, rather than the **k**-point sampled
primitive cell.
Properties, such as density-matrix calculated in **line 42**, will recover the full, primitive cell symmetry,
since they are obtained from a summation over all symmetry equivalent fragments in the supercell.
This is confirmed by the population analysis, which shows that the boron atom 2 has the same population than
boron atom 0, despite being part of a different primitive cell within the supercell:

.. code-block:: console

    Population analysis
    -------------------
	0 B:       q=  0.17325874  s=  0.00000000
	     0 0 B 1s          =  1.98971008
	     1 0 B 2s          =  0.76417671
	     2 0 B 2px         =  0.69095149
	     3 0 B 2py         =  0.69095149
	     4 0 B 2pz         =  0.69095149
	1 N:       q= -0.17325874  s=  0.00000000
	     5 1 N 1s          =  1.99053993
	     6 1 N 2s          =  1.17403392
	     7 1 N 2px         =  1.33622830
	     8 1 N 2py         =  1.33622830
	     9 1 N 2pz         =  1.33622830
	2 B:       q=  0.17325874  s=  0.00000000
	    10 2 B 1s          =  1.98971008
	    11 2 B 2s          =  0.76417671
	    12 2 B 2px         =  0.69095149
	    13 2 B 2py         =  0.69095149
	    14 2 B 2pz         =  0.69095149
	3 N:       q= -0.17325874  s=  0.00000000
	    15 3 N 1s          =  1.99053993
	    16 3 N 2s          =  1.17403392
	    17 3 N 2px         =  1.33622830
	    18 3 N 2py         =  1.33622830
	    19 3 N 2pz         =  1.33622830
        ...



.. [1] Max Nusspickel and George H. Booth, Phys. Rev. X 12, 011046 (2022).
.. [2] The definition of :math:`\eta` in Vayesta differs from Ref. 1 by a factor of :math:`1/2`. In Ref. 1 the
       occupation numbers of the spin traced density matrix (between 0 and 2) are compared against :math:`\eta`, whereas in Vayesta the eigenvalues of
       the spinned density-matrix (between 0 and 1) are used. The latter definition allows for a more natural comparison between spin-restricted
       and unrestricted calculations with the same value of :math:`\eta`.
