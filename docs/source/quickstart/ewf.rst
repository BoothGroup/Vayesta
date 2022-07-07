.. include:: /include/links.rst
.. _ewf:

Wave Function Based Embedding (EWF)
===================================

This tutorial introduces the ``EWF`` class to perform wave function based quantum embedding,
as presented in `Phys. Rev. X 12, 011046 <REF_PRX_>`_ [1]_.

Water Molecule
--------------

.. literalinclude:: /../../../vayesta/examples/ewf/molecules/01-simple-ccsd.py
    :linenos:

Compared to the DMET embedding class presented :ref:`above <dmet>`,
there are two main differences in the setup of the embedded wave function calculation in **lines 24--25**:

* The keyword argument ``bath_options=dict(threshold=1e-6)`` defines the threshold for the MP2 bath natural orbitals
  (the variable :math:`\eta` in Ref. `1 <1_>`_) [2]_.

* No fragmentation scheme is defined.

In the embedded wave function method, a fragmentation into atomic fragments, defined in terms of `intrinsic atomic orbitals <REF_IAO_>`_ (IAOs)
is assumed by default. This is equivalent to adding the following lines before calling the kernel in **line 25**:

.. code-block:: python

    with emb.iao_fragmentation() as f:
        f.add_all_atomic_fragments()

which, in turn, is equivalent to

.. code-block:: python

    with emb.iao_fragmentation() as f:
        for atom in range(mol.natm):
            f.add_atomic_fragment(atom)

Other fragmentations schemes are still possible, but have to be specified manually.


Extended Systems
----------------

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

.. [1] Max Nusspickel and George H. Booth, Phys. Rev. X 12, 011046 (2022).
.. [2] The definition of :math:`\eta` in Vayesta differs from Ref. 1 by a factor of :math:`1/2`. In Ref. 1 the
       occupation numbers of the spin traced density matrix (between 0 and 2) are compared against :math:`\eta`, whereas in Vayesta the eigenvalues of
       the spinned density-matrix (between 0 and 1) are used. The latter definition allows for a more natural comparison between spin-restricted
       and unrestricted calculations with the same value of :math:`\eta`.

