.. include:: /include/links.rst

.. _fragments:

Defining Fragments
==================

Fragments are added within a fragmentation context manager,
which determines in which way orbitals are constructed for particular fragment choices.

Fragmentation
-------------

In order to define fragments in an *ab initio* system, one needs to specify
a set of orthogonal projectors onto subspaces.
There is no unique way to define such projectors---the question where one atom or orbital 'ends' and the next one begins
within a molecule or solid is fundamentally ill-posed.

Nevertheless, we do not require the projectors to be uniquely defined in order to perform quantum embedding calculations.
Within the embedded wave function (EWF) framework, the systematic improvability to the exact result is
guaranteed via an expansion of the bath space and appropriate choice of property functionals [1]_.
The only requirement on the fragments is that they are orthogonal and (when taken together) they span the complete *occupied* space.
For DMET calculations, it is additionally desirable for the union of the fragment spaces to span the entire *virtual* space.

.. note::

    To check if a user-defined fragmentation is orthogonal and complete in the occupied/virtual space,
    lock for the following line in the output:

    .. code-block:: console

        Fragmentation: orthogonal= True, occupied-complete= True, virtual-complete= False

For *ab initio* systems atomic projectors can be defined in terms of three different types of local atomic orbitals:

* `Intrinsic atomic orbitals (IAO) <REF_IAO_>`_
* IAOs augmented with `projected atomic orbitals (PAO) <REF_PAO_>`_
* Symmetrically (Löwdin) orthogonalized atomic orbitals (SAO)

For lattice models, on the other hand, the mapping between the site basis and the sites is clearer and more natural to define.

:numref:`table_fragmentation` shows a comparison of the the different fragmentation types:

.. _table_fragmentation:
.. list-table:: Comparison of fragmentation types
   :widths: 20 40 10 10 20
   :header-rows: 1

   * - Type
     - Context manager
     - DMET
     - EWF
     - Comments
   * - IAO
     - :python:`iao_fragmentation(minao='auto')`
     - No
     - Yes
     - Default for :python:`EWF`
   * - IAO+PAO
     - :python:`iaopao_fragmentation(minao='auto')`
     - Yes
     - Yes
     -
   * - SAO
     - :python:`sao_fragmentation()`
     - Yes
     - Yes
     -
   * - Site basis
     - :python:`site_fragmentation()`
     - Yes
     - Yes
     - For lattice models only

The minimal reference basis for the IAO construction can be selected with the :python:`minao`
argument. By default a suitable basis set will be chosen automatically :python:`'auto'`.

.. note::

    Generally different fragmentations should not be combined, as the resulting fragments are not guaranteed to be orthogonal.
    The only exception to this are the IAO and IAO+PAO fragmentation, which can be combined as long as no atom is added twice:

    .. code-block:: python

        with emb.iao_fragmentation() as f:
            f.add_atomic_fragment(0)
        with emb.iaopao_fragmentation() as f:
            f.add_atomic_fragment(1)


Adding Fragments
----------------

Within the fragmentation context manager, fragments can be added using the methods
:python:`add_atomic_fragment` and :python:`add_orbital_fragment`.

The :python:`add_atomic_fragment` method can accept both atom indices and symbols and can further filter specific orbital types
with the :python:`orbital_filter` argument.
The capabilities are best demonstrated in example ``examples/ewf/molecules/12-custom-fragments.py``:

.. literalinclude:: /../../examples/ewf/molecules/12-custom-fragments.py
    :linenos:

The :python:`add_orbital_fragment` method allows selecting orbitals (IAOs, PAOs, or SAOs)
to define fragments, either via their indices or labels.


.. [1] Exact here means free of any error due to the embedding procedure. Any external errors
       (for example by using a CCSD solver) remain.
