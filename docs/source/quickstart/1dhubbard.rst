.. include:: /include/links.rst
.. _1dhubbard:

Custom Hamiltonians
==========================

Model Hamiltonians are commonly used for simulating systems of interest both in Quantum Chemistry calculations as well as in Condensed-matter studies.
Vayesta_ enables the creation of customized Hamiltonians by employing the module `ref:lattmod` and the functions `ref:Hubbard_1d` and `ref:LatticeMF`
contained in this class.

To perform calculations using this feature, the required modules should be imported as shown in the following snippet:

.. literalinclude:: 1d_hubbard.py
   :lines: 1-3

The initial conditions for an user-defined Hubbard's model are declared as shown in the following lines of code:

.. literalinclude:: 1d_hubbard.py
   :lines: 5-7

The creation of the Lattice model is performed automatically by Vayesta_ using these commands:

.. literalinclude:: 1d_hubbard.py
   :lines: 5-7

where the function `ref:Hubbard1D` is specialized to create a 1-D Hubbard's model based on the user-defined variables such as *nsite*, *nelectron*, and
*hubbard_u*. An important feature of Vayesta_ is the way in which desired properties are printed and stored. This is done in a new folder created in the
current working directory and called **vayesta_output** where a *log_file* and *error_log* are placed. The user has the possibility to name the output
file provided in `ref:Hubbard1D` function.

The declared model can be further studied by using the module `ref:LatticeMF`, where a mean-field calculation can be carried out using the object created
in `ref:Hubbard1D` function as is shown below:

.. literalinclude:: 1d_hubbard.py
   :lines: 10-11

Likewise, the  `ref:LatticeMf` function will automatically select to carry out between **HF** or **UHF** mean-field methods based on the total spin
number.

The Vayesta_ embedding methods can be further used to numerically study these systems as introduced in the previous tutorials. In this sense, the
`ref:EWF` function employs the *mf* object created by `ref:LatticeMF` and complemented with the options *bno_threshold* and the option *fragment_type*
which indicates the use of **sites** instead of **atoms** as displayed in the following snipet:

.. literalinclude:: 1d_hubbard.py
   :lines: 14-17

The fragmentation procedure is carried out in the function `ref:site_fragmentation` and all the corresponding fragements are added utilizing the function
`ref:add_atomic_fragment`. At this point, the extension of the embedding can be declared as a first argument of this function, where the combination
**0** and **sym_factor=nsite** indicates a single-site embedding. The corresponding calculation is performed using the attribute **.kernel()**. The
energy per electron can be computed for both cases (i.e MF and EWF) as indicated below:

.. literalinclude:: 1d_hubbard.py
   :lines: 18-19

Due to the flexibility of the embedding methodology different combinations of site-size clusters can be explored. To do so, the function
'ref:add_atomic_fragment' should be correspondingly changed. As an example, a double-site embedding can be declared as:

.. literalinclude:: 1d_hubbard.py
   :lines: 24-25

where the arguments **[0,1]** and **sym_factor=nsite//2** indicates the use of a *dimerized* version of the 1-D Hubbard's model which is depicted in
**Figure(1)**.

.. figure:: figures/1dhubbfig.png
   :alt: aperiodic hubbard model
   :align: center
   :figclass: align-center

   **Figure(1)** Schematic depiction of the 1-D Hubbard model, half filling with double-site embedding fragmentation.


Finally, the results for this calculation can be obtained as indicated in the following snippet:

.. literalinclude:: 1d_hubbard.py
   :lines: 27-28
