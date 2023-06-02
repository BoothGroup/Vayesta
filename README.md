Vayesta
=======

Vayesta is a Python package for performing correlated wave function-based quantum embedding in
*ab initio* molecules and solids, as well as lattice models.

* [Documentation](https://boothgroup.github.io/Vayesta/intro.html)
* [Quickstart](https://boothgroup.github.io/Vayesta/quickstart/index.html)
* [Changelog](../master/CHANGELOG)


Installation
------------

To install, clone the repository

```
git clone git@github.com:BoothGroup/Vayesta.git
```

Install the package using `pip` from the top-level directory, which requires CMake

```
python -m pip install . --user
```

To perform DMET calculations, leverage MPI parallelism, and to use [`ebcc`](https://github.com/BoothGroup/ebcc) solvers, optional dependencies must be installed. See the documentation for details.


Quickstart
----------

Examples of how to use Vayesta can be found in the `vayesta/examples` directory
and a quickstart guide can be found in the [documentation](https://boothgroup.github.io/Vayesta/quickstart/index.html).


Authors
-------

M. Nusspickel, O. J. Backhouse, B. Ibrahim, A. Santana-Bonilla, C. J. C. Scott, G. H. Booth


Citing Vayesta
--------------

The following papers should be cited in publications which make use of Vayesta:

[Max Nusspickel, Basil Ibrahim and George H. Booth, arXiv:2210.14561 (2023)](https://arxiv.org/abs/2210.14561).

[Max Nusspickel and George H. Booth, Phys. Rev. X 12, 011046 (2022)](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.12.011046).

Publication which utilize Extended Density-matrix Embedding Theory (EDMET) should also cite:

[Charles J. C. Scott and George H. Booth, Phys. Rev. B 104, 245114 (2021)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.104.245114).
