.. include:: /include/links.rst

.. _sec_mpi:

Parallel Computing with MPI
===========================

.. warning::

    Not all functions have been tested in combinations with MPI.
    It is always adviced to perform a smaller test run, in order to verify that parallel and serial excecution yield the same results.
    Please open an issue on the `GitHub page <Vayesta_issues_>`_ to report any bugs or unexpected behavior.

Vayesta_ can construct and solve multiple quantum embedding problems (as defined by the fragmentation) in parallel,
using the `Message Passing Interface (MPI) <MPI_>`_ and the Python bindings provided by `mpi4py <MPI4PY_>`_.

.. note::

    ``mpi4py`` can be installed using pip: :console:`[~]$ pip install mpi4py`

Running an MPI Job
------------------

Running a calculation in parallel is as simple as excecuting :console:`[~]$ mpirun -np N jobscript.py`
in the console, where ``N`` is the desired number of MPI processes.
For the best possible parallelization, use as many MPI processes as there are fragments
(for example three for an atomic fragmentation of a water molecule).
However, for large systems it can be necessary to use fewer MPI processes than fragments---
the processes will then calculate their assigned set of embedding problems in serial.
It is never advised to use more MPI processes than there are fragments; the additional processes
will simply be idle.

Additional considerations
-------------------------

While any job script should in principle also work in parallel,
there are some additional considerations, which mainly concern file IO and logging.
They are demonstrated at this example, which can be found at ``vayeste/examples/ewf/molecules/90-mpi.py``:

.. literalinclude:: /../../examples/ewf/molecules/90-mpi.py
    :linenos:

* Vayesta will generate a separate logging file for each MPI rank, but PySCF_ does not. To avoid
  chaotic logging, it is adviced to give the ``mol`` object of each MPI process a unique output name (see **line 17**).
* PySCF does not support MPI by default. The mean-field calculation will thus simple be performed on each MPI process individually,
  and Vayesta will discard all solutions, except that obtained on the master process (rank 0).
  To save electricy, the  function ``vayesta.mpi.mpi.scf(mf)`` can be used to restrict the mean-field calculation to
  the master process from the beginning (see **line 22**).
* Output should only be printed on the master process.
  The property ``mpi.is_master`` (identical to ``mpi.rank == 0``) can be used to check if the current MPI process is the master process (see **line 30**)
