.. include:: /include/links.rst

Installing OpenBLAS from source
-------------------------------

The following is an example of installation of OpenBLAS, which can be important for
multithreaded solutions to the embedding problems and other linear algebra operations required.

This installation example is carried out using **gcc** compilers.

.. note::

   The installation of OpenBLAS_ should be done locally.

To install a local version, a specific directory should be created where the library is stored:

.. code-block:: console

   [~]$ mkdir $HOME/work
   [~]$ export WORK=$HOME/work


The last line of code creates an enviromental variable called WORK, which will store the absolute path declared for the directory `work`.

To install an **openmp** version of OpenBLAS_, a series of steps need to be followed. Firstly, specific options should be selected as is shown in the following lines of code:

.. code-block:: console

   [~]$ export USE_OPENMP=1
   [~]$ export NO_WARMUP=1
   [~]$ export BUILD_RELAPACK=0
   [~]$ export DYNAMIC_ARCH=0
   [~]$ export CC=gcc
   [~]$ export FC=gfortran
   [~]$ export HOSTCC=gcc


The last 3 options ensures the use of **GNU** compilers. Similarly, a set of optimized `FLAGS` is defined in the following way:

.. code-block:: console

   [~]$ export COMMON_OPT="-O3 -ftree-vectorize -fprefetch-loop-arrays --param prefetch-latency=300"
   [~]$ export CFLAGS="-O3 -ftree-vectorize -fprefetch-loop-arrays --param prefetch-latency=300"
   [~]$ export FCOMMON_OPT="-O3 -ftree-vectorize -fprefetch-loop-arrays --param prefetch-latency=300"
   [~]$ export FCFLAGS="-O3 -ftree-vectorize -fprefetch-loop-arrays --param prefetch-latency=300"

To install OpenBLAS, the following commands should be used:

.. code-block:: console

   [~]$ OPENBLAS_DIR=$HOME/work/openblas
   [~]$ make -j4 BINARY=64 INTERFACE=64 LIBNAMESUFFIX=openmp
   [~]$ make PREFIX=$OPENBLAS_DIR LIBNAMESUFFIX=openmp install

This concludes the installation of OpenBLAS. The library can be found in the path `$HOME/work/openblas`.
