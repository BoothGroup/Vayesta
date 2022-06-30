.. _faq:

=======
FAQ
=======

Installation 
===============

Installing Vayesta_ from source can be also carried out using many different options. In this section, we will review different manners of installing Vayesta_ using OpenBLAS_ existing version in HPC facilities. 


Cluster Installation
^^^^^^^^^^^^^^^^^^^^

Vayesta uses OpenBLAS_ for fast numerical operations. If you are a cluster user where the command `module` is available, it is very likely OpenBLAS_ is already installed. To locate the library, the following command can be used: 

.. code-block:: console

   [~]$ module show openblas


which will print the path to the library, the compiler used and more flags used during the installation process. If there are more than one version of the library, one can use the command:

.. code-block:: console

   [~]$ module available

to print all modules that are installed in the cluster. Once you have found the desired version of OpenBLAS_, you can enable it by typing:

.. code-block:: console

   [~]$ module load openblas/VERSION-openmp/

which will make it accessible for **CMake**. Similarly, the library **hdf5** can be enabled by using the procedure shown above for OpenBLAS_. We strongly recomend to use a **hdf5-parallel** for better performance.


Installing OpenBLAS
--------------------

The installation can be carried out using different compilers. To do so, we will
perform the installation using **gcc** or **amd**.

.. note::

   The installation of OpenBLAS_ should be done locally, i.e., within your `$HOME`  directory.  

To install a local (i.e in your `$HOME`) version, one needs to define a specific directory as is shown in the following lines of code:

.. code-block:: console

   [~]$ mkdir $HOME/work
   [~]$ export WORK=$HOME/work


This creates a enviromental variable called WORK, which will store the absolute path declared for the directory `work`.

To install an **openmp** version of OpenBLAS_, a series of steps need to be followed. Firstly, specific options should be selected which implies the use of a threaded version (first option), as is shown in the following lines of code:

.. code-block:: console

   [~]$ export USE_OPENMP=1
   [~]$ export NO_WARMUP=1
   [~]$ export BUILD_RELAPACK=0
   [~]$ export DYNAMIC_ARCH=0
   [~]$ export CC=gcc
   [~]$ export FC=gfortran
   [~]$ export HOSTCC=gcc


The last 3 options ensures the use of **GNU** compilers. This can be changed to **Intel** compilers (i.e icc and ifort). Similarly, a set of optimized `FLAGS` is defined in the following way:

.. code-block:: console

   [~]$ export COMMON_OPT="-O3 -ftree-vectorize -fprefetch-loop-arrays --param prefetch-latency=300"
   [~]$ export CFLAGS="-O3 -ftree-vectorize -fprefetch-loop-arrays --param prefetch-latency=300"
   [~]$ export FCOMMON_OPT="-O3 -ftree-vectorize -fprefetch-loop-arrays --param prefetch-latency=300"
   [~]$ export FCFLAGS="-O3 -ftree-vectorize -fprefetch-loop-arrays --param prefetch-latency=300"

To install OpenBLAS_, the following commands should be used:

.. code-block:: console
		
   [~]$ make -j4 BINARY=64 INTERFACE=64 LIBNAMESUFFIX=openmp
   [~]$ make PREFIX=$OPENBLAS_DIR LIBNAMESUFFIX=openmp install

This concludes the installation of OpenBLAS_. The library can be found in the path `$HOME/work/openblas`.
   
Python3 environment
--------------------

As a cluster user, Python_ is also provided as a part of your initial environment. It is important to have an updated version of Python_, since many High-Performance Cluster facilities declares as a dafault the version 2.7, whilst Vayesta_ requires a 3+ version. To check the Python version, one can simply type:

.. code-block:: console

   [~]$ python

This will call python and display the following message:

.. code-block:: console

   Python 2.7.5 (default, Aug 13 2020, 02:51:10) 
   [GCC 4.8.5 20150623 (Red Hat 4.8.5-39)] on linux2
   Type "help", "copyright", "credits" or "license" for more information.
   >>> 

This indicates that the default version is 2.7.5. In this case, one needs to search for the correct Python_ version, that can be done using the module command as indicated in the following lines of code:

.. code-block:: console

   [~]$ module available
   [~]$ module load pythonX.X

where **X.X** indicates the version that has been chosen. We strongly suggest tahe latest available version of Python_ available in your cluster.

An important point is to configure the command **pip** to point to the directory `$HOME/work`. This can be done by typing the following lines of code:

.. code-block:: console

   [~]$ export PYTHONUSERBASE=${WORK}/.local
   [~]$ export PATH=$PYTHONUSERBASE/bin:$PATH
   [~]$ export PYTHONPATH=$PYTHONUSERBASE/lib/pythonX.X/site-packages:$PYTHONPATH

This ensures that the future installations will be stored in this directory.


Installing mpi4py
------------------

To install **mpi4py**, the following command is used to build the library:

.. code-block:: bash

   env MPICC=/../mpicc python -m pip install --force --user mpi4py

This ensures the creation of the library locally.

Installing PySCF
----------------

Once the previous steps have been sucessfully carried out, PySCF_ can be installed. The following steps will provide guidance for this process:

.. code-block:: console
		
   [~]$ git... 


This provides a local version of PySCF_ which can be linked to Vayesta_.


.. _GitHub: https://github.com/
.. _OpenBLAS: https://github.com/xianyi/OpenBLAS
.. _Vayesta: https://vayesta.com
.. _Python: https://www.python.org/
.. _PySCF: https://pyscf.org/
.. _pip: https://pypi.org/project/pip/
