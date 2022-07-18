.. include:: /include/links.rst
.. _install:

============
Installation
============

Vayesta_ can be installed using pip_ or from source.

Installing with pip
===================

The simplest way to install Vayesta is to use the ``setup.py``:

.. code-block:: console

   [~]$ git clone https://github.com/BoothGroup/Vayesta
   [~]$ cd Vayesta
   [~]$ pip install .

All required python packages, such as NumPy_ and PySCF_ will be installed automatically.


Installation from Source
========================

.. note::
   Vayesta requires the following python packages when installing from source: NumPy_, SciPy_, PySCF_, and h5py_)

To install Vayesta from source, clone the GitHub repository and use ``cmake`` and ``make`` to compile:

.. code-block:: console

   [~]$ git clone https://github.com/BoothGroup/Vayesta .
   [~]$ cd Vayesta/vayesta/libs
   [~]$ mkdir build && cd build
   [~]$ cmake ..
   [~]$ make

To ensure that Vayesta is found by the Python interpreter when calling :python:`import vayesta`,
the installation location needs to be prepended to the ``PYTHONPATH`` environment variable as follows:

.. code-block:: console

    [~]$ export PYTHONPATH:"/path/to/vayesta":$PYTHONPATH

This command can be added to the ``~/.profile`` file located in the home dirctory, to ensure that it is excecuted for every new shell instance.

..
    For more detailed installation procedures and troubleshooting, please refer to the :ref:`faq`.

Running Tests
=============

After installation it is a good idea to run the test suite with ``pytest`` using the following command:

.. code-block:: console

   [~]$ pytest vayesta/tests
