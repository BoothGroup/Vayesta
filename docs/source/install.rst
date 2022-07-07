.. include:: /include/links.rst
.. _install:

============
Installation
============

Vayesta_ can be installed using pip_ or from source.

Installing with pip
===================

The simplest way to install Vayesta_ is to use pip_:

.. code-block:: console

   pip install vayesta

All required python packages, such as NumPy_ and PySCF_ will be installed automatically.


Installation from Source
========================

To install Vayesta_ from source, clone the GitHub repository and use ``cmake`` and ``make`` to compile:

.. code-block:: console

   git clone https://github.com/BoothGroup/Vayesta .
   cd Vayesta/vayesta/libs
   mkdir build && cd build
   cmake ..
   make

.. note::
    When installing from source, make sure that PySCF_ and its dependencies are installed first.


For more detailed installation procedures and troubleshooting, please refer to the :ref:`faq`.

Running Tests
=============

After installation, Vayesta_ can run the test using the following command:

.. code-block:: console

   pytest vayesta/tests
