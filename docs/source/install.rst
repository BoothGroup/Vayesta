.. include:: /include/links.rst
.. _install:

============
Installation
============

Vayesta_ can be installed using pip_ for a quickstart or also installed
from source. This section will present instructions in how to build
Vayesta_.


Using pip
==========

The simplest way to install Vayesta_ is to use pip_ :

.. code-block:: console

   pip install vayesta


Installation from source
==========================

As an alternatie to pip_, Vayesta_ can be cloned from its GitHub repository. This can be done by creating a new folder in your `$HOME` directory in which Vayesta_ is intended to be installed. Once this is done, the following lines of code should create Vayesta_

.. code-block:: console

   git clone https://github.com/BoothGroup/Vayesta ./git/vayesta
   cd git/vayesta/vayesta/libs
   mkdir build && cd build
   cmake ..
   make

.. note::

   A compiled version of OpenBLAS_ should be accessible in your environment where the libopenblas_XXX.so can be found.


For a more detailed installation procedures, the user can refer to the
:ref:`faq`. section.

Testing Vayesta
====================

After installation, Vayesta_ can be tested by typing the following command:

.. code-block:: console

   python setup.py test

which suffices for testing the main computational capabilities of the code.
