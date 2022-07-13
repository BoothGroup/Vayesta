New Vayesta Documentation
==========================

This documentation has linked the vayesta source code with the documentation via sphinx-apidocs. It provides a new link called **Modules**, where all
classes and functions are listed. 


How to Build the HTML Docs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Documentation can be built going to the directory docs and typing:
   
.. code-block:: bash

    [~] cd docs
    [~] make html

The Documentation will be compiled. To find the html tree, one needs to get follow these steps:

.. code-block:: bash

    [~] cd build
    [~] cd html 
    [~] firefox intro.html
    
where firefox can be replaced by mozilla/safari or your preferred web-browser.


Building using sphinx-apidocs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The construction of the apidocs using the documentation of the code, can be performed in the following way:

1. One can construct the initial **source** and **build** folders via:

.. code-block:: bash

   [~] sphinx-quickstart 
   
In the first option, you can split the source and build folder. Once this is done, the conf.py file needs to be changed. 
This is done by opening the **conf.py** located in within the source directory and adding/changing the following 3 blocks
of code:

.. code-block:: python
   
   import os
   import sys  
   sys.path.insert(0, os.path.abspath('../vayesta/'))

.. code-block:: python

   extensions = ['sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc']
   exclude_patterns = ['**tests**']

.. code-block:: python
 
   html_theme = 'sphinx_rtd_theme'

Also, to correctly include the recently **modules.rst** file, one needs to declare it in the **index.rst** tree. This can
be done by replacing to this block of code:

.. code-block:: bash

  .. toctree::
     :maxdepth: 3
     :caption: Contents:

     intro
     install
     quickstart/index
     faq/index
     modules

This ensures that the **modules.rst** file is also part of the Documentation tree.

Finally, in the folder where the folder **source** is located, one can compile the Documentation by typing:

.. code-block:: bash
   
   [~] sphinx-apidocs -o source/ ../vayesta/ ../vayesta/tests/*
   [~] make clean
   [~] make html
   
The **sphinx-apidocs** will take some time to create all the necessary **.rst** docs. Afterwars, one can compile the documentation 
in the usual **make** way.



