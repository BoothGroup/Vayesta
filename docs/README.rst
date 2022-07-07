Vayesta Docs 
===================

This is the scaffold of the tutorials. To compile it, one needs only to type the following commands in the outer folder where the make files are.

.. code-block:: console

   [~]$ make clean
   [~]$ make html

The **html** results can be displayed via an internet browser (chrome, firefox, etc), following these lines of code

.. code-block:: console

   [~]$ cd build/html 
   [~]$ firefox index.html


Cloning this branch:
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console
   git clone -b tutorials --single-branch git@github.com:BoothGroup/Vayesta.git

Pushing into this branch:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

make changes, if you have used the -b tutorials then 

.. code-block:: console

   git remote show origin 

must show the following result

.. code-block:: console

   * remote origin
  Fetch URL: git@github.com:BoothGroup/Vayesta.git
  Push  URL: git@github.com:BoothGroup/Vayesta.git
  HEAD branch: master
  Remote branch:
    tutorials tracked
  Local branch configured for 'git pull':
    tutorials merges with remote tutorials
  Local ref configured for 'git push':
    tutorials pushes to tutorials (up-to-date)

If this is **true**, then you dont need to do anything else of defining origins. Then, to push changes, you just need to do

.. code-block:: console

    git status  
    git add --all -f  (sometimes the build directory is seen in the .gitignore)
    git pull
    git push

and the changes will be done in the tutorials branch.
