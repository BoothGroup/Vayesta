"""Density Matrix Embedding Theory (DMET) method
Author: Charles Scott
email:  cjcargillscott@gmail.com
"""

try:
    import cvxpy
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("CVXPY is required for DMET correlation potential fitting.")

from .dmet import DMET