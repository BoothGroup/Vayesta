import logging
import numpy as np


log = logging.getLogger(__name__)

class SymmetryGroup:
    """Detect symmetry group automatically (use spglib?)."""

    def __init__(self, mol, xtol=1e-8, check_basis=True, check_label=False):
        self.mol = mol
        self.xtol = xtol
        self.check_basis = check_basis
        self.check_label = check_label
        self.translation = None

    @property
    def natom(self):
        return self.mol.natm

    @property
    def nao(self):
        return self.mol.nao

    @property
    def dimension(self):
        return getattr(self.mol, 'dimension', 0)

    def compare_atoms(self, atom1, atom2, check_basis=None, check_label=None):
        """Compare atom symbol and (optionally) basis between atom1 and atom2."""
        if check_basis is None:
            check_basis = self.check_basis
        if check_label is None:
            check_label = self.check_label
        if check_label:
            type1 = self.mol.atom_symbol(atom1)
            type2 = self.mol.atom_symbol(atom2)
        else:
            type1 = self.mol.atom_pure_symbol(atom1)
            type2 = self.mol.atom_pure_symbol(atom2)
        if (type1 != type2):
            return False
        if not check_basis:
            return True
        bas1 = self.mol._basis[self.mol.atom_symbol(atom1)]
        bas2 = self.mol._basis[self.mol.atom_symbol(atom2)]
        return (bas1 == bas2)

    def get_closest_atom(self, coords):
        """pos in internal coordinates."""
        dists = np.linalg.norm(self.mol.atom_coords()-coords, axis=1)
        idx = np.argmin(dists)
        return idx, dists[idx]

    def add_rotation(self, order, axis, center, unit='ang'):
        log.critical(("The specification of rotational symmetry between fragments has changed."
                      " Check examples/ewf/73-rotational-symmetry.py for the new syntax."))
        raise NotImplementedError

    def set_translations(self, nimages):
        """Set translational symmetry.

        Parameters
        ----------
        nimages : array(3)
            Number of translationally symmetric images in the direction of the first, second,
            and third lattice vector.
        """
        self.translation = np.asarray(nimages)

    def clear_translations(self):
        self.translations = None
