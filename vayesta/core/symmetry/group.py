import numpy as np


class SymmetryGroup:
    """Detect symemtry group automatically (use spglib?)."""

    def __init__(self, mol, xtol=1e-8, check_basis=True, check_label=False):
        self.mol = mol
        self.xtol = xtol
        self.check_basis = check_basis
        self.check_label = check_label
        self.rotations = []
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

    def add_rotation(self, order, axis, center, unit='Ang'):
        """Add rotational symmetry."""
        axis = np.asarray(axis)
        center = np.asarray(center)
        self.rotations.append((order, axis, center, unit))

    def set_translation(self, translation):
        """Set translational symmetry."""
        self.translation = np.asarray(translation)
