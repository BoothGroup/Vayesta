import logging
import itertools

import numpy as np
import scipy
import scipy.spatial

import pyscf
import pyscf.symm

import vayesta
import vayesta.core
from vayesta.core.util import *


log = logging.getLogger(__name__)

BOHR = 0.529177210903

class SymmetryOperation:

    def __init__(self, group):
        self.group = group
        log.debugv("Creating %s", self)

    @property
    def mol(self):
        return self.group.mol

    @property
    def xtol(self):
        return self.group.xtol

    @property
    def natom(self):
        return self.group.natom

    @property
    def nao(self):
        return self.group.nao

    def __call__(self):
        raise AbstractMethodError

    def apply_to_point(self, r0):
        raise AbstractMethodError

    def get_atom_reorder(self):
        """Reordering of atoms for a given rotation.

        Parameters
        ----------

        Returns
        -------
        reorder: list
        inverse: list
        """
        reorder = np.full((self.natom,), -1, dtype=int)
        inverse = np.full((self.natom,), -1, dtype=int)

        def assign():
            success = True
            for atom0, r0 in enumerate(self.mol.atom_coords()):
                r1 = self.apply_to_point(r0)
                atom1, dist = self.group.get_closest_atom(r1)
                if dist > self.xtol:
                    log.error("No symmetry related atom found for atom %d. Closest atom is %d with distance %.3e a.u.",
                              atom0, atom1, dist)
                    success = False
                elif not self.group.compare_atoms(atom0, atom1):
                    log.error("Atom %d is not symmetry related to atom %d.", atom1, atom0)
                    success = False
                else:
                    log.debug("Atom %d is symmetry related to atom %d.", atom1, atom0)
                    reorder[atom1] = atom0
                    inverse[atom0] = atom1
            return success

        if not assign():
            return None, None

        assert (not np.any(reorder == -1))
        assert (not np.any(inverse == -1))
        assert np.all(np.arange(self.natom)[reorder][inverse] == np.arange(self.natom))
        return reorder, inverse

    def get_ao_reorder(self, atom_reorder):
        if atom_reorder is None:
            return None, None
        aoslice = self.mol.aoslice_by_atom()[:,2:]
        reorder = np.full((self.mol.nao,), -1)
        inverse = np.full((self.mol.nao,), -1)
        for atom0 in range(self.natom):
            atom1 = atom_reorder[atom0]
            aos0 = list(range(aoslice[atom0,0], aoslice[atom0,1]))
            aos1 = list(range(aoslice[atom1,0], aoslice[atom1,1]))
            reorder[aos0[0]:aos0[-1]+1] = aos1
            inverse[aos1[0]:aos1[-1]+1] = aos0
        assert not np.any(reorder == -1)
        assert not np.any(inverse == -1)
        assert np.all(np.arange(self.nao)[reorder][inverse] == np.arange(self.nao))
        return reorder, inverse

    def rotate_angular_orbitals(self, a, rotmats):
        """Rotate between orbitals in p,d,f,... shells."""
        ao_loc = self.mol.ao_loc
        ao_start = ao_loc[0]
        b = a.copy()
        for bas, ao_end in enumerate(ao_loc[1:]):
            l = self.mol.bas_angular(bas)
            # s orbitals do not require rotation:
            if l == 0:
                ao_start = ao_end
                continue
            #rot = self.angular_rotmats[l]
            rot = rotmats[l]
            size = ao_end - ao_start
            assert (rot.shape == (size, size))
            slc = np.s_[ao_start:ao_end]
            #log.debugv("bas= %d, ao_start= %d, ao_end= %d, l= %d, diag(rot)= %r",
            #    bas, ao_start, ao_end, l, np.diag(rot))
            b[slc] = einsum('x...,xy->y...', a[slc], rot)
            ao_start = ao_end
        return b


class SymmetryIdentity(SymmetryOperation):

    def __repr__(self):
        return "Identity"

    def __call__(self, a, **kwargs):
        return a

    def apply_to_point(self, r0):
        return r0

    def get_atom_reorder(self):
        reorder = list(range(self.mol.natm))
        return reorder, reorder


class SymmetryInversion(SymmetryOperation):

    def __init__(self, group, center=(0,0,0)):
        if not np.all(np.asarray(center) == 0):
            raise NotImplementedError
        self.center = center
        super().__init__(group)

        self.atom_reorder = self.get_atom_reorder()[0]
        if self.atom_reorder is None:
            raise RuntimeError("Symmetry %s not found" % self)
        self.ao_reorder = self.get_ao_reorder(self.atom_reorder)[0]

    def __repr__(self):
        return "Inversion(%g,%g,%g)" % tuple(self.center)

    def apply_to_point(self, r0):
        return -r0

    def __call__(self, a, axis=0):
        if hasattr(axis, '__len__'):
            for ax in axis:
                a = self(a, axis=ax)
            return a
        if isinstance(a, (tuple, list)):
            return tuple([self(x, axis=axis) for x in a])
        a = np.moveaxis(a, axis, 0)
        # Reorder AOs according to new atomic center
        a = a[self.ao_reorder]
        # Invert angular momentum in shells with l=1,3,5,... (p,f,h,...):
        rotmats = [(-1)**i * np.eye(n) for (i, n) in enumerate(range(1,19,2))]
        a = self.rotate_angular_orbitals(a, rotmats)
        a = np.moveaxis(a, 0, axis)
        return a


class SymmetryRotation(SymmetryOperation):

    def __init__(self, group, rotvec, center=(0,0,0), unit='Bohr'):
        self.rotvec = np.asarray(rotvec, dtype=float)
        self.center = np.asarray(center, dtype=float)
        if unit.lower().startswith('ang'):
            self.center = self.center/BOHR
        super().__init__(group)

        self.atom_reorder = self.get_atom_reorder()[0]
        if self.atom_reorder is None:
            raise RuntimeError("Symmetry %s not found" % self)
        self.ao_reorder = self.get_ao_reorder(self.atom_reorder)[0]

        self.angular_rotmats = pyscf.symm.basis._ao_rotation_matrices(self.mol, self.as_matrix())

    def __repr__(self):
        return "Rotation(%g,%g,%g)" % tuple(self.rotvec)

    def as_matrix(self):
        return scipy.spatial.transform.Rotation.from_rotvec(self.rotvec).as_matrix()

    def apply_to_point(self, r0):
        rot = self.as_matrix()
        return np.dot(rot, (r0 - self.center)) + self.center

    def __call__(self, a, axis=0):
        if hasattr(axis, '__len__'):
            for ax in axis:
                a = self(a, axis=ax)
            return a
        if isinstance(a, (tuple, list)):
            return tuple([self(x, axis=axis) for x in a])
        a = np.moveaxis(a, axis, 0)
        # Reorder AOs according to new atomic center
        a = a[self.ao_reorder]
        # Rotate between orbitals in p,d,f,... shells
        a = self.rotate_angular_orbitals(a, self.angular_rotmats)
        a = np.moveaxis(a, 0, axis)
        return a

class SymmetryTranslation(SymmetryOperation):

    def __init__(self, group, vector, boundary=None, atom_reorder=None, ao_reorder=None):
        self.vector = np.asarray(vector)
        super().__init__(group)

        if boundary is None:
            boundary = getattr(self.mol, 'boundary', 'PBC')
        if np.ndim(boundary) == 0:
            boundary = 3*[boundary]
        elif np.ndim(boundary) == 1 and len(boundary) == 2:
            boundary = [boundary[0], boundary[1], 'PBC']
        self.boundary = boundary

        # Atom reorder
        self.atom_reorder_phases = None
        self.ao_reorder_phases = None
        if atom_reorder is None:
            atom_reorder, _, self.atom_reorder_phases = self.get_atom_reorder()
        self.atom_reorder = atom_reorder
        assert (self.atom_reorder is not None)
        # AO reorder
        if ao_reorder is None:
            ao_reorder, _, self.ao_reorder_phases = self.get_ao_reorder()
        self.ao_reorder = ao_reorder
        assert (self.ao_reorder is not None)

    def __repr__(self):
        return "Translation(%f,%f,%f)" % tuple(self.vector)

    def __call__(self, a, axis=0):
        """Apply symmetry operation along AO axis."""
        if hasattr(axis, '__len__'):
            for ax in axis:
                a = self(a, axis=ax)
            return a
        if isinstance(a, (tuple, list)):
            return tuple([self(x, axis=axis) for x in a])
        if self.ao_reorder_phases is None:
            return np.take(a, self.ao_reorder, axis=axis)
        bc = tuple(axis*[None] + [slice(None, None, None)] + (a.ndim-axis-1)*[None])
        return np.take(a, self.ao_reorder, axis=axis) * self.ao_reorder_phases[bc]

    def inverse(self):
        return type(self)(self.mol, -self.vector, boundary=self.boundary, atom_reorder=np.argsort(self.atom_reorder))

    @property
    def lattice_vectors(self):
        return self.mol.lattice_vectors()

    @property
    def inv_lattice_vectors(self):
        return np.linalg.inv(self.lattice_vectors)

    @property
    def boundary_phases(self):
        return np.asarray([1 if (b.lower() == 'pbc') else -1 for b in self.boundary])

    @property
    def vector_xyz(self):
        """Translation vector in real-space coordinates (unit = Bohr)."""
        latvecs = self.lattice_vectors
        vector_xyz = np.dot(self.lattice_vectors, self.vector)
        return vector_xyz

    @property
    def inverse_atom_reorder(self):
        if self.atom_reorder is None:
            return None
        return np.argsort(self.atom_reorder)

    def get_atom_reorder(self):
        """Reordering of atoms for a given translation.

        Parameters
        ----------

        Returns
        -------
        reorder: list
        inverse: list
        phases: list
        """
        atom_coords_abc = np.dot(self.mol.atom_coords(), self.inv_lattice_vectors)

        def get_atom_at(pos):
            """pos in internal coordinates."""
            for dx, dy, dz in itertools.product([0,-1,1], repeat=3):
                if self.group.dimension in (1, 2) and (dz != 0): continue
                if self.group.dimension == 1 and (dy != 0): continue
                dr = np.asarray([dx, dy, dz])
                phase = np.product(self.boundary_phases[dr!=0])
                dists = np.linalg.norm(atom_coords_abc + dr - pos, axis=1)
                idx = np.argmin(dists)
                if (dists[idx] < self.xtol):
                    return idx, phase
            return None, None

        reorder = np.full((self.natom,), -1)
        inverse = np.full((self.natom,), -1)
        phases = np.full((self.natom,), 0)
        for atom0, coords0 in enumerate(atom_coords_abc):
            atom1, phase = get_atom_at(coords0 + self.vector)
            if atom1 is None:
                return None, None, None
            if not self.group.compare_atoms(atom0, atom1):
                return None, None, None
            reorder[atom1] = atom0
            inverse[atom0] = atom1
            phases[atom0] = phase
        assert (not np.any(reorder == -1))
        assert (not np.any(inverse == -1))
        assert (not np.any(phases == 0))
        assert np.all(np.arange(self.natom)[reorder][inverse] == np.arange(self.natom))
        return reorder, inverse, phases

    def get_ao_reorder(self, atom_reorder=None, atom_reorder_phases=None):
        if atom_reorder is None:
            atom_reorder = self.atom_reorder
        if atom_reorder_phases is None:
            atom_reorder_phases = self.atom_reorder_phases
        if atom_reorder is None:
            return None, None, None
        aoslice = self.mol.aoslice_by_atom()[:,2:]
        reorder = np.full((self.mol.nao,), -1)
        inverse = np.full((self.mol.nao,), -1)
        if atom_reorder_phases is not None:
            phases = np.full((self.mol.nao,), 0)
        else:
            phases = None
        for atom0 in range(self.natom):
            atom1 = atom_reorder[atom0]
            aos0 = list(range(aoslice[atom0,0], aoslice[atom0,1]))
            aos1 = list(range(aoslice[atom1,0], aoslice[atom1,1]))
            reorder[aos0[0]:aos0[-1]+1] = aos1
            inverse[aos1[0]:aos1[-1]+1] = aos0
            if atom_reorder_phases is not None:
                phases[aos0[0]:aos0[-1]+1] =  atom_reorder_phases[atom1]
        assert not np.any(reorder == -1)
        assert not np.any(inverse == -1)
        if atom_reorder_phases is not None:
            assert not np.any(phases == 0)
        assert np.all(np.arange(self.nao)[reorder][inverse] == np.arange(self.nao))
        return reorder, inverse, phases

if __name__ == '__main__':

    def test_translation():
        import pyscf
        import pyscf.pbc
        import pyscf.pbc.gto
        import pyscf.pbc.scf
        import pyscf.pbc.tools
        import pyscf.pbc.df

        cell = pyscf.pbc.gto.Cell()
        cell.a = 3*np.eye(3)
        cell.atom = 'He 0 0 0'
        cell.unit = 'Bohr'
        cell.basis = 'def2-svp'
        #cell.basis = 'sto-3g'
        cell.build()
        #cell.dimension = 2

        sc = [1,1,2]
        cell = pyscf.pbc.tools.super_cell(cell, sc)

        t = Translation(cell, [0, 0, 1/2])

        df = pyscf.pbc.df.GDF(cell)
        df.auxbasis = 'def2-svp-ri'
        df.build()

        #print(t.ao_reorder)
        #aux_reorder = t.get_ao_reorder(cell=df.auxcell)[0]
        #print(aux_reorder)
        taux = t.change_cell(df.auxcell)
        print(t.ao_reorder)
        print(taux.ao_reorder)

        #print(trans.atom_reorder)
        #print(trans.ao_reorder)
        #trans = Translation(cell, [0,1/5,2/3])
        #trans = Translation(cell, [0,3/5,2/3])

        #mo0 = np.eye(cell.nao)
        #mo1 = trans(mo0)

    def test_rotation():
        import pyscf
        import pyscf.gto
        import vayesta.misc
        import vayesta.misc.molecules

        mol = pyscf.gto.Mole()
        mol.atom = vayesta.misc.molecules.arene(6)
        mol.build()

        vec = np.asarray([0, 0, 1])
        op = SymmetryRotation(mol, 6, vec)

        reorder, inv = op.get_atom_reorder()
        print(reorder)
        print(inv)



    test_rotation()
