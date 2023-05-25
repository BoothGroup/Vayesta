import logging
import itertools
import numpy as np
import scipy
import scipy.spatial
import pyscf
import pyscf.symm
import vayesta
import vayesta.core
from vayesta.core.util import AbstractMethodError, einsum


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

    def __call__(self, a, *args, axis=0, **kwargs):
        return self.call_wrapper(a, *args, axis=axis, **kwargs)

    def call_wrapper(self, a, *args, axis=0, **kwargs):
        """Common pre- and post-processing for all symmetries.

        Symmetry specific processing is performed in call_kernel."""
        if hasattr(axis, '__len__'):
            for ax in axis:
                a = self(a, *args, axis=ax, **kwargs)
            return a
        if isinstance(a, (tuple, list)):
            return tuple([self(x, *args, axis=axis, **kwargs) for x in a])
        a = np.moveaxis(a, axis, 0)
        # Reorder AOs according to new atomic center
        a = a[self.ao_reorder]
        a = self.call_kernel(a, *args, **kwargs)
        a = np.moveaxis(a, 0, axis)
        return a

    def call_kernel(self, *args, **kwargs):
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
            rot = rotmats[l]
            size = ao_end - ao_start

            # It is possible that multiple shells are contained in a single 'bas'!
            nl = rot.shape[0]
            assert (size % nl == 0)
            for shell0 in range(0, size, nl):
                shell = np.s_[ao_start+shell0:ao_start+shell0+nl]
                b[shell] = einsum('x...,xy->y...', a[shell], rot)
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
        center = np.asarray(center, dtype=float)
        self.center = center

        super().__init__(group)

        self.atom_reorder = self.get_atom_reorder()[0]
        if self.atom_reorder is None:
            raise RuntimeError("Symmetry %s not found" % self)
        self.ao_reorder = self.get_ao_reorder(self.atom_reorder)[0]

    def __repr__(self):
        return "Inversion(%g,%g,%g)" % tuple(self.center)

    def apply_to_point(self, r0):
        return 2*self.center - r0

    def call_kernel(self, a):
        rotmats = [(-1)**i * np.eye(n) for (i, n) in enumerate(range(1,19,2))]
        a = self.rotate_angular_orbitals(a, rotmats)
        return a


class SymmetryReflection(SymmetryOperation):

    def __init__(self, group, axis, center=(0,0,0)):
        center = np.asarray(center, dtype=float)
        self.center = center
        self.axis = np.asarray(axis)/np.linalg.norm(axis)
        super().__init__(group)

        self.atom_reorder = self.get_atom_reorder()[0]
        if self.atom_reorder is None:
            raise RuntimeError("Symmetry %s not found" % self)
        self.ao_reorder = self.get_ao_reorder(self.atom_reorder)[0]

        # A reflection can be decomposed into a C2-rotation + inversion
        # We use this to derive the angular transformation matrix:
        rot = scipy.spatial.transform.Rotation.from_rotvec(self.axis*np.pi).as_matrix()
        try:
            angular_rotmats = pyscf.symm.basis._momentum_rotation_matrices(self.mol, rot)
        except AttributeError:
            angular_rotmats = pyscf.symm.basis._ao_rotation_matrices(self.mol, rot)
        # Inversion of p,f,h,... shells:
        self.angular_rotmats =[(-1)**i * x for (i, x) in enumerate(angular_rotmats)]

    def __repr__(self):
        return "Reflection(%g,%g,%g)" % tuple(self.axis)

    def as_matrix(self):
        """Householder matrix. Does not account for shifted origin!"""
        return np.eye(3) - 2*np.outer(self.axis, self.axis)

    def apply_to_point(self, r0):
        """Householder transformation."""
        r1 = r0 - 2*np.dot(np.outer(self.axis, self.axis), r0-self.center)
        return r1

    def call_kernel(self, a):
        a = self.rotate_angular_orbitals(a, self.angular_rotmats)
        return a


class SymmetryRotation(SymmetryOperation):

    def __init__(self, group, rotvec, center=(0,0,0)):
        self.rotvec = np.asarray(rotvec, dtype=float)
        self.center = np.asarray(center, dtype=float)
        super().__init__(group)

        self.atom_reorder = self.get_atom_reorder()[0]
        if self.atom_reorder is None:
            raise RuntimeError("Symmetry %s not found" % self)
        self.ao_reorder = self.get_ao_reorder(self.atom_reorder)[0]
        try:
            self.angular_rotmats = pyscf.symm.basis._momentum_rotation_matrices(self.mol, self.as_matrix())
        except AttributeError:
            self.angular_rotmats = pyscf.symm.basis._ao_rotation_matrices(self.mol, self.as_matrix())

    def __repr__(self):
        return "Rotation(%g,%g,%g)" % tuple(self.rotvec)

    def as_matrix(self):
        return scipy.spatial.transform.Rotation.from_rotvec(self.rotvec).as_matrix()

    def apply_to_point(self, r0):
        rot = self.as_matrix()
        return np.dot(rot, (r0 - self.center)) + self.center

    def call_kernel(self, a):
        a = self.rotate_angular_orbitals(a, self.angular_rotmats)
        return a


class SymmetryTranslation(SymmetryOperation):

    def __init__(self, group, vector, boundary=None, atom_reorder=None, ao_reorder=None):
        self.vector = np.asarray(vector, dtype=float)
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

    def call_kernel(self, a):
        if self.ao_reorder_phases is None:
            return a
        return a * self.ao_reorder_phases[tuple([np.s_[:]] + (a.ndim-1)*[None])]

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
