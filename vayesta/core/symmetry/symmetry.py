import logging

import numpy as np
from vayesta.core.symmetry import tsymmetry


log = logging.getLogger(__name__)


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

       >>> angle_between((1, 0, 0), (0, 1, 0))
       1.5707963267948966
       >>> angle_between((1, 0, 0), (1, 0, 0))
       0.0
       >>> angle_between((1, 0, 0), (-1, 0, 0))
       3.141592653589793
    """
    u1 = unit_vector(v1)
    u2 = unit_vector(v2)
    return np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))

class Symmetry:

    def __init__(self, mf, log=log):
        self.mf = mf
        self.nsubcells = None
        self.log = log

        if self.has_pbc:
            #self.nsubcells = self.find_subcells(respect_dm1=self.mf.make_rdm1())
            # MF is FoldedSCF
            if hasattr(mf, 'nsubcells'):
                self.nsubcells = mf.nsubcells
                self.log.info("Found %d x %d x %d primitive subcells in unit cell", *self.nsubcells)

    @property
    def mol(self):
        return self.mf.mol

    @property
    def cell(self):
        if (self.pbcndims == 0):
            raise AttributeError
        return self.mol

    @property
    def natom(self):
        return self.mol.natm

    @property
    def natom_unique(self):
        mf = getattr(self.mf, 'kmf', self.mf)
        return mf.mol.natm

    def get_unique_atoms(self): #, r_tol=1e-5):
        return list(range(self.natom_unique))

        #if self.nsubcells is None:
        #    return list(range(self.natom))
        #avecs = self.primitive_lattice_vectors()
        #bvecs = np.linalg.inv(avecs.T)
        #atoms = []
        #for atom, coords in enumerate(self.atom_coords()):
        #    coords = np.dot(coords, bvecs.T)
        #    if np.any(coords >= (1.0-r_tol)):
        #        continue
        #    atoms.append(atom)
        #return atoms

    def primitive_lattice_vectors(self):
        return self.lattice_vectors() / self.nsubcells[:,None]

    def atom_coords(self, unit='Bohr'):
        return self.mol.atom_coords(unit=unit)

    def lattice_vectors(self):
        return self.cell.lattice_vectors()

    @property
    def pbcndims(self):
        """Number of periodic boundary conditions: 0 = No PBC, 1 = 1D 2 = 2D, 3 = 3D."""
        self.mol.lattice_vectors()
        try:
            self.mol.lattice_vectors()
        except AttributeError:
            return 0
        return self.mol.dimension

    @property
    def has_pbc(self):
        return (self.pbcndims > 0)

    def compare_atoms(self, atom1, atom2, respect_labels=False, respect_basis=True):
        """Compare atom symbol and (optionally) basis between atom1 and atom2."""

        if respect_labels:
            type1 = self.mol.atom_symbol(atom1)
            type2 = self.mol.atom_symbol(atom2)
        else:
            type1 = self.mol.atom_pure_symbol(atom1)
            type2 = self.mol.atom_pure_symbol(atom2)

        if type1 != type2:
            return False
        if not respect_basis:
            return True
        bas1 = self.mol._basis[self.mol.atom_symbol(atom1)]
        bas2 = self.mol._basis[self.mol.atom_symbol(atom2)]
        if bas1 != bas2:
            return False
        return True


    def find_subcells(self, respect_basis=True, respect_labels=False, respect_dm1=None, r_tol=1e-5, dm1_tol=1e-6):
        """Find subcells within cell, with unit cell vectors parallel to the supercell.

        Parameters
        ----------
        respect_basis: bool, optional
            If True, the basis functions are considered when determining the symmetry. Default: True.
        respect_labels: bool, optional
            If True, the labels of atoms (such as "H1" or "C*") are considered when determining the symmetry. Default: False.
        respect_dm1: array or None, optional
            If a (tuple of) density-matrix is passed, it is considered when determining the symmetry. Default: None.
        r_tol: float, optional
            Real space tolerance to determine if two atoms are symmetry equivalent. Default: 1e-5.
        dm1_tol: float, optional
            Density-matrix tolerance to determine the symmetry. Default: 1e-6.

        Returns
        -------
        nsubcells: tuple(3)
            Number of primitive subcells in (a0, a1, a2) direction.
        """
        avecs = self.lattice_vectors()
        bvecs = np.linalg.inv(avecs.T)
        coords = self.atom_coords()

        if respect_dm1 is not None:
            # RHF
            if np.ndim(respect_dm1[0]) == 1:
                dm1s = [respect_dm1]
                dm1_tol *= 2
            # UHF
            else:
                dm1s = respect_dm1

        coords_internal = np.einsum('ar,br->ab', coords, bvecs)

        checked = []    # Keep track of vectors which were already checked, to improve performance
        dvecs = [[] for i in range(3)]
        for atm1 in range(self.natom):
            for atm2 in range(atm1+1, self.natom):

                # 1) Compare atom symbol and basis
                equal = self.compare_atoms(atm1, atm2, respect_basis=respect_basis, respect_labels=False)
                if not equal:
                    continue

                pos1 = coords_internal[atm1]
                pos2 = coords_internal[atm2]
                dvec = (pos2 - pos1)

                # 3) Check parallel to one and only lattice vector
                dim = np.argwhere(abs(dvec) > r_tol)
                if len(dim) != 1:
                    continue
                dim = dim[0][0]

                # Check if vector has been checked before
                if checked:
                    diff = (dvec[None] - np.asarray(checked))
                    checked_before =  np.allclose(np.amin(abs(diff), axis=0), 0)
                    if checked_before:
                        continue
                checked.append(dvec)

                # 4) Check if dvec already in dvecs or an integer multiple of a vector already in dvecs
                if dvecs[dim]:
                    found = False
                    for i in range(1, 100):
                        diff = np.linalg.norm(dvec[None]/i - dvecs[dim], axis=1)
                        if np.any(diff < r_tol):
                            found = True
                            break
                    if found:
                        continue

                # 5) Check if dvec is valid symmetry for all atoms:
                tvec = np.dot(dvec, avecs)
                reorder, _, phases = tsymmetry.reorder_atoms(self.cell, tvec, unit='Bohr')
                if reorder is None:
                    continue

                # Check if mean-field is symmetric
                if respect_dm1 is not None:
                    reorder, _, phases = tsymmetry.reorder_atoms2aos(self.cell, reorder, phases)
                    assert np.allclose(phases, 1)
                    dmsym = True
                    for dm1 in dm1s:
                        dm1t = dm1[reorder][:,reorder]
                        if not np.allclose(dm1, dm1t, rtol=0, atol=dm1_tol):
                            dmsym = False
                            break
                    if not dmsym:
                        continue

                # 6) Add as new dvec
                dvecs[dim].append(dvec)

        # Check that not more than a single vector was found for each dimension
        assert np.all([(len(dvecs[d]) <= 1) for d in range(3)])
        nsubcells = [(1/(dvecs[d][0][d]) if dvecs[d] else 1) for d in range(3)]
        assert np.allclose(np.rint(nsubcells), nsubcells)

        nsubcells = np.rint(nsubcells).astype(int)
        return nsubcells


    #def find_primitive_cells_old(self, tol=1e-5):
    #    if self.pbcndims == 0:
    #        raise ValueError()

    #    # Construct all atom1 < atom2 vectors
    #    #for atom1, coords1 in enumerate(self.atom_coords()):
    #    #    for atom2, coords2 in enumerate(self.atom_coords
    #    coords = self.atom_coords()
    #    aavecs = (coords[:,None,:] - coords[None,:,:])

    #    #print(coords[3] - coords[1])
    #    #print(aavecs[3,1])


    #    tvecs = aavecs.reshape(self.natom*self.natom, 3)
    #    #tvecs = np.unique(tvecs, axis=0)

    #    # Filter zero T vectors
    #    tvecs = np.asarray([t for t in tvecs if (np.linalg.norm(t) > 1e-6)])

    #    # Filter unique T vectors
    #    tvecs_unique = []
    #    for tvec in tvecs[1:]:
    #        if np.sum(tvec) < 0:
    #            tvec *= -1
    #        if len(tvecs_unique) == 0:
    #            tvecs_unique.append(tvec)
    #            continue
    #        unique = True
    #        for utvec in tvecs_unique:
    #            if min(np.linalg.norm(tvec - utvec), np.linalg.norm(tvec + utvec)) < tol:
    #                unique = False
    #                break
    #        if unique:
    #            tvecs_unique.append(tvec)
    #    tvecs = np.asarray(tvecs_unique)

    #    # Filter and sort according to parallelity to lattice vectors
    #    latvec = self.lattice_vectors()
    #    tvecs_dim = [[], [], []]
    #    for tvec in tvecs:
    #        for d in range(3):
    #            angle = min(angle_between(tvec, latvec[d]), angle_between(-tvec, latvec[d]))
    #            if angle < 1e-5:
    #                tvecs_dim[d].append(tvec)
    #                break
    #    #tvecs = np.asarray(tvecs_para)
    #    print(tvecs_dim[0])
    #    print(tvecs_dim[1])
    #    print(tvecs_dim[2])

    #    # Find most 
    #    for d in range(3):
    #        pass


if __name__ == '__main__':
    from timeit import default_timer as timer

    import pyscf
    import pyscf.pbc
    import pyscf.pbc.gto
    import pyscf.pbc.scf
    import pyscf.pbc.tools

    cell = pyscf.pbc.gto.Cell()
    cell.a = 3*np.eye(3)
    #cell.a[0,2] += 2.0
    #cell.atom = ['He 0 0 0', 'He 0 0 1', 'He 0 0 2.0000000001', 'He 0 1 2', 'He 2 2 1', 'He 5 5 5']
    #cell.atom = ['He %f %f %f' % tuple(xyz) for xyz in np.random.rand(4,3)]
    cell.atom = 'He 0 0 0'
    cell.unit = 'Bohr'
    cell.basis = 'def2-svp'
    cell.build()
    #cell.dimension = 2


    sc = [1, 2, 3]
    cell = pyscf.pbc.tools.super_cell(cell, sc)
    #cell.atom += ['Ne 1 2 3']
    #cell.build()

    mf = pyscf.pbc.scf.RHF(cell)
    mf = mf.density_fit(auxbasis='def2-svp-ri')
    mf.kernel()
    dm1 = mf.make_rdm1()
    dm1 += 1e-7*np.random.rand(*dm1.shape)

    sym = Symmetry(cell)
    t0 = timer()
    ncells = sym.find_primitive_ncells(respect_dm1=dm1)

    print(timer()-t0)
    print(ncells)
