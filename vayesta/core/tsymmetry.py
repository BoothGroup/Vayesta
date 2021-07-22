"""Translational symmetry module."""
import logging
import itertools

import numpy as np

from .helper import to_bohr

log = logging.getLogger(__name__)


def get_mesh_tvecs(cell, tvecs, unit='Ang'):
    rvecs = cell.lattice_vectors()
    if (np.ndim(tvecs) == 1 and len(tvecs) == 3):
        mesh = tvecs
        tvecs = [rvecs[d]/mesh[d] for d in range(3)]
    else:
        tvecs = to_bohr(tvecs, unit)
        mesh = []
        for d in range(3):
            nd = np.round(np.linalg.norm(rvecs[d]) / np.round(np.linalg.norm(tvecs[d])))
            if abs(nd - int(nd)) > 1e-14:
                raise ValueError("Translationally vectors not consumerate with lattice vectors. Correct units?")
            mesh.append(int(nd))
    #if cell.dimension == 2 and (mesh[1] != 1):
    return mesh, tvecs


def tsymmetric_atoms(cell, rvecs, xtol=1e-8, unit='Ang', check_element=True, check_basis=True):
    """Get indices of translationally symmetric atoms.

    Parameters
    ----------
    cell: pyscf.pbc.gto.Cell
        Unit cell.
    rvecs: (3, 3) array
        The rows contain the real space translation vectors.
    xtol: float, optional
        Tolerance to identify equivalent atom positions. Default: 1e-8
    unit: ['Ang', 'Bohr']
        Unit of `rvecs` and `xtol`. Default: 'Ang'.

    Returns
    -------
    indices: list
        List with length `cell.natm`. Each element represents the lowest atom index of a
        translationally symmetry equivalent atom.
    """
    rvecs = to_bohr(rvecs, unit)

    # Reciprocal lattice vectors
    bvecs = np.linalg.inv(rvecs.T)
    atom_coords = cell.atom_coords()
    indices = [0]
    for atm1 in range(1, cell.natm):
        type1 = cell.atom_pure_symbol(atm1)
        bas1 = cell._basis[cell.atom_symbol(atm1)]
        # Position in internal coordinates:
        pos1 = np.dot(atom_coords[atm1], bvecs.T)
        for atm2 in set(indices):
            # 1) Compare element symbol
            if check_element:
                type2 = cell.atom_pure_symbol(atm2)
                if type1 != type2:
                    continue
            # 2) Compare basis set
            if check_basis:
                bas2 = cell._basis[cell.atom_symbol(atm2)]
                if bas1 != bas2:
                    continue
            # 3) Check distance modulo lattice vectors
            pos2 = np.dot(atom_coords[atm2], bvecs.T)
            dist = (pos2 - pos1)
            dist -= np.rint(dist)
            if np.linalg.norm(dist) < xtol:
                # atm1 and atm2 are symmetry equivalent
                log.debug("Atom %d is translationally symmetric to atom %d", atm1, atm2)
                indices.append(atm2)
                break
        else:
            # No symmetry related atom could be found; append own index
            indices.append(atm1)

    return indices


def reorder_atoms(cell, tvec, unit='Ang'):
    """Reordering of atoms for a given translation.

    Parameters
    ----------
    tvec: (3) array
        Translation vector.

    Returns
    -------
    reorder: list
    inverse: list
    """
    tvec = to_bohr(tvec, unit)

    rvecs = cell.lattice_vectors()
    bvecs = np.linalg.inv(rvecs.T)
    atom_coords = np.dot(cell.atom_coords(), bvecs.T)

    def get_atom_at(pos, xtol=1e-8):
        for dx, dy, dz in itertools.product([0,-1,1], repeat=3):
            if cell.dimension in (1, 2) and (dz != 0): continue
            if cell.dimension == 1 and (dy != 0): continue
            dr = np.asarray([dx, dy, dz])
            dists = np.linalg.norm(atom_coords + dr - pos, axis=1)
            idx = np.argmin(dists)
            if dists[idx] < xtol:
                return idx
        return None

    reorder = cell.natm*[None]
    inverse = cell.natm*[None]
    tvec = np.dot(tvec, bvecs.T)
    for atm0 in range(cell.natm):
        atm1 = get_atom_at(atom_coords[atm0] + tvec)
        if atm1 is None:
            return None, None
        reorder[atm1] = atm0
        inverse[atm0] = atm1

    assert np.all(np.arange(cell.natm)[reorder][inverse] == np.arange(cell.natm))

    return reorder, inverse

def reorder_aos(cell, tvec, unit='Ang'):
    atom_reorder, atom_inverse = reorder_atoms(cell, tvec, unit=unit)
    if atom_reorder is None:
        return None, None
    aoslice = cell.aoslice_by_atom()[:,2:]
    nao = cell.nao_nr()
    reorder = nao*[None]
    inverse = nao*[None]
    for atm0 in range(cell.natm):
        atm1 = atom_reorder[atm0]
        aos0 = list(range(aoslice[atm0,0], aoslice[atm0,1]))
        aos1 = list(range(aoslice[atm1,0], aoslice[atm1,1]))
        reorder[aos0[0]:aos0[-1]+1] = aos1
        inverse[aos1[0]:aos1[-1]+1] = aos0

    assert np.all(np.arange(nao)[reorder][inverse] == np.arange(nao))

    return reorder, inverse


if __name__ == '__main__':
    import pyscf
    import pyscf.pbc.gto
    import pyscf.pbc.tools

    cell = pyscf.pbc.gto.Cell()
    cell.atom='H 0 0 0'
    cell.basis='sto-3g'
    cell.a = np.eye(3)
    cell.dimension = 1
    cell.build()

    ncopy = 4
    cell = pyscf.pbc.tools.super_cell(cell, [ncopy,1,1])

    reorder, inverse = reorder_atoms(cell, cell.a[0]/ncopy, unit='B')
    print(reorder)
    print(inverse)

    reorder, inverse = reorder_aos(cell, cell.a[0]/ncopy, unit='B')
    print(reorder)
    print(inverse)
