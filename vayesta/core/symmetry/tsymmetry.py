"""Translational symmetry module."""
import logging
import itertools

import numpy as np
from pyscf.lib.parameters import BOHR

log = logging.getLogger(__name__)


def to_bohr(a, unit):
    if unit[0].upper() == "A":
        return a / BOHR
    if unit[0].upper() == "B":
        return a
    raise ValueError("Unknown unit: %s" % unit)


def get_mesh_tvecs(cell, tvecs, unit="Ang"):
    rvecs = cell.lattice_vectors()
    if np.ndim(tvecs) == 1 and len(tvecs) == 3:
        mesh = tvecs
        tvecs = [rvecs[d] / mesh[d] for d in range(3)]
    elif np.ndim(tvecs) == 2 and tvecs.shape == (3, 3):
        for d in range(3):
            if np.all(tvecs[d] == 0):
                tvecs[d] = rvecs[d]
        tvecs = to_bohr(tvecs, unit)
        mesh = []
        for d in range(3):
            nd = np.round(np.linalg.norm(rvecs[d]) / np.round(np.linalg.norm(tvecs[d])))
            if abs(nd - int(nd)) > 1e-14:
                raise ValueError("Translationally vectors not consumerate with lattice vectors. Correct units?")
            mesh.append(int(nd))
    else:
        raise ValueError("Unknown set of T-vectors: %r" % tvecs)
    return mesh, tvecs


def loop_tvecs(cell, tvecs, unit="Ang", include_origin=False):
    mesh, tvecs = get_mesh_tvecs(cell, tvecs, unit)
    log.debugv("nx= %d ny= %d nz= %d", *mesh)
    log.debugv("tvecs=\n%r", tvecs)
    for dz, dy, dx in itertools.product(range(mesh[2]), range(mesh[1]), range(mesh[0])):
        if not include_origin and (abs(dx) + abs(dy) + abs(dz) == 0):
            continue
        t = dx * tvecs[0] + dy * tvecs[1] + dz * tvecs[2]
        yield (dx, dy, dz), t


def tsymmetric_atoms(cell, rvecs, xtol=1e-8, unit="Ang", check_element=True, check_basis=True):
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
            dist = pos2 - pos1
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


def compare_atoms(cell, atm1, atm2, check_basis=True):
    type1 = cell.atom_pure_symbol(atm1)
    type2 = cell.atom_pure_symbol(atm2)
    if type1 != type2:
        return False
    if not check_basis:
        return True
    bas1 = cell._basis[cell.atom_symbol(atm1)]
    bas2 = cell._basis[cell.atom_symbol(atm2)]
    if bas1 != bas2:
        return False
    return True


def reorder_atoms(cell, tvec, boundary=None, unit="Ang", check_basis=True):
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
    if boundary is None:
        if hasattr(cell, "boundary"):
            boundary = cell.boundary
        else:
            boundary = "PBC"
    if np.ndim(boundary) == 0:
        boundary = 3 * [boundary]
    elif np.ndim(boundary) == 1 and len(boundary) == 2:
        boundary = [boundary[0], boundary[1], "PBC"]

    tvec = to_bohr(tvec, unit)

    rvecs = cell.lattice_vectors()
    bvecs = np.linalg.inv(rvecs.T)
    log.debugv("lattice vectors=\n%r", rvecs)
    log.debugv("inverse lattice vectors=\n%r", bvecs)
    atom_coords = np.dot(cell.atom_coords(), bvecs.T)
    log.debugv("Atom coordinates:")
    for atm, coords in enumerate(atom_coords):
        log.debugv("%3d %f %f %f", atm, *coords)
    log.debugv("boundary= %r", boundary)
    for d in range(3):
        if boundary[d].upper() == "PBC":
            boundary[d] = 1
        elif boundary[d].upper() == "APBC":
            boundary[d] = -1
        else:
            raise ValueError("Boundary= %s" % boundary)
    boundary = np.asarray(boundary)
    log.debugv("boundary= %r", boundary)

    def get_atom_at(pos, xtol=1e-8):
        for dx, dy, dz in itertools.product([0, -1, 1], repeat=3):
            if cell.dimension in (1, 2) and (dz != 0):
                continue
            if cell.dimension == 1 and (dy != 0):
                continue
            dr = np.asarray([dx, dy, dz])
            phase = np.product(boundary[dr != 0])
            # log.debugv("dx= %d dy= %d dz= %d phase= %d", dx, dy, dz, phase)
            # print(atom_coords.shape, dr.shape, pos.shape)
            dists = np.linalg.norm(atom_coords + dr - pos, axis=1)
            idx = np.argmin(dists)
            if dists[idx] < xtol:
                return idx, phase
            # log.debugv("atom %d not close with distance %f", idx, dists[idx])
        return None, None

    natm = cell.natm
    reorder = np.full((natm,), -1)
    inverse = np.full((natm,), -1)
    phases = np.full((natm,), 0)
    tvec_internal = np.dot(tvec, bvecs.T)
    for atm0 in range(cell.natm):
        atm1, phase = get_atom_at(atom_coords[atm0] + tvec_internal)
        if atm1 is None or not compare_atoms(cell, atm0, atm1, check_basis=check_basis):
            return None, None, None
        log.debugv("atom %d T-symmetric to atom %d for translation %s", atm1, atm0, tvec)
        reorder[atm1] = atm0
        inverse[atm0] = atm1
        phases[atm0] = phase
    assert not np.any(reorder == -1)
    assert not np.any(inverse == -1)
    assert not np.any(phases == 0)

    assert np.all(np.arange(cell.natm)[reorder][inverse] == np.arange(cell.natm))

    return reorder, inverse, phases


def reorder_atoms2aos(cell, atom_reorder, atom_phases):
    if atom_reorder is None:
        return None, None, None
    aoslice = cell.aoslice_by_atom()[:, 2:]
    nao = cell.nao_nr()
    reorder = np.full((nao,), -1)
    inverse = np.full((nao,), -1)
    phases = np.full((nao,), 0)
    for atm0 in range(cell.natm):
        atm1 = atom_reorder[atm0]
        aos0 = list(range(aoslice[atm0, 0], aoslice[atm0, 1]))
        aos1 = list(range(aoslice[atm1, 0], aoslice[atm1, 1]))
        reorder[aos0[0] : aos0[-1] + 1] = aos1
        inverse[aos1[0] : aos1[-1] + 1] = aos0
        phases[aos0[0] : aos0[-1] + 1] = atom_phases[atm1]
    assert not np.any(reorder == -1)
    assert not np.any(inverse == -1)
    assert not np.any(phases == 0)

    assert np.all(np.arange(nao)[reorder][inverse] == np.arange(nao))

    return reorder, inverse, phases


def reorder_aos(cell, tvec, unit="Ang"):
    atom_reorder, atom_inverse, atom_phases = reorder_atoms(cell, tvec, unit=unit)
    return reorder_atoms2aos(cell, atom_reorder, atom_phases)
    # if atom_reorder is None:
    #    return None, None, None
    # aoslice = cell.aoslice_by_atom()[:,2:]
    # nao = cell.nao_nr()
    # reorder = np.full((nao,), -1)
    # inverse = np.full((nao,), -1)
    # phases = np.full((nao,), 0)
    # for atm0 in range(cell.natm):
    #    atm1 = atom_reorder[atm0]
    #    aos0 = list(range(aoslice[atm0,0], aoslice[atm0,1]))
    #    aos1 = list(range(aoslice[atm1,0], aoslice[atm1,1]))
    #    reorder[aos0[0]:aos0[-1]+1] = aos1
    #    inverse[aos1[0]:aos1[-1]+1] = aos0
    #    phases[aos0[0]:aos0[-1]+1] = atom_phases[atm1]
    # assert not np.any(reorder == -1)
    # assert not np.any(inverse == -1)
    # assert not np.any(phases == 0)

    # assert np.all(np.arange(nao)[reorder][inverse] == np.arange(nao))

    # return reorder, inverse, phases


def _make_reorder_op(reorder, phases):
    def reorder_op(a, axis=0):
        if isinstance(a, (tuple, list)):
            return tuple([reorder_op(x, axis=axis) for x in a])
        bc = tuple(axis * [None] + [slice(None, None, None)] + (a.ndim - axis - 1) * [None])
        return np.take(a, reorder, axis=axis) * phases[bc]

    return reorder_op


def get_tsymmetry_op(cell, tvec, unit="Ang"):
    reorder, inverse, phases = reorder_aos(cell, tvec, unit=unit)
    if reorder is None:
        # Not a valid symmetry
        return None
    tsym_op = _make_reorder_op(reorder, phases)
    return tsym_op


if __name__ == "__main__":
    import vayesta

    log = vayesta.log

    import pyscf
    import pyscf.pbc.gto
    import pyscf.pbc.tools

    cell = pyscf.pbc.gto.Cell()
    cell.atom = "H 0 0 0, H 0 1 2"
    cell.basis = "sto-3g"
    cell.a = np.eye(3)
    cell.dimension = 1
    cell.build()

    ncopy = 4
    cell = pyscf.pbc.tools.super_cell(cell, [ncopy, 1, 1])

    reorder, inverse, phases = reorder_atoms(cell, cell.a[0] / ncopy, unit="B")
    print(reorder)
    print(inverse)

    reorder, inverse, phases = reorder_aos(cell, cell.a[0] / ncopy, unit="B")
    print(reorder)
    print(inverse)
