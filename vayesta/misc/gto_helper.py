import numpy as np

import itertools


def loop_neighbor_cells(lattice_vectors=None, dimension=3):
    if (dimension == 0):
        yield np.zeros(3)
        return
    dxs = dys = (0,)
    dzs = (-1, 0, 1)
    if dimension > 1:
        dys = (-1, 0, 1)
    if dimension > 2:
        dxs = (-1, 0, 1)
    for dr in itertools.product(dxs, dys, dzs):
        if lattice_vectors is None:
            yield np.asarray(dr)
        yield np.dot(dr, lattice_vectors)

def get_atom_distances(mol, point, dimension=None):
    """Get array containing the distances of all atoms to the specified point.

    Parameters
    ----------
    mol: PySCF Mole or Cell object
    point: Array(3)

    Returns
    -------
        Distances: Array(n(atom))
    """
    coords = mol.atom_coords()
    if hasattr(mol, 'lattice_vectors'):
        latvec = mol.lattice_vectors()
        dim = dimension if dimension is not None else mol.dimension
    else:
        latvec = None
        dim = 0

    distances = []
    for atm, r0 in enumerate(coords):
        dists = [np.linalg.norm(point - (r0+dr)) for dr in loop_neighbor_cells(latvec, dim)]
        distances.append(np.amin(dists))
    return np.asarray(distances)

def get_atom_shells(mol, point, dimension=None, decimals=5):

    distances = get_atom_distances(mol, point, dimension=dimension)
    drounded = distances.round(decimals)
    sort = np.argsort(distances, kind='stable')
    d_uniq, inv = np.unique(drounded[sort], return_inverse=True)
    shells = inv[np.argsort(sort)]

    return shells, distances


if __name__ == '__main__':

    import pyscf
    import pyscf.pbc
    import pyscf.pbc.gto
    import pyscf.pbc.tools

    import vayesta
    import vayesta.misc
    from vayesta.misc import solids

    cell = pyscf.pbc.gto.Cell()
    cell.a, cell.atom = solids.diamond()
    cell.build()
    cell = pyscf.pbc.tools.super_cell(cell, (2, 2, 2))

    point = cell.atom_coord(0)
    shells, dists = get_atom_shells(cell, point)
    print("Periodic boundary conditions:")
    for i in range(cell.natm):
        print('atom= %2d  distance= %12.8f  shell= %2d' % (i, dists[i], shells[i]))
    #uniq, idx, counts = np.unique(shells, return_index=True, return_counts=True)

    mol = cell.to_mol()
    shells, dists = get_atom_shells(mol, point)
    print("Open boundary conditions:")
    for i in range(mol.natm):
        print('atom= %2d  distance= %12.8f  shell= %2d' % (i, dists[i], shells[i]))
