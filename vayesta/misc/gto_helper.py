import numpy as np

import itertools


def loop_neighbor_cells(lattice_vectors=None, dimension=3):
    if (dimension == 0):
        yield np.zeros(3)
        return
    dxs = (-1, 0, 1)
    dys = dzs = (0,)
    if dimension > 1:
        dys = (-1, 0, 1)
    if dimension > 2:
        dzs = (-1, 0, 1)
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

def make_counterpoise_fragments(mol, fragments, full_basis=True, add_rest_fragment=True, dump_input=True):
    '''Make mol objects for counterpoise calculations.

    Parameters
    ----------
    fragments : iterable
    full_basis : bool, optional
    add_rest_fragment : bool, optional

    Returns
    -------
    fmols : list
    '''
    GHOST_PREFIX = "GHOST-"
    atom = mol.format_atom(mol.atom, unit=1.0)
    atom_symbols = [mol.atom_symbol(atm_id) for atm_id in range(mol.natm)]

    def make_frag_mol(frag):
        f_mask = numpy.isin(atom_symbols, frag)
        if sum(f_mask) == 0:
            raise ValueError("No atoms found for fragment: %r", frag)
        fmol = mol.copy()
        fatom = []
        for atm_id, atm in enumerate(atom):
            sym = atm[0]
            # Atom is in fragment
            if f_mask[atm_id]:
                fatom.append(atm)
            # Atom is NOT in fragment [only append if full basis == True]
            elif full_basis:
                sym_new = GHOST_PREFIX + sym
                fatom.append([sym_new, atm[1]])
                # Change basis dictionary
                if isinstance(fmol.basis, dict) and (sym in fmol.basis.keys()):
                    fmol.basis[sym_new] = fmol.basis[sym]
                    del fmol.basis[sym]
            # Remove from basis [not necessary, since atom is not present anymore, but cleaner]:
            elif isinstance(fmol.basis, dict) and (sym in fmol.basis.keys()):
                del fmol.basis[sym]

        # Rebuild fragment mol object
        fmol.atom = fatom
        fmol._built = False
        fmol.build(dump_input, False)
        return fmol

    fmols = []
    for frag in fragments:
        fmol = make_frag_mol(frag)
        fmols.append(fmol)

    # Add fragment containing all atoms not part of any specified fragments
    if add_rest_fragment:
        rest_mask = numpy.full((mol.natm,), True)
        # Set all atoms to False that are part of a fragment
        for frag in fragments:
            rest_mask = numpy.logical_and(numpy.isin(atom_symbols, frag, invert=True), rest_mask)
        if numpy.any(rest_mask):
            rest_frag = numpy.asarray(atom_symbols)[rest_mask]
            fmol = make_frag_mol(rest_frag)
            fmols.append(fmol)

    # TODO: Check that no atom is part of more than one fragments

    return fmols

if __name__ == '__main__':

    import pyscf
    import pyscf.pbc
    import pyscf.pbc.gto
    import pyscf.pbc.tools
    from vayesta.misc import solids

    cell = pyscf.pbc.gto.Cell()
    cell.a, cell.atom = solids.graphene()
    cell.dimension = 2
    cell.build()
    cell = pyscf.pbc.tools.super_cell(cell, (2, 2, 1))

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
