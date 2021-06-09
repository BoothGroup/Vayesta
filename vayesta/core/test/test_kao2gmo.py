from timeit import default_timer as timer

import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.gto
import pyscf.pbc.scf
import pyscf.pbc.tools

import vayesta
import vayesta.core
from vayesta.core import kao2gmo


def make_2d_hexagonal_cell(atoms, a, c, supercell=False, cell_params=None):
    amat = np.asarray([
            [a, 0, 0],
            [a/2, a*np.sqrt(3.0)/2, 0],
            [0, 0, c]])
    coords_internal = np.asarray([
        [2.0, 2.0, 3.0],
        [4.0, 4.0, 3.0]])/6
    coords = np.dot(coords_internal, amat)
    atom = []
    for i in range(len(atoms)):
        if atoms[i] is not None:
            atom.append((atoms[i], coords[i]))
    cell = pyscf.pbc.gto.Cell()
    cell.a = amat
    cell.atom = atom

    if cell_params is None:
        cell_params = {}

    cell.basis = cell_params.get('basis', 'def2-svp')
    if cell_params.get('pseudo', False):
        cell.pseudo = cell_params['pseudo']
    cell.verbose = cell_params.get('verbose', 10)
    cell.dimension = 2
    cell.output = cell_params.get('output', 'pyscf.txt')
    cell.build()
    if supercell:
        cell = pyscf.pbc.tools.super_cell(cell, supercell)
    return cell


def test_graphene():

    cell = make_2d_hexagonal_cell(('C', 'C'), a=2.46, c=20.0)
    #kpts = cell.make_kpts([3,3,3])
    kpts = cell.make_kpts([2,2,2])

    kmf = pyscf.pbc.scf.KRHF(cell, kpts)
    auxbasis = 'def2-svp-ri'
    kmf = kmf.density_fit(auxbasis=auxbasis)
    kmf.kernel()

    mf = pyscf.pbc.tools.k2gamma.k2gamma(kmf)

    # Select only some MOs, to speed the test up
    nocc = 10
    nvir = 10
    mo_occ = mf.mo_coeff[:,:nocc]
    mo_vir = mf.mo_coeff[:,-nvir:]
    mo_coeff = np.hstack((mo_occ, mo_vir))

    t0 = timer()
    eris = kao2gmo.gdf_to_eris(kmf.with_df, mo_coeff, nocc=nocc, only_ovov=True)
    print("T(gdf_to_eris)= %.3f s" % (timer()-t0))

    # Check correctness:
    mf = mf.density_fit(auxbasis=auxbasis)
    eris2 = mf.with_df.ao2mo((mo_occ, mo_vir, mo_occ, mo_vir), compact=False).reshape((nocc,nvir,nocc,nvir))
    print("Error norm: L2= %.2e  Linf= %.2e" % (np.linalg.norm(eris['ovov'] - eris2), abs(eris['ovov']-eris2).max()))


if __name__ == '__main__':
    test_graphene()
