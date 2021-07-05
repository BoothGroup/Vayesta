 import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.gto
import pyscf.pbc.scf

import vayesta
import vayesta.ewf

dists = np.arange(2.35, 2.6501, 0.05)
basis = 'def2-svp'
kmesh = [2,2,1]

bno_threshold = [1e-4, 1e-5, 1e-6, 1e-7]

sc_mode = 1
#sc_mode = 2

def make_graphene(a, c, atoms=['C', 'C']):
    cell = pyscf.pbc.gto.Cell()
    cell.a = np.asarray([
            [a, 0, 0],
            [a/2, a*np.sqrt(3.0)/2, 0],
            [0, 0, c]])
    coords_internal = np.asarray([
        [2.0, 2.0, 3.0],
        [4.0, 4.0, 3.0]])/6
    coords = np.dot(coords_internal, cell.a)
    atom = []
    for i in range(len(atoms)):
        if atoms[i] is not None:
            atom.append([atoms[i], coords[i]])
    cell.atom = atom
    cell.verbose = 10
    cell.output = 'pyscf_out.txt'
    cell.basis = basis
    cell.dimension = 2
    cell.build()

    return cell

for d in dists:

    cell = make_graphene(d, c=20.0)
    kpts = cell.make_kpts(kmesh)

    # Hartree-Fock
    kmf = pyscf.pbc.scf.KRHF(cell, kpts)
    kmf = kmf.density_fit(auxbasis='def2-svp-ri')
    kmf.kernel()

    ecc = vayesta.ewf.EWF(kmf, bno_threshold=bno_threshold)
    ecc.make_atom_fragment(0, sym_factor=2)
    ecc.kernel()

    with open('ewf-ccsd.txt', 'a') as f:
        energies = ecc.get_energies()
        fmt = ("%6.3f" + len(energies)*"  %+16.8f" + "\n")
        f.write(fmt % (d, *energies))

    scecc = vayesta.ewf.EWF(kmf, bno_threshold=bno_threshold, sc_mode=sc_mode)
    scecc.make_all_atom_fragments(sym_factor=0.25)
    scecc.tailor_all_fragments()
    scecc.kernel()

    with open('scewf-ccsd.txt', 'a') as f:
        energies = scecc.get_energies()
        fmt = ("%6.3f" + len(energies)*"  %+16.8f" + "\n")
        f.write(fmt % (d, *energies))
