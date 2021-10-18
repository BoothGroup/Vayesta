import argparse

import numpy as np

import pyscf
import pyscf.pbc
import pyscf.lo

import vayesta
import vayesta.ewf

import structure

parser = argparse.ArgumentParser()
parser.add_argument('--kmesh', type=int, nargs=2, default=False)
parser.add_argument('--basis', default='cc-pVDZ')
parser.add_argument('--auxbasis', default='cc-pVDZ-ri')
parser.add_argument('--gate', type=float, default=0.0)
parser.add_argument('--gate-range', type=float, nargs=3, default=[-0.16, 0.04, 0.2])
parser.add_argument('--nc', type=int, default=0)
args = parser.parse_args()

# Cell

for gate in np.arange(*args.gate_range):
    print("Now calculating gate %.2f" % gate)

    cell = pyscf.pbc.gto.Cell()
    #cell.a, cell.atom = structure.get_3x3()
    cell.a, cell.atom = structure.get_5x5()
    cell.spin = 1
    cell.dimension = 2
    cell.verbose = 10
    cell.basis = args.basis
    cell.build()

    # MF
    if args.kmesh:
        kpts = cell.get_kpts(args.kmesh)
        mf = pyscf.pbc.scf.KUHF(cell, kpts)
    else:
        kpts = None
        mf = pyscf.pbc.scf.UHF(cell)

    mf = mf.density_fit(auxbasis=args.auxbasis)

    # potential
    if gate:
        # Lowdin orbitals
        ovlp = mf.get_ovlp()
        e, v = np.linalg.eigh(ovlp)
        c_lo = np.dot(v*(e**-0.5), v.T.conj())

        #c_lo2 = pyscf.lo.orth_ao(cell, 'lowdin')
        #c_lo2 = pyscf.lo.orth_ao(cell, 'lowdin', pre_orth_ao=None)
        #for i in range(c_lo.shape[-1]):
        #    assert (np.allclose(c_lo[:,i], c_lo2[:,i]) or np.allclose(c_lo[:,i], -c_lo2[:,i])), "%r %r" % (c_lo[:,i], c_lo2[:,i])
        #1/0

        # Graphene layer orbitals:
        layer = [l[1].startswith('C') for l in cell.ao_labels(None)]
        sc = np.dot(ovlp, c_lo[:,layer])
        hgate = mf.get_hcore() + gate * np.dot(sc, sc.T)

        mf.get_hcore = lambda *args : hgate

    # Run MF
    mf.kernel()

    # Embedding
    ecc = vayesta.ewf.EWF(mf, bath_type=None, make_rdm1=True)
    ecc.sao_fragmentation()

    # Define fragment
    def get_closest_atom(point, exclude):
        coords = cell.atom_coords().copy()
        distances = np.linalg.norm(coords - point, axis=1)
        distances[exclude] = np.inf
        return np.argmin(distances)
    fragment_atoms = [0, 1, 2]
    for i in range(args.nc):
        idx = get_closest_atom(cell.atom_coord(0), exclude=fragment_atoms)
        print("Adding atom %d at %r" % (idx, cell.atom_coord(idx)))
        fragment_atoms.append(idx)

    frag = ecc.add_atomic_fragment(fragment_atoms)
    dm1_mf = ecc.mf.make_rdm1()
    ecc.pop_analysis(dm1=dm1_mf, filename='pop-mf-%.2f.txt' % gate)

    ecc.kernel()
    frag.pop_analysis(filename='pop-cc-%.2f.txt' % gate)
