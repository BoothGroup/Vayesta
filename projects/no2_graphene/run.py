import argparse

import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.dft
import pyscf.lo

import vayesta
import vayesta.ewf

from structures import NO2_Graphene

parser = argparse.ArgumentParser()
parser.add_argument('--structure', type=int, default=-1)
parser.add_argument('--kmesh', type=int, nargs=2, default=False)
parser.add_argument('--basis', default='cc-pVDZ')
parser.add_argument('--auxbasis', default='cc-pVDZ-ri')
parser.add_argument('--gate-range', type=float, nargs=3, default=[-0.01, 0.01, 0.002])
parser.add_argument('--gates', type=float, nargs='*', default=[0.0])
#parser.add_argument('--gate-range', type=float, nargs=3, default=[-0.004, 0.004, 0.002])
parser.add_argument('--invert-scan', action='store_true')
parser.add_argument('--trace', action='store_true')
parser.add_argument('--gate-index', type=int)
parser.add_argument('--supercell', type=int, nargs=2, default=[5, 5])
#parser.add_argument('--supercell', type=int, nargs=2, default=[6, 6])
parser.add_argument('--vacuum-size', type=float, default=30.0)
parser.add_argument('--z-graphene', type=float)
parser.add_argument('--verbose', type=int, default=10)
parser.add_argument('--dimension', type=int, default=2)
parser.add_argument('--spin', type=int, default=1)
parser.add_argument('--lindep-threshold', type=float)
# --- MF
parser.add_argument('--xc', default=None)
parser.add_argument('--mf-only', action='store_true')
parser.add_argument('--scf-conv-tol', type=float, default=1e-9)
# --- Embedding
parser.add_argument('--fragment-type', default='iao')
parser.add_argument('--nc', type=int, default=0)
parser.add_argument('--eta', type=float)
parser.add_argument('--augmented-basis', type=int, default=1)
parser.add_argument('--dmet-threshold', type=float, default=1e-4)
args =parser.parse_args()

if args.z_graphene is None:
    args.z_graphene = args.vacuum_size / 2

# Cell

def get_gate_potential(mf, gate):
    """In AO basis"""
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
    layer = [l[1].startswith('C') for l in mf.mol.ao_labels(None)]
    sc = np.dot(ovlp, c_lo[:,layer])
    v_gate = gate*np.dot(sc, sc.T)
    return v_gate

if args.gates is not None:
    gates = args.gates
else:
    gates = np.arange(args.gate_range[0], args.gate_range[1]+1e-12, args.gate_range[2])
if args.invert_scan:
    gates = gates[::-1]

dm1_mf = None
for idx, gate in enumerate(gates):

    if args.gate_index is not None and idx != args.gate_index:
        continue

    print("Now calculating gate %.3f" % gate)

    cell = pyscf.pbc.gto.Cell()
    no2_graphene = NO2_Graphene(args.supercell, structure=args.structure, vacuum_size=args.vacuum_size, z_graphene=args.z_graphene)
    cell.a, cell.atom = no2_graphene.get_amat_atom()
    cell.dimension = args.dimension
    cell.spin = args.spin
    cell.verbose = args.verbose
    if args.augmented_basis:
        cell.basis = {'N' : 'aug-%s' % args.basis,
                      'O' : 'aug-%s' % args.basis,
                      'C' : args.basis}
    else:
        cell.basis = args.basis
    cell.lindep_threshold = args.lindep_threshold
    cell.build()

    # MF
    if args.kmesh:
        kpts = cell.get_kpts(args.kmesh)
        mf = pyscf.pbc.scf.KUHF(cell, kpts)
    else:
        kpts = None
        if args.xc is None:
            mf = pyscf.pbc.scf.UHF(cell)
        else:
            mf = pyscf.pbc.dft.UKS(cell)
            mf.xc = args.xc
    mf.conv_tol = args.scf_conv_tol
    mf.max_cycle = 100

    mf = mf.density_fit(auxbasis=args.auxbasis)

    # Gate potential
    if gate:
        v_gate = get_gate_potential(mf, gate)
        hcore_orig = mf.get_hcore
        hcore_gate = lambda *args : (hcore_orig(*args) + v_gate)
        mf.get_hcore = hcore_gate
    else:
        v_gate = None
        hcore_orig = hcore_gate = mf.get_hcore

    # Run MF
    if args.trace:
        print("Running HF with initial guess")
        mf.kernel(dm0=dm1_mf)
    else:
        print("Running HF without initial guess")
        mf.kernel()
    dm1_mf = mf.make_rdm1()

    if not mf.converged:
        print("MF not converged at gate= %.3f !" % gate)

    # MF energy without gate
    e_mf_gate = mf.e_tot
    mf.e_tot = mf.energy_tot(h1e=hcore_orig())
    print("E(MF)= % 16.8f Ha  E(MF+gate)= % 16.8f Ha" % (mf.e_tot, e_mf_gate))

    # Embedding
    opts = {'make_rdm1' : True, 'dmet_threshold' : args.dmet_threshold}
    if args.eta is None:
        ecc = vayesta.ewf.EWF(mf, bath_type=None, **opts)
    else:
        ecc = vayesta.ewf.EWF(mf, bno_threshold=args.eta, **opts)

    ecc.pop_analysis(dm1=dm1_mf, local_orbitals='mulliken', filename='mulliken-mf-%.3f.pop' % gate)
    ecc.pop_analysis(dm1=dm1_mf, local_orbitals='lowdin',   filename='lowdin-mf-%.3f.pop' % gate)
    ecc.pop_analysis(dm1=dm1_mf, local_orbitals='iao+pao',  filename='iao+pao-mf-%.3f.pop' % gate)

    if not args.mf_only:

        if args.fragment_type == 'iao':
            ecc.iao_fragmentation()
        elif args.fragment_type == 'iaopao':
            ecc.iaopao_fragmentation()
        elif args.fragment_type == 'sao':
            ecc.sao_fragmentation()

        if v_gate is not None:
            ecc.get_fock_for_energy = lambda *args : (ecc.get_fock(*args) + v_gate)

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
        ecc.kernel()

        frag.pop_analysis(local_orbitals='mulliken', filename='mulliken-cc-%.3f.pop' % gate)
        frag.pop_analysis(local_orbitals='lowdin',   filename='lowdin-cc-%.3f.pop' % gate)
        frag.pop_analysis(local_orbitals='iao+pao',  filename='iao+pao-cc-%.3f.pop' % gate)

        e_cc = ecc.e_tot
    else:
        e_cc = np.nan

    with open('energies.txt', 'a') as f:
        f.write('%.4f  % 16.8f  % 16.8f\n' % (gate, mf.e_tot, e_cc))
