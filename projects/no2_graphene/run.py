import argparse

import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.scf
import pyscf.pbc.dft
import pyscf.lo

import vayesta
import vayesta.ewf
from vayesta.misc import scf_with_mpi
import vayesta.core
from vayesta.core.mpi import mpi
from vayesta.core import foldscf

from structures import NO2_Graphene

parser = argparse.ArgumentParser()
parser.add_argument('--structure', type=int, default=-1)
parser.add_argument('--kmesh', type=int, nargs=3, default=False)
parser.add_argument('--basis', default='cc-pVDZ')
parser.add_argument('--basis-no2', default='aug-cc-pVTZ')
#parser.add_argument('--auxbasis', default='cc-pVDZ-ri')
parser.add_argument('--auxbasis', default=None)
#parser.add_argument('--gate-range', type=float, nargs=3, default=[-0.01, 0.01, 0.002])
#parser.add_argument('--gate-range', type=float, nargs=3, default=[-0.1, 0.1, 0.02])
parser.add_argument('--gate-range', type=float, nargs=3, default=[-0.2, 0.2, 0.05])

parser.add_argument('--gates', type=float, nargs='*', default=None)
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
parser.add_argument('--scf-max-cycle', type=int, default=100)
parser.add_argument('--scf-conv-tol', type=float, default=1e-9)
parser.add_argument('--scf-with-mpi', action='store_true')
# --- Embedding
parser.add_argument('--fragment-type', default='iao')
parser.add_argument('--nc', type=int, default=0)
parser.add_argument('--atomic-fragments', action='store_true')
parser.add_argument('--eta', type=float)
parser.add_argument('--dmet-threshold', type=float, default=1e-4)
args =parser.parse_args()

if args.z_graphene is None:
    args.z_graphene = args.vacuum_size / 2

# Cell
def get_gate_potential(mf, gate, atoms='C'):
    """In AO basis"""
    # Lowdin orbitals
    ovlp = mf.get_ovlp()
    e, v = np.linalg.eigh(ovlp)
    c_lo = np.dot(v*(e**-0.5), v.T.conj())
    # Graphene layer orbitals:
    layer = [l[1].startswith(atoms) for l in mf.mol.ao_labels(None)]
    sc = np.dot(ovlp, c_lo[:,layer])
    v_gate = gate*np.dot(sc, sc.T.conj())
    return v_gate

def get_gate_potential_kpts(mf, gate, atoms='C'):
    """In AO basis"""
    # Lowdin orbitals
    ovlp = mf.get_ovlp()
    nk = len(mf.kpts)
    v_gate = []
    for k in range(nk):
        ovlpk = ovlp[k]
        e, v = np.linalg.eigh(ovlpk)
        c_lo = np.dot(v*(e**-0.5), v.T.conj())
        # Graphene layer orbitals:
        layer = [l[1].startswith(atoms) for l in mf.mol.ao_labels(None)]
        sc = np.dot(ovlpk, c_lo[:,layer])
        v_gate.append(gate*np.dot(sc, sc.T.conj()))
    return np.asarray(v_gate)

if args.gates is not None:
    gates = args.gates
else:
    gates = np.arange(args.gate_range[0], args.gate_range[1]+1e-12, args.gate_range[2])
if args.invert_scan:
    gates = gates[::-1]

dm1_mf = None
for idx, gate in enumerate(gates):

    vayesta.new_log('gate-%.3f' % gate)

    if args.gate_index is not None and idx != args.gate_index:
        continue

    print("Now calculating gate %.3f" % gate)

    cell = pyscf.pbc.gto.Cell()
    no2_graphene = NO2_Graphene(args.supercell, structure=args.structure, vacuum_size=args.vacuum_size, z_graphene=args.z_graphene)
    cell.a, cell.atom = no2_graphene.get_amat_atom()
    cell.dimension = args.dimension
    nkpts = np.product(args.kmesh) if args.kmesh else 1
    cell.spin = (args.spin * nkpts)    # Spin is always defined for supercell, even for k-point samling!
    cell.verbose = args.verbose
    cell.basis = {'N' : args.basis_no2,
                  'O' : args.basis_no2,
                  'C' : args.basis}
    cell.lindep_threshold = args.lindep_threshold
    cell.output = ('pyscf.mpi%d.txt' % mpi.rank)
    cell.build()

    # MF
    if args.kmesh:
        kpts = cell.get_kpts(args.kmesh)
        if args.xc is None:
            mf = pyscf.pbc.scf.KUHF(cell, kpts)
        else:
            mf = pyscf.pbc.dft.KUKS(cell, kpts)
            mf.xc = args.xc
    else:
        kpts = None
        if args.xc is None:
            mf = pyscf.pbc.scf.UHF(cell)
        else:
            mf = pyscf.pbc.dft.UKS(cell)
            mf.xc = args.xc
    mf.conv_tol = args.scf_conv_tol
    mf.max_cycle = args.scf_max_cycle

    if args.auxbasis is not None:
        auxbasis = args.auxbasis
    else:
        if isinstance(args.basis, dict):
            auxbasis = {key : ('%s-ri' % args.basis[key]) for key in args.basis}
        else:
            auxbasis = '%s-ri' % args.basis
    print("RI basis: %s" % auxbasis)
    mf = mf.density_fit(auxbasis=auxbasis)
    mf.with_df._j_only = False

    # Gate potential
    if gate:
        hcore_orig = mf.get_hcore
        if hasattr(mf, 'kpts'):
            v_gate = get_gate_potential_kpts(mf, gate)
        else:
            v_gate = get_gate_potential(mf, gate)
        hcore_gate = lambda mf, *args : (hcore_orig(*args) + v_gate)
        mf.get_hcore = hcore_gate.__get__(mf)
    else:
        v_gate = None
        hcore_orig = hcore_gate = mf.get_hcore

    if args.scf_with_mpi:
        mf = scf_with_mpi(mf)

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

    # CCSD wants HF object
    if args.xc is not None:
        if kpts is None:
            mf_hf = pyscf.pbc.scf.UHF(cell)
        else:
            mf_hf = pyscf.pbc.scf.KUHF(cell)
        mf_hf.__dict__.update(mf.__dict__)
        mf = mf_hf

    opts = dict(make_rdm1=True, dmet_threshold=args.dmet_threshold)
    if v_gate is not None:
        # At this point we need to fold the hcore_orig to the supercell
        if kpts is not None:
            scell, phase = foldscf.get_phase(cell, kpts)
            #hcore_for_energy = lambda emb, *args : foldscf.k2bvk_2d(hcore_orig(*args), phase)
            def hcore_for_energy(emb, *args):
                return foldscf.k2bvk_2d(hcore_orig(*args), phase)
        else:
            hcore_for_energy = lambda emb, *args : hcore_orig(*args)
        opts['overwrite'] = dict(get_hcore_for_energy=hcore_for_energy)
    if args.eta is None:
        emb = vayesta.ewf.EWF(mf, bath_type=None, **opts)
    else:
        emb = vayesta.ewf.EWF(mf, bno_threshold=args.eta, **opts)

    emb.pop_analysis(dm1=emb.mf.make_rdm1(), local_orbitals='mulliken', filename='mulliken-mf-%.3f.pop' % gate)
    emb.pop_analysis(dm1=emb.mf.make_rdm1(), local_orbitals='lowdin',   filename='lowdin-mf-%.3f.pop' % gate)
    emb.pop_analysis(dm1=emb.mf.make_rdm1(), local_orbitals='iao+pao',  filename='iao+pao-mf-%.3f.pop' % gate)

    if not args.mf_only:

        if args.fragment_type == 'iao':
            emb.iao_fragmentation()
        elif args.fragment_type == 'iaopao':
            emb.iaopao_fragmentation()
        elif args.fragment_type == 'sao':
            emb.sao_fragmentation()

        #if v_gate is not None:
            #emb.get_fock_for_energy = lambda *args : (emb.get_fock(*args) + v_gate)
            #emb.get_hcore_for_energy = lambda : (emb.get_hcore() + v_gate)
        #    emb.get_hcore_for_energy = hcore_orig

        if args.atomic_fragments:
            emb.add_all_atomic_fragments()  # Primitive cell only!
            emb.kernel()

            # NEW CCSD DM:
            dm1 = emb.make_rdm1_ccsd()

            # Check DM
            if mpi.is_master:
                nelec = np.trace(dm1[0] + dm1[1])
                print("Nelec= %.8f tr(DM)= %.8f err= %.8f" % (cell.nelectron, nelec, nelec-cell.nelectron))
                spin = np.trace(dm1[0] - dm1[1])
                print("spin= %.8f tr(DM)= %.8f err= %.8f" % (cell.spin, spin, spin-cell.spin))
                e, v = np.linalg.eigh(dm1[0])
                print("alpha: min(n)= %.8f max(n)= %.8f" % (e[0], e[-1]))
                e, v = np.linalg.eigh(dm1[1])
                print("alpha: min(n)= %.8f max(n)= %.8f" % (e[0], e[-1]))

            emb.pop_analysis(dm1, mo_coeff=emb.mf.mo_coeff, local_orbitals='mulliken', filename='mulliken-cc-%.3f.pop' % gate)
            emb.pop_analysis(dm1, mo_coeff=emb.mf.mo_coeff, local_orbitals='lowdin', filename='lowdin-cc-%.3f.pop' % gate)
            emb.pop_analysis(dm1, mo_coeff=emb.mf.mo_coeff, local_orbitals='iao+pao', filename='iao+pao-cc-%.3f.pop' % gate)
        else:
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

            frag = emb.add_atomic_fragment(fragment_atoms)
            emb.kernel()

            frag.pop_analysis(local_orbitals='mulliken', filename='mulliken-cc-%.3f.pop' % gate)
            frag.pop_analysis(local_orbitals='lowdin',   filename='lowdin-cc-%.3f.pop' % gate)
            frag.pop_analysis(local_orbitals='iao+pao',  filename='iao+pao-cc-%.3f.pop' % gate)

        e_cc = emb.e_tot
    else:
        e_cc = np.nan

    if mpi.is_master:
        with open('energies.txt', 'a') as f:
            f.write('%.4f  % 16.8f  % 16.8f  % 16.8f\n' % (gate, mf.e_tot, emb.e_mf, e_cc))
