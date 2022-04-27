import argparse
import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.scf
import pyscf.pbc.tools
from pyscf.pbc.scf.addons import kconj_symmetry_

import vayesta
import vayesta.ewf
from vayesta.misc import solids
from vayesta.core.fragmentation import IAOPAO_Fragmentation


parser = argparse.ArgumentParser()
parser.add_argument('--tm', choices=['Ni', 'Fe', 'Co', 'Mn'], default='Ni')
parser.add_argument('--load-path')
parser.add_argument('--load-mf', action='store_true')
parser.add_argument('--load-cderi', action='store_true')
parser.add_argument('--solver', choices=['CCSD', 'FCI'], default='CCSD')
parser.add_argument('--eta', type=float, default=1e-4)
parser.add_argument('--bath', choices=['DMET', 'MP2-BNO'], default='MP2-BNO')
parser.add_argument('--vr', type=float, nargs='*', default=[1.0, 0.9, 0.8])
parser.add_argument('--kmesh', type=int, nargs=3, default=[2,2,2])
parser.add_argument('--supercell', type=int, nargs=3)
parser.add_argument('--fragmentation')
parser.add_argument('--fragment-atoms', type=int, nargs='*', default=[0])
parser.add_argument('--fragment-orbitals', nargs='*')
parser.add_argument('--rsdf', action='store_true')
parser.add_argument('--kconj', action='store_true')
parser.add_argument('--basis', default='def2-svp')
parser.add_argument('--auxbasis')
args = parser.parse_args()


# in Bohr^3
v0 = {
        'Mn' : 158.9,
        'Fe' : 144.1,
        'Co' : 137.0,
        'Ni' : 128.0,
    }[args.tm]

def get_a(v):
    """Volume to lattice constant"""
    return (4*v)**(1.0/3) * 0.529177


load_cderi = load_scf = None
if args.load_path:
    if args.load_cderi:
        load_cderi = '{}/cderi-%.2f.h5'.format(args.load_path)
    if args.load_mf:
        load_scf = '{}/mf-%.2f.chk'.format(args.load_path)

if args.auxbasis is None:
    args.auxbasis = '%s-ri' % args.basis

# Loop relative volumes
dm0 = None
for vr in args.vr:

    v = vr*v0
    a = get_a(v)
    print("Lattice const for volume %f: %f" % (v, a))

    cell = pyscf.pbc.gto.Cell()
    cell.a, cell.atom = solids.rocksalt(atoms=[args.tm, 'O'], a=a)
    cell.basis = args.basis
    cell.output = 'pyscf.out'
    cell.verbose = 10
    cell.build()

    if args.supercell is not None:
        cell = pyscf.pbc.tools.super_cell(cell, args.supercell)

    if np.product(args.kmesh) > 1:
        kpts = cell.make_kpts(args.kmesh)
    else:
        kpts = None

    # Hartree-Fock
    if kpts is None:
        mf = pyscf.pbc.scf.RHF(cell)
    else:
        mf = pyscf.pbc.scf.KRHF(cell, kpts)
    mf.chkfile = ('mf-%.2f.chk' % a)
    if args.rsdf:
        mf = mf.rs_density_fit(auxbasis=args.auxbasis)
    else:
        mf = mf.density_fit(auxbasis=args.auxbasis)
    df = mf.with_df
    if load_cderi:
        df._cderi = (load_cderi % a)
        print("Loading CDERI from '%s'" % df._cderi)
        df.build(with_j3c=False)
    else:
        mf.with_df._cderi_to_save = ('cderi-%.2f.h5' % a)
    if load_scf:
        print("Loading SCF from '%s'" % (load_scf % a))
        dm0 = mf.from_chk(load_scf % a)

    # k,-k symmetry:
    if args.kconj:
        mf = kconj_symmetry_(mf)
    print("Running HF...")
    mf.kernel(dm0=dm0)
    print("HF done.")
    dm0 = mf.make_rdm1()

    # Embedded calculation will automatically unfold the k-point sampled mean-field
    opts = dict(store_dm1=True, store_dm2=True)
    if args.bath == 'DMET':
        opts['bath_type'] = args.bath
    else:
        opts['bno_threshold'] = args.eta
    if args.solver == 'FCI':
        opts['solver'] = 'FCI'

    emb = vayesta.ewf.EWF(mf, **opts)
    if args.fragmentation == 'iao+pao':
        emb.iaopao_fragmentation()
    else:
        emb.iao_fragmentation()
    emb.add_atomic_fragment(args.fragment_atoms, orbital_filter=args.fragment_orbitals, add_symmetric=False)
    #if args.orbitals:
    #    emb.add_atomic_fragment(0, orbital_filter=args.orbitals)
    #else:
    #    emb.add_atomic_fragment(0)

    emb.kernel()
    emb.t1_diagnostic()
    frag_tm = emb.fragments[0]
    frag_tm.pop_analysis(full=True)

    proj = IAOPAO_Fragmentation(emb.mf, None)
    proj.kernel()

    s = emb.get_ovlp()

    def get_ssz(frag, atom1, atom2):
        name1, indices1 = proj.get_atomic_fragment_indices(atom1)
        name2, indices2 = proj.get_atomic_fragment_indices(atom2)
        f1 = proj.get_frag_coeff(indices1)
        f2 = proj.get_frag_coeff(indices2)

        # Projector...
        c = frag.cluster.c_active
        p1 = np.linalg.multi_dot((c.T, s, f1, f1.T, s, c))
        p2 = np.linalg.multi_dot((c.T, s, f2, f2.T, s, c))

        ssz = frag.get_cluster_ssz((p1, p2))
        return ssz

    ssz = get_ssz(frag_tm, 0, 0)

    with open('results.txt', 'a') as f:
        fmt = '%.4f  % .8f  % .8f  % .6f  % .6f\n'
        f.write(fmt % (a, emb.e_mf, emb.e_tot, ssz, 2*np.sqrt(ssz)))
