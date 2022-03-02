import argparse
import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.scf
import pyscf.pbc.dft
import pyscf.pbc.tools
from pyscf.pbc.scf.addons import kconj_symmetry_

import vayesta
import vayesta.ewf
from vayesta.misc import solids
from vayesta.core.fragmentation import IAOPAO_Fragmentation
from vayesta.core.fragmentation import IAOPAO_Fragmentation_UHF

parser = argparse.ArgumentParser()
parser.add_argument('--tm', choices=['Ni', 'Fe', 'Co', 'Mn'], default='Ni')
parser.add_argument('--load-mf')
parser.add_argument('--load-df')
parser.add_argument('--solver', choices=['CCSD', 'MP2', 'FCI'], default='CCSD')
parser.add_argument('--eta', type=float, default=1e-5)
parser.add_argument('--bath', choices=['DMET', 'MP2-BNO'], default='MP2-BNO')
parser.add_argument('--vr', type=float, nargs='*', default=[1.0, 0.9, 0.8])
parser.add_argument('--a-range', type=float, nargs=3)
parser.add_argument('--kmesh', type=int, nargs=3, default=[2,2,2])
parser.add_argument('--supercell', type=int, nargs=3)
parser.add_argument('--fragmentation')
parser.add_argument('--fragment-atoms', type=int, nargs='*', default=[0])
parser.add_argument('--fragment-orbitals', nargs='*')
parser.add_argument('--df', choices=['RSGDF', 'GDF'], default='RSGDF')
parser.add_argument('--kconj', type=int, choices=[0, 1], default=1)
parser.add_argument('--basis', default='def2-svp')
parser.add_argument('--preconv-mf', type=int, choices=[0, 1], default=0)
parser.add_argument('--auxbasis')
parser.add_argument('--exp-to-discard', type=float)
parser.add_argument('--xc')
parser.add_argument('--init-guess')
parser.add_argument('--uhf', action='store_true')
parser.add_argument('--newton', action='store_true')
parser.add_argument('--scf-max-cycle', type=int, default=100)
args = parser.parse_args()


# in Bohr^3
#v0 = {
#        'Mn' : 158.9,
#        'Fe' : 144.1,
#        'Co' : 137.0,
#        'Ni' : 128.0,
#    }#[args.tm]
#   Mn 4.5498343939777826
#   Fe 4.403949029865562
#   Co 4.330397991995583
#   Ni 4.233415999999999

def get_a(v):
    """Volume to lattice constant"""
    return (4*v)**(1.0/3) * 0.529177

if args.a_range is None:
    if args.tm == 'Ni':
        args.a_range = (3.0, 4.4, 0.2)
    elif args.tm == 'Fe':
        args.a_range = (3.0, 4.6, 0.2)
    elif args.tm == 'Co':
        args.a_range = (3.0, 4.6, 0.2)
    elif args.tm == 'Mn':
        args.a_range = (3.0, 4.8, 0.2)

if args.auxbasis is None:
    args.auxbasis = '%s-ri' % args.basis

def get_mf(cell, kpts):
    mod = pyscf.pbc.scf if (args.xc is None) else pyscf.pbc.dft
    cls = getattr(mod, '%s%s%s' % (
        ('' if kpts is None else 'K'),
        ('U' if args.uhf else 'R'),
        ('HF' if args.xc is None else 'KS')))
    if kpts is None:
        mf = cls(cell)
    else:
        mf = cls(cell, kpts)
    if args.xc is not None:
        mf.xc = args.xc
    return mf

# Loop relative volumes
dm0 = None
for a in np.arange(args.a_range[0], args.a_range[1]+1e-14, args.a_range[2])[::-1]:

    #v = vr*v0
    #a = get_a(v)
    #print("Lattice const for volume %f: %f" % (v, a))
    print("Lattice const= %f" % a)

    cell = pyscf.pbc.gto.Cell()
    cell.a, cell.atom = solids.rocksalt(atoms=[args.tm, 'O'], a=a)
    cell.basis = args.basis
    cell.output = 'pyscf.out'
    cell.verbose = 10
    if args.exp_to_discard:
        cell.exp_to_discard = args.exp_to_discard
    cell.build()

    if args.supercell is not None:
        cell = pyscf.pbc.tools.super_cell(cell, args.supercell)

    if np.product(args.kmesh) > 1:
        kpts = cell.make_kpts(args.kmesh)
    else:
        kpts = None

    mf = get_mf(cell, kpts)
    mf.chkfile = ('mf-%.2f.chk' % a)
    if args.df == 'RSGDF':
        mf = mf.rs_density_fit(auxbasis=args.auxbasis)
    else:
        mf = mf.density_fit(auxbasis=args.auxbasis)
    df = mf.with_df
    if args.load_df:
        df._cderi = (args.load_df % a)
        print("Loading CDERI from '%s'" % df._cderi)
        df.build(with_j3c=False)
    else:
        mf.with_df._cderi_to_save = ('cderi-%.2f.h5' % a)
        df.build(j_only=False)

    # (k,-k)-symmetry:
    if bool(args.kconj) and kpts is not None:
        if bool(args.preconv_mf):
            print("Preconverge mean-field without (k,-k)-symmetry")
            tol_orig = mf.conv_tol, mf.conv_tol_grad
            mf.conv_tol, mf.conv_tol_grad = 1e-4, 1e-2
            mf.kernel(dm0=dm0)
            mf.conv_tol, mf.conv_tol_grad = tol_orig
            dm0 = mf.make_rdm1()
        print("Running mean-field with (k,-k)-symmetry")
        mf = kconj_symmetry_(mf)

    mf.max_cycle = args.scf_max_cycle
    if args.newton:
        mf = mf.newton()

    # --- Init guess
    if args.load_mf:
        print("Loading SCF from '%s'" % (args.load_mf % a))
        dm0 = mf.from_chk(args.load_mf % a)
    elif dm0 is None and args.init_guess == 'af2':
        assert (np.product(args.supercell) == 8)
        def init_afm(dm0, delta=0.1):
            dm0 = dm0.copy()
            # 3d-orbitals
            da = cell.search_ao_label(['^%d %s 3d' % (i, args.tm) for i in [0, 2, 4, 6]])
            db = cell.search_ao_label(['^%d %s 3d' % (i, args.tm) for i in [8, 10, 12, 14]])
            if kpts is None:
                da = np.ix_(da, da)
                db = np.ix_(db, db)
            else:
                da = np.ix_(list(range(len(kpts))), da, da)
                db = np.ix_(list(range(len(kpts))), db, db)
            ddm = delta*dm0[0][da]
            # Sites A -> Majority spin=a
            dm0[0][da] += ddm
            dm0[1][da] -= ddm
            # Sites B -> Majority spin=b
            ddm = delta*dm0[1][db]
            dm0[0][db] -= ddm
            dm0[1][db] += ddm
            return dm0
        dm0 = mf.get_init_guess()
        print(dm0.shape)
        dm0 = init_afm(mf.get_init_guess())

    print("Running mean-field...")
    mf.kernel(dm0=dm0)
    print("Mean-field done.")
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

    emb.pop_analysis(dm1=dm0, filename='mf-pop-%.2f.txt' % a, full=True)
    emb.pop_analysis(dm1=dm0, filename='mf-pop-%.2f-iaopao.txt' % a, local_orbitals='iao+pao', full=True)

    # For Sz2
    if args.uhf:
        proj = IAOPAO_Fragmentation_UHF(emb.mf, emb.log)
    else:
        proj = IAOPAO_Fragmentation(emb.mf, emb.log)
    proj.kernel()

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
    frag_tm.pop_analysis(filename='pop-%.2f.txt' % a, full=True)
    frag_tm.pop_analysis(filename='pop-%.2f-iaopao.txt' % a, local_orbitals='iao+pao', full=True)

    s = emb.get_ovlp()

    def get_ssz(frag, atom1, atom2):
        name1, indices1 = proj.get_atomic_fragment_indices(atom1)
        name2, indices2 = proj.get_atomic_fragment_indices(atom2)

        # Projector...
        if args.uhf:
            f1a, f1b = proj.get_frag_coeff(indices1)
            f2a, f2b = proj.get_frag_coeff(indices2)
            ca, cb = frag.cluster.c_active
            p1a = np.linalg.multi_dot((ca.T, s, f1a, f1a.T, s, ca))
            p1b = np.linalg.multi_dot((cb.T, s, f1b, f1b.T, s, cb))
            p2a = np.linalg.multi_dot((ca.T, s, f2a, f2a.T, s, ca))
            p2b = np.linalg.multi_dot((cb.T, s, f2b, f2b.T, s, cb))
            p1 = (p1a, p1b)
            p2 = (p2a, p2b)
        else:
            f1 = proj.get_frag_coeff(indices1)
            f2 = proj.get_frag_coeff(indices2)
            c = frag.cluster.c_active
            p1 = np.linalg.multi_dot((c.T, s, f1, f1.T, s, c))
            p2 = np.linalg.multi_dot((c.T, s, f2, f2.T, s, c))

        sz = frag.get_cluster_sz(p1)
        ssz = frag.get_cluster_ssz(p1, p2)
        return sz, ssz

    sz, ssz = get_ssz(frag_tm, 0, 0)

    vol = (a**3 / 4.0)
    with open('results.txt', 'a') as f:
        fmt = '%.4f  %.4f  %5r  % .8f  % .8f  % .6f  % .6f\n'
        f.write(fmt % (a, vol, mf.converged, emb.e_mf, emb.e_tot, 2*sz, 2*np.sqrt(ssz)))
