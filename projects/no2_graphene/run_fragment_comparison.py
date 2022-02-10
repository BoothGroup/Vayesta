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
parser.add_argument('--kmesh', type=int, nargs=2, default=False)
parser.add_argument('--basis', default='cc-pVDZ')
parser.add_argument('--auxbasis', default='cc-pVDZ-ri')
parser.add_argument('--nc', type=int, default=4)
parser.add_argument('--supercell', type=int, nargs=2, default=[5, 5])
parser.add_argument('--vacuum-size', type=float, default=20.0)
parser.add_argument('--z-graphene', type=float, default=10.0)
parser.add_argument('--verbose', type=int, default=10)
parser.add_argument('--dimension', type=int, default=2)
parser.add_argument('--spin', type=int, default=1)
parser.add_argument('--lindep-threshold', type=float)
parser.add_argument('--scf-conv-tol', type=float, default=1e-9)
parser.add_argument('--fragment-type', default='iao')
parser.add_argument('--eta', type=float)
parser.add_argument('--augmented-basis', type=int, default=1)
args = parser.parse_args()

# Cell

cell = pyscf.pbc.gto.Cell()
no2_graphene = NO2_Graphene(args.supercell, vacuum_size=args.vacuum_size, z_graphene=args.z_graphene)
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
    mf = pyscf.pbc.scf.UHF(cell)
mf.conv_tol = args.scf_conv_tol
mf.max_cycle = 100

mf = mf.density_fit(auxbasis=args.auxbasis)

# Run MF
mf.kernel()

if not mf.converged:
    print("MF not converged!")

print("E(MF)= % 16.8f Ha" % mf.e_tot)
dm1_mf = mf.make_rdm1()

np.save('dm1-mf.npy', dm1_mf)

# --- Embedding

if args.eta is None:
    ecc = vayesta.ewf.EWF(mf, bath_type=None, make_rdm1=True)
else:
    ecc = vayesta.ewf.EWF(mf, bno_threshold=args.eta, make_rdm1=True)

ecc.pop_analysis(dm1=dm1_mf, local_orbitals='mulliken', filename='mulliken-mf.pop')
ecc.pop_analysis(dm1=dm1_mf, local_orbitals='lowdin',   filename='lowdin-mf.pop')
ecc.pop_analysis(dm1=dm1_mf, local_orbitals='iao+pao',  filename='iao+pao-mf.pop')

if args.fragment_type == 'iao':
    ecc.iao_fragmentation()
elif args.fragment_type == 'iaopao':
    ecc.iaopao_fragmentation()
elif args.fragment_type == 'sao':
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
ecc.kernel()

nactive = max(frag.results.n_active)
frag.pop_analysis(local_orbitals='mulliken', filename='mulliken-cc-%d.pop' % nactive)
frag.pop_analysis(local_orbitals='lowdin',   filename='lowdin-cc-%d.pop' % nactive)
frag.pop_analysis(local_orbitals='iao+pao',  filename='iao+pao-cc-%d.pop' % nactive)

with open('energies.txt', 'a') as f:
    f.write('% 16.8f  %3d  % 16.8f\n' % (mf.e_tot, nactive, ecc.e_tot))
