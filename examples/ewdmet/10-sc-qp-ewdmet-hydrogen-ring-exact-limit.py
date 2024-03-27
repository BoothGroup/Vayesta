import numpy as np
import scipy

import pyscf
import pyscf.scf

import vayesta
import vayesta.ewf
from vayesta.misc.molecules import ring
from dyson import MBLGF, AuxiliaryShift, FCI, MixedMBLGF, NullLogger, Lehmann
from dyson.util import build_spectral_function

a = 1
natom = 10
nfrag = 1
maxiter = 100
nmom_max_fci = (4,4)
nmom_max_bath=1

mol = pyscf.gto.Mole()
mol.atom = ring('H', natom, a)
mol.basis = 'sto-3g'
mol.output = 'pyscf.out'
mol.verbose = 4
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()
assert mf.converged
chempot = (mf.mo_energy[natom//2-1] + mf.mo_energy[natom//2] ) / 2
gf_hf = Lehmann(mf.mo_energy, np.eye(mf.mo_coeff.shape[0]), chempot=chempot)

# Full system FCI GF
expr = FCI["1h"](mf)
th = expr.build_gf_moments(nmom_max_fci[0]) 
expr = FCI["1p"](mf)
tp = expr.build_gf_moments(nmom_max_fci[1])

solverh = MBLGF(th, log=NullLogger())
solverp = MBLGF(tp, log=NullLogger())
solver = MixedMBLGF(solverh, solverp)
solver.kernel()
se = solver.get_self_energy()
solver = AuxiliaryShift(th[1]+tp[1], se, natom, log=NullLogger())
solver.kernel()
static_potential = se.as_static_potential(mf.mo_energy, eta=1e-2)
gf = solver.get_greens_function()
dm = gf.occupied().moment(0) * 2.0
nelec_gf = np.trace(dm)
print("Exact GF nelec: %s"%nelec_gf)

sc = mf.get_ovlp() @ mf.mo_coeff
new_fock = sc @ (th[1] + tp[1] + static_potential) @ sc.T
e, mo_coeff = scipy.linalg.eigh(new_fock, mf.get_ovlp()) 
chempot = (e[natom//2-1] + e[natom//2] ) / 2
gf_static = Lehmann(e, np.eye(mf.mo_coeff.shape[0]), chempot=chempot)

gap = lambda gf: gf.physical().virtual().energies[0] - gf.physical().occupied().energies[-1]
dynamic_gap = gap(gf)
static_gap = gap(gf_static)

# QP-EwDMET GF
emb = vayesta.ewf.EWF(mf, solver='FCI', bath_options=dict(bathtype='full', dmet_threshold=1e-12), solver_options=dict(conv_tol=1e-15, n_moments=nmom_max_fci))
emb.qpewdmet_scmf(proj=1, maxiter=maxiter)
with emb.site_fragmentation() as f:
    with f.rotational_symmetry(order=int(natom/nfrag), axis='z') as rot:
        f.add_atomic_fragment(range(nfrag))
emb.kernel()

hf_gap = gap(gf_hf)
emb_dynamic_gap = gap(emb.with_scmf.gf)
emb_static_gap = gap(emb.with_scmf.gf_qp)
print("Ran for %s iterations"%emb.with_scmf.iteration)
print("Hartree-Fock gap: %s"%hf_gap)
print("Dynamic gap, FCI: %s  Emb: %s"%(dynamic_gap, emb_dynamic_gap))
print("Static  gap, FCI: %s  Emb: %s"%(static_gap, emb_static_gap))