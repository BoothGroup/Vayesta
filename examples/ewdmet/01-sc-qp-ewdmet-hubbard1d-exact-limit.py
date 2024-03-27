import numpy as np

import vayesta
import vayesta.ewf
from vayesta.lattmod import Hubbard1D, LatticeRHF

from dyson import MBLGF, AuxiliaryShift, FCI, MixedMBLGF, NullLogger, Lehmann

nsite = 10
nelec = nsite
u = 6
nfrag = 2

nmom_max_fci = (4,4)
nmom_max_bath=1




hubbard = Hubbard1D(nsite=nsite, nelectron=nelec, hubbard_u=u, verbose=0)
mf = LatticeRHF(hubbard)
mf.kernel()

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
e, mo_coeff = np.linalg.eigh(new_fock) 
chempot = (e[natom//2-1] + e[natom//2] ) / 2
gf_static = Lehmann(e, mo_coeff, chempot=chempot)

gap = lambda gf: gf.physical().virtual().energies[0] - gf.physical().occupied().energies[-1]
dynamic_gap = gap(gf)
static_gap = gap(gf_static)

# QP-EwDMET GF
opts = dict(sc=False, store_hist=True, aux_shift=True, store_scfs=True, diis=True, damping=0, static_potential_conv_tol=1e-6, use_sym=False, eta=1e-2)
emb = vayesta.ewf.EWF(mf, solver='FCI', bath_options=dict(bathtype='full', dmet_threshold=1e-12), solver_options=dict(conv_tol=1e-15, n_moments=nmom_max_fci))
emb.qpewdmet_scmf(proj=1, maxiter=1000, **opts)
with emb.site_fragmentation() as f:
    f.add_atomic_fragment(range(nfrag))
emb.kernel()

emb_dynamic_gap = gap(emb.with_scmf.gf)
emb_static_gap = gap(emb.with_scmf.gf_qp)
print("Ran for %s iterations"%emb.with_scmf.iteration)
print("Dynamic gap, FCI: %s  Emb: %s"%(dynamic_gap, emb_dynamic_gap))
print("Static  gap, FCI: %s  Emb: %s"%(static_gap, emb_static_gap))