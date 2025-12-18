import numpy as np

import vayesta
import vayesta.egf
from vayesta.lattmod import Hubbard1D, LatticeRHF
from dyson import MBLGF, AuxiliaryShift, FCI, Lehmann, Spectral

nsite = 10
nelec = nsite
u = 3
nfrag = 2

nmom_max_fci = (4,4)
nmom_max_bath=1

hubbard = Hubbard1D(nsite=nsite, nelectron=nelec, hubbard_u=u, verbose=0)
mf = LatticeRHF(hubbard)
mf.kernel()

# Full system FCI GF
expr = FCI.hole.from_mf(mf)
th = expr.build_gf_moments(nmom_max_fci[0]) 
expr = FCI.particle.from_mf(mf)
tp = expr.build_gf_moments(nmom_max_fci[1])

solverh = MBLGF(th)
solverh.kernel()
solverp = MBLGF(tp)
solverp.kernel()

result = Spectral.combine_dyson(solverh.result, solverp.result)
se = result.get_self_energy()

static_potential = se.as_static_potential(mf.mo_energy, eta=1e-2)
gf = result.get_greens_function()
dm = gf.occupied().moment(0) * 2.0
nelec_gf = np.trace(dm)
print("Exact GF nelec: %s"%nelec_gf)

sc = mf.get_ovlp() @ mf.mo_coeff
new_fock = sc @ (th[1] + tp[1] + static_potential) @ sc.T
e, mo_coeff = np.linalg.eigh(new_fock) 
chempot = (e[nelec//2-1] + e[nelec//2] ) / 2
gf_static = Lehmann(e, mo_coeff, chempot=chempot)

gap = lambda gf: gf.physical().virtual().energies[0] - gf.physical().occupied().energies[-1]
dynamic_gap = gap(gf)
static_gap = gap(gf_static)

# QP-EwDMET GF
opts = dict(proj=1, use_sym=False)
emb = vayesta.egf.EGF(mf, solver='FCI', **opts, bath_options=dict(bathtype='dmet', dmet_threshold=1e-12), solver_options=dict(conv_tol=1e-15, n_moments=nmom_max_fci))
nimages = [nsite//nfrag, 1, 1]
emb.symmetry.set_translations(nimages)
with emb.site_fragmentation() as f:
    f.add_atomic_fragment(range(nfrag))
emb.kernel()

emb_dynamic_gap = gap(emb.gf)
print("Dynamic gap, FCI: %s  Emb: %s"%(dynamic_gap, emb_dynamic_gap))
