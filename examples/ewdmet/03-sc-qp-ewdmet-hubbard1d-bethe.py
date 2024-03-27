import numpy as np
import vayesta
import vayesta.ewf
from vayesta.lattmod import Hubbard1D, LatticeRHF
from vayesta.lattmod.bethe import hubbard1d_bethe_gap
from dyson import Lehmann, FCI, CCSD

u = 3
natom = 128
nelec = natom
nfrag = 2
nmom_max_bath = 1
nmom_max_fci = (4,4)
solv = 'FCI'
EXPR = FCI if solv=='FCI' else CCSD
hubbard = Hubbard1D(nsite=natom, nelectron=nelec, hubbard_u=u, verbose=0)
mf = LatticeRHF(hubbard)
mf.kernel()

chempot = (mf.mo_energy[natom//2-1] + mf.mo_energy[natom//2] ) / 2
gf_hf = Lehmann(mf.mo_energy, np.eye(mf.mo_coeff.shape[0]), chempot=chempot)


emb = vayesta.ewf.EWF(mf, solver=solv, bath_options=dict(bathtype='ewdmet', max_order=nmom_max_bath, order=nmom_max_bath, dmet_threshold=1e-12), solver_options=dict(conv_tol=1e-12, n_moments=nmom_max_fci))
emb.qpewdmet_scmf(proj=2, maxiter=10)
nimages = [natom//nfrag, 1, 1]
emb.symmetry.set_translations(nimages)
with emb.site_fragmentation() as f:
    f.add_atomic_fragment(list(range(nfrag)))
emb.kernel()

gap = lambda gf: gf.physical().virtual().energies[0] - gf.physical().occupied().energies[-1]
emb_dynamic_gap = gap(emb.with_scmf.gf)
emb_static_gap = gap(emb.with_scmf.gf_qp)
hf_gap = gap(gf_hf)
bethe_gap = hubbard1d_bethe_gap(1,u, interval=(1,200))

print("Ran for %s iterations"%emb.with_scmf.iteration)
print("Bethe ansatz gap: %s "%(bethe_gap))
print("Hartree-Fock gap: %s"%(hf_gap))
print("Dynamic GF gap:   %s"%(emb_dynamic_gap))
print("Static GF gap:    %s"%(emb_static_gap))
