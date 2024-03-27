import numpy as np

import vayesta
import vayesta.ewf
from vayesta.lattmod import Hubbard1D, LatticeRHF

# Plot the spectrum
from dyson.util import build_spectral_function
import matplotlib.pyplot as plt

nsite = 12
nelec = nsite
nfrag = 2
u = 3

nmom_max_fci = (4,4)
nmom_max_bath=1


hubbard = Hubbard1D(nsite=nsite, nelectron=nelec, hubbard_u=u, verbose=0)
mf = LatticeRHF(hubbard)
mf.kernel()

emb = vayesta.ewf.EWF(mf, solver='FCI', bath_options=dict(bathtype='ewdmet', max_order=1, order=1, dmet_threshold=1e-10), solver_options=dict(conv_tol=1e-12, init_guess='cisd', n_moments=nmom_max_fci))
nimages = [nsite//nfrag, 1, 1]
emb.symmetry.set_translations(nimages)
with emb.site_fragmentation() as f:
    f.add_atomic_fragment(list(range(nfrag)))
emb.qpewdmet_scmf(maxiter=100, proj=1)
emb.kernel()
gf, gf_qp = emb.with_scmf.get_greens_function()

fig, ax = plt.subplots(2,1, figsize=(16,9))
    
grid = np.linspace(-5, 11, 1024)
sf_hf = build_spectral_function(mf.mo_energy, np.eye(mf.mo_occ.size), grid, eta=0.1)
ax[0].plot(grid, sf_hf, 'r-', label='HF')
ax[1].plot(grid, sf_hf, 'r-', label='HF')
    
sf_dynamic = build_spectral_function(gf.energies, gf.couplings, grid, eta=0.1)
sf_static = build_spectral_function(gf_qp.energies, gf_qp.couplings, grid, eta=0.1)
ax[0].plot(grid, sf_dynamic, "b-", label="QP-EwDMET (1 Proj, dynamic)")
ax[1].plot(grid, sf_static, "g-", label="QP-EwDMET (1 Proj, static)")
    

emb = vayesta.ewf.EWF(mf, solver='FCI', bath_options=dict(bathtype='ewdmet', max_order=1, order=1, dmet_threshold=1e-10), solver_options=dict(conv_tol=1e-12, init_guess='cisd', n_moments=nmom_max_fci))
nimages = [nsite//nfrag, 1, 1]
emb.symmetry.set_translations(nimages)
with emb.site_fragmentation() as f:
    f.add_atomic_fragment(list(range(nfrag)))
emb.qpewdmet_scmf(maxiter=100, proj=2)
emb.kernel()
gf, gf_qp = emb.with_scmf.get_greens_function()

sf_dynamic = build_spectral_function(gf.energies, gf.couplings, grid, eta=0.1)
sf_static = build_spectral_function(gf_qp.energies, gf_qp.couplings, grid, eta=0.1)
ax[0].plot(grid, sf_dynamic, "m-", label="QP-EwDMET (2 Proj, dynamic)")
ax[1].plot(grid, sf_static, "y-", label="QP-EwDMET (2, Proj, static)")
    

ax[0].set_title('U = %d'%u)
ax[0].legend()
ax[1].legend()

plt.savefig("hubbard_spectral_function.png")