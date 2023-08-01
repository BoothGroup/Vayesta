import numpy as np

import pyscf
import pyscf.scf

import vayesta
import vayesta.ewf
from vayesta.misc.molecules import ring
# Plot the spectrum
from dyson.util import build_spectral_function
import matplotlib.pyplot as plt

natom = 120
nfrag = 1
maxiter = 30
nmom_max_fci = (8,8)
nmom_max_bath=1



for a in [1]:

    mol = pyscf.gto.Mole()
    mol.atom = ring('H', natom, a)
    mol.basis = 'sto-3g'
    mol.output = 'pyscf.out'
    mol.verbose = 4
    mol.build()

    # Hartree-Fock
    mf = pyscf.scf.RHF(mol)
    mf.kernel()


    emb = vayesta.ewf.EWF(mf, solver='FCI', bath_options=dict(bathtype='ewdmet', max_order=1, order=1, dmet_threshold=1e-10), solver_options=dict(conv_tol=1e-12, init_guess='cisd', n_moments=nmom_max_fci))
    with emb.site_fragmentation() as f:
        with f.rotational_symmetry(order=int(natom/nfrag), axis='z') as rot:
            f.add_atomic_fragment(range(nfrag))

    emb.qpewdmet_scmf(maxiter=maxiter, proj=1)
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
    with emb.site_fragmentation() as f:
        with f.rotational_symmetry(order=int(natom/nfrag), axis='z') as rot:
            f.add_atomic_fragment(range(nfrag))
    emb.qpewdmet_scmf(maxiter=maxiter, proj=2)
    emb.kernel()
    gf, gf_qp = emb.with_scmf.get_greens_function()

    sf_dynamic = build_spectral_function(gf.energies, gf.couplings, grid, eta=0.1)
    sf_static = build_spectral_function(gf_qp.energies, gf_qp.couplings, grid, eta=0.1)
    ax[0].plot(grid, sf_dynamic, "m-", label="QP-EwDMET (2 Proj, dynamic)")
    ax[1].plot(grid, sf_static, "y-", label="QP-EwDMET (2, Proj, static)")
    

    ax[0].set_title('a = %d'%a)
    ax[0].legend()
    ax[1].legend()
    
    plt.savefig("H-ring.png")