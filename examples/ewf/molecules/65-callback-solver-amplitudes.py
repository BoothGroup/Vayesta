import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.fci
import vayesta
import vayesta.ewf
from vayesta.misc.molecules import ring
from vayesta.core.types.wf.t_to_c import t1_rhf, t2_rhf

# User defined FCI solver - takes pyscf mf as input and returns RDMs
# The mf argment contains the hamiltonain in the orthonormal cluster basis
# Pyscf or other solvers may be used to solve the cluster problem and may return RDMs, CISD amplitudes or CCSD amplitudes
def cisd_solver(mf):
    ci = pyscf.ci.CISD(mf)
    energy, civec = ci.kernel()
    c0, c1, c2 = ci.cisdvec_to_amplitudes(civec)

    # To use CI amplitudues use return the following line and set energy_functional='wf' to use the projected energy in the EWF arguments below
    # return dict(c0=c0, c1=c1, c2=c2, converged=True, energy=ci.e_corr)

    # Convert CISD amplitudes to CCSD amplitudes to be able to make use of the patitioned cumulant energy functional
    t1 = t1_rhf(c1/c0) 
    t2 = t2_rhf(t1, c2/c0)
    return dict(t1=t1, t2=t2, l1=t1, l2=t2, converged=True, energy=ci.e_corr)

def ccsd_solver(mf, dm=False):
    if type(mf.mo_coeff) == tuple:
        cc = UCCSD(mf)
    else:
        cc = pyscf.cc.CCSD(mf)
    cc.kernel()
    t1, t2 = cc.t1, cc.t2
    l1, l2 = cc.solve_lambda()
    return dict(t1=t1, t2=t2, l1=l1, l2=l2, converged=True, energy=cc.e_corr)

def fci_solver(mf, dm=False):
    h1e = mf.get_hcore()
    h2e = mf._eri
    norb = h1e[0].shape[-1]
    nelec = mf.mol.nelec
    energy, civec = pyscf.fci.direct_spin0.kernel(h1e, h2e, norb, nelec, conv_tol=1.e-14)
    return dict(civec=civec, converged=True, energy=energy)

natom = 10
mol = pyscf.gto.Mole()
mol.atom = ring("H", natom, 1.5)
mol.basis = "sto-3g"
mol.output = "pyscf.out"
mol.verbose = 5
mol.symmetry = True
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# CISD
cisd = pyscf.ci.CISD(mf)
cisd.kernel()

# CCSD
ccsd = pyscf.cc.CCSD(mf)
ccsd.kernel()

# FCI
fci = pyscf.fci.FCI(mf)
fci.kernel()

# Vayesta options
use_sym = True
nfrag = 1 
bath_opts = dict(bathtype="dmet") 


def init_frag(emb):
    # Set up fragments
    with emb.iao_fragmentation() as f:
        if use_sym:
            # Add rotational symmetry
            with f.rotational_symmetry(order=natom//nfrag, axis=[0, 0, 1]):
                f.add_atomic_fragment(range(nfrag))
        else:
            # Add all atoms as separate fragments 
            f.add_all_atomic_fragments()
    return emb

# Run vayesta with user defined CISD solver
emb_ci = vayesta.ewf.EWF(mf, solver="CALLBACK",  energy_functional='dm-t2only', bath_options=bath_opts, solver_options=dict(callback=cisd_solver))
emb_ci = init_frag(emb_ci)  
emb_ci.kernel()

# Run vayesta with user defined CCSD solver
emb_cc = vayesta.ewf.EWF(mf, solver="CALLBACK",  energy_functional='dm-t2only', bath_options=bath_opts, solver_options=dict(callback=ccsd_solver))
emb_cc = init_frag(emb_cc)
emb_cc.kernel()

# Run vayesta with user defined FCI solver
emb_fci = vayesta.ewf.EWF(mf, solver="CALLBACK",  energy_functional='dm-t2only', bath_options=bath_opts, solver_options=dict(callback=fci_solver))
emb_fci = init_frag(emb_fci)
emb_fci.kernel()

print("Hartree-Fock energy            : %s"%mf.e_tot)
print("CISD Energy                    : %s"%cisd.e_tot)
print("Emb. CCSD Partitioned Cumulant : %s"%emb_ci.e_tot)
print("CCSD Energy                    : %s"%ccsd.e_tot)
print("Emb. CCSD Partitioned Cumulant : %s"%emb_cc.e_tot)
print("FCI  Energy                    : %s"%fci.e_tot)
print("Emb. FCI  Partitioned Cumulant : %s"%emb_fci.e_tot)
