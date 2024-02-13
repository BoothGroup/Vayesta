import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.mcscf

import vayesta
import vayesta.ewf

mol = pyscf.gto.Mole()
mol.atom = ["N 0 0 0", "N 0 0 2"]
mol.basis = "cc-pvdz"
mol.output = "pyscf.out"
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Reference CASCI
casci = pyscf.mcscf.CASCI(mf, 8, 10)
casci.kernel()

# Reference CASSCF
casscf = pyscf.mcscf.CASSCF(mf, 8, 10)
casscf.kernel()

# Both these alternative specifications will always use an ebcc solver.
# emb = vayesta.ewf.EWF(mf, solver=f'EB{ansatz}',  solver_options=dict(solve_lambda=False))
# emb = vayesta.ewf.EWF(mf, solver='ebcc', solver_options=dict(solve_lambda=False, ansatz=ansatz))

def get_emb_result(ansatz, bathtype="full"):
    # Uses fastest available solver for given ansatz; PySCF if available, otherwise ebcc.
    emb = vayesta.ewf.EWF(
        mf, solver=ansatz, bath_options=dict(bathtype=bathtype), solver_options=dict(solve_lambda=False)
    )
    with emb.iao_fragmentation() as f:
        with f.rotational_symmetry(2, "y", center=(0, 0, 1)):
            f.add_atomic_fragment(0)
    emb.kernel()
    return emb.e_tot


e_ccsd = get_emb_result("CCSD", "full")
e_CCSDT = get_emb_result("CCSDT", "dmet")
e_ccsdtprime = get_emb_result("CCSDt'", "full")


# CCSDt will set DMET orbitals as active space if no CAS is specified. To use a CAS, we must specify it explicitly.

# embfull = vayesta.ewf.EWF(mf, solver='ebcc', bath_options=dict(bathtype='full'), solver_options=dict(solve_lambda=False))
# for ffull, fact in zip( embfull.loop(), emb.loop() ):
#     f.set_cas(c_occ=ffull.cluster.c_active_occ, c_vir=ffull.cluster.c_active_vir)
# emb.kernel()
e_CCSDt_ = get_emb_result("CCSDt", "dmet")
e_CCSDt = get_emb_result("CCSDt", "full")

print("E(HF)=                                           %+16.8f Ha" % mf.e_tot)
print("E(CASCI)=                                        %+16.8f Ha" % casci.e_tot)
print("E(CASSCF)=                                       %+16.8f Ha" % casscf.e_tot)
print("E(CCSD, complete)=                               %+16.8f Ha" % e_ccsd)
print("E(emb. CCSDT, DMET CAS)=                         %+16.8f Ha" % e_CCSDT)
print("E(emb. CCSDt, DMET CAS)=                         %+16.8f Ha" % e_CCSDt_)
print("E(emb. CCSDt', complete+DMET active space)=      %+16.8f Ha" % e_ccsdtprime)
print("E(emb. CCSDt, complete+DMET active space)=       %+16.8f Ha" % e_CCSDt)
