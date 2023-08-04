import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.mcscf

import vayesta
import vayesta.ewf

mol = pyscf.gto.Mole()
mol.atom = ['N 0 0 0', 'N 0 0 2']
mol.basis = 'aug-cc-pvdz'
mol.output = 'pyscf.out'
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

def get_emb_result(ansatz, bathtype='full'):
    # Uses fastest available solver for given ansatz; PySCF if available, otherwise ebCC
    emb = vayesta.ewf.EWF(mf, solver=ansatz, bath_options=dict(bathtype=bathtype),
                          solver_options=dict(solve_lambda=False))
    # Both these alternative specifications will always use an ebCC solver.
    #emb = vayesta.ewf.EWF(mf, solver=f'EB{ansatz}', bath_options=dict(bathtype=bathtype),
    #                      solver_options=dict(solve_lambda=False))
    #emb = vayesta.ewf.EWF(mf, solver='EBCC', bath_options=dict(bathtype=bathtype),
    #                      solver_options=dict(solve_lambda=False, ansatz=ansatz))

    with emb.iao_fragmentation() as f:
        with f.rotational_symmetry(2, "y", center=(0, 0, 1)):
            f.add_atomic_fragment(0)
    emb.kernel()
    return emb.e_tot

e_ccsd = get_emb_result('CCSD', 'full')
e_ccsdt = get_emb_result('CCSDT', 'dmet')
e_ccsdtprime = get_emb_result("CCSDt'", 'full')

print("E(HF)=                                           %+16.8f Ha" % mf.e_tot)
print("E(CASCI)=                                        %+16.8f Ha" % casci.e_tot)
print("E(CASSCF)=                                       %+16.8f Ha" % casscf.e_tot)
print("E(CCSD, complete)=                               %+16.8f Ha" % e_ccsd)
print("E(emb. CCSDT, DMET CAS)=                         %+16.8f Ha" % e_ccsdt)
print("E(emb. CCSDt', complete+DMET active space)=      %+16.8f Ha" % e_ccsdtprime)
