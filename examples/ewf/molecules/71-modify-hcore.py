import numpy as np

import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc

import vayesta
import vayesta.ewf

mol = pyscf.gto.Mole()
mol.atom = """
O  0.0000   0.0000   0.1173
H  0.0000   0.7572  -0.4692
H  0.0000  -0.7572  -0.4692
"""
mol.basis = 'cc-pVDZ'
mol.output = 'pyscf.out'
mol.build()

def get_v_mod(mf, shift):
    """Get potential on oxygen atom."""
    ovlp = mf.get_ovlp()
    # Construct Lowdin orbitals:
    e, v = np.linalg.eigh(ovlp)
    c_sao = np.dot(v*(e**-0.5), v.T)
    # Get projector onto oxygen-SAO space:
    oxygen = [ao[1].startswith('O') for ao in mf.mol.ao_labels(None)]
    sc = np.dot(ovlp, c_sao[:,oxygen])
    v_mod = shift*np.dot(sc, sc.T)
    return v_mod

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
# Save original H_core function and overwrite mean-field
hcore_orig = mf.get_hcore
# Shift oxygen atom by -1 Hartree -> electrons move to oxygen
v_mod = get_v_mod(mf, -1.0)
mf.get_hcore = (lambda mf, *args : hcore_orig(*args) + v_mod).__get__(mf)
mf.kernel()

# get_hcore_for_energy must be overwritten with original H_core function,
# if the energy should be calculated without the shift
emb = vayesta.ewf.EWF(mf, bno_threshold=1e-6, store_dm1=True,
        overwrite=dict(get_hcore_for_energy=(lambda emb, *args : hcore_orig())))
emb.iao_fragmentation()
emb.add_all_atomic_fragments()
emb.kernel()

emb.fragments[0].pop_analysis()
