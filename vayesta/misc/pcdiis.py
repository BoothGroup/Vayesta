import numpy as np

import pyscf
import pyscf.lib


class PCDIIS(pyscf.lib.diis.DIIS):
    """https://doi.org/10.1021/acs.jctc.7b00892"""

    def __init__(self, pref, *args, **kwargs):
        self.pref = pref
        super().__init__(*args, **kwargs)


    def update(self, x):
        y = np.dot(x, self.pref)
        y_new = super().update(y)
        yy_inv = np.linalg.inv(np.dot(y_new.T.conj(), y_new))
        x_new = np.linalg.multi_dot((y_new, yy_inv, y_new.T.conj()))
        return x_new


if __name__ == '__main__':
    import pyscf.gto
    import pyscf.scf

    mol = pyscf.gto.Mole(atom='H 0 0 0 ; F 0 0 1', basis='cc-pVDZ')
    mol.build()
    hf = pyscf.scf.RHF(mol)
    hf.kernel()

    nocc = np.count_nonzero(hf.mo_occ > 0)
    c = hf.mo_coeff[:,:nocc]
    diis = PCDIIS(c)
    dm0 = hf.make_rdm1()
    dm1 = diis.update(dm0)
    # Missing overlap matrix?
    assert np.allclose(dm0, dm1)
