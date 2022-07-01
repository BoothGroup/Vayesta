import copy
import logging

import numpy as np
import scipy
import scipy.linalg

log = logging.getLogger(__name__)


def update_mf(mf, dm1, canonicalize=True, inplace=False, damping=0.0, diis=None):
    """p-DMET mean-field update."""

    if not inplace:
        mf = copy.copy(mf)
    if damping > 0:
        dm1 = (1-damping)*dm1 + damping*mf.make_rdm1()
    if diis:
        dm1 = diis.update(dm1)
    symerr = abs(dm1-dm1.T).max()
    if symerr > 1e-10:
        log.warning("Density-matrix not symmetric! L(inf)= %.3e", symerr)

    ovlp = mf.get_ovlp()
    mo_occ, mo_coeff = scipy.linalg.eigh(dm1, b=ovlp, type=2)
    mo_occ, mo_coeff = mo_occ[::-1], mo_coeff[:,::-1]
    nocc = np.count_nonzero(mf.mo_occ > 0)
    log.info("p-DMET occupation numbers:\n%r", mo_occ)
    occ_homo = mo_occ[nocc-1]
    occ_lumo = mo_occ[nocc]
    if abs(occ_homo - occ_lumo) < 1e-8:
        raise RuntimeError("Degeneracy in MO occupation.")

    if canonicalize:
        fock = mf.get_fock()
        occ, vir = np.s_[:nocc], np.s_[nocc:]
        foo = np.linalg.multi_dot((mo_coeff[:,occ].T, fock, mo_coeff[:,occ]))
        eo, ro = np.linalg.eigh(foo)
        log.debug("Occupied eigenvalues:\n%r", eo)
        mo_coeff[:,occ] = np.dot(mo_coeff[:,occ], ro)
        fvv = np.linalg.multi_dot((mo_coeff[:,vir].T, fock, mo_coeff[:,vir]))
        ev, rv = np.linalg.eigh(fvv)
        log.debug("Virtual eigenvalues:\n%r", ev)
        mo_coeff[:,vir] = np.dot(mo_coeff[:,vir], rv)

    mf.mo_coeff = mo_coeff
    mf.mo_energy = None
    mf.e_tot = mf.energy_tot()
    return mf


if __name__ == '__main__':
    import pyscf
    import pyscf.gto
    import pyscf.scf
    import pyscf.cc

    logging.basicConfig(level=logging.DEBUG)

    mol = pyscf.gto.Mole(atom='H 0 0 0 ; F 0 0 2', basis='cc-pvdz')
    mol.build()

    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    log.debug("HF eigenvalues:\n%r", mf.mo_energy)

    cc = pyscf.cc.CCSD(mf)
    cc.kernel()

    dm1 = cc.make_rdm1(ao_repr=True)
    mf = update_mf(mf, dm1)
