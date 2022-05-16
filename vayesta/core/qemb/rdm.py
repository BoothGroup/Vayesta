"""Democratically partitioned RDMs"""

import numpy as np

from vayesta.core.util import *


def make_rdm1_demo_rhf(emb, ao_basis=False, with_mf=True, symmetrize=True):
    """Make democratically partitioned one-particle reduced density-matrix from fragment calculations.

    Warning: A democratically partitioned DM is only expected to yield reasonable results
    for full fragmentations (eg, Lowdin-AO or IAO+PAO fragmentation).

    Parameters
    ----------
    ao_basis: bool, optional
        Return the density-matrix in the AO basis. Default: False.
    with_mf: bool, optional
        Add the mean-field contribution to the density-matrix (double counting is accounted for).
        Is only used if `partition = 'dm'`. Default: False.
    symmetrize: bool, optional
        Symmetrize the density-matrix at the end of the calculation. Default: True.

    Returns
    -------
    dm1: (n, n) array
        One-particle reduced density matrix in AO (if `ao_basis=True`) or MO basis (default).
    """
    ovlp = emb.get_ovlp()
    mo_coeff = emb.mo_coeff
    dm1 = np.zeros((emb.nmo, emb.nmo))
    if with_mf is True:
        dm1[np.diag_indices(emb.nocc)] = 2
    for f in emb.fragments:
        emb.log.debugv("Now adding projected DM of fragment %s", f)
        ddm = f.results.wf.make_rdm1(with_mf=False)
        cf = f.cluster.c_active
        rf = dot(mo_coeff.T, ovlp, cf)
        pf = f.get_fragment_projector(cf)
        dm1 += einsum('xi,ij,px,qj->pq', pf, ddm, rf, rf)
    if symmetrize:
        dm1 = (dm1 + dm1.T)/2
    if ao_basis:
        dm1 = dot(mo_coeff, dm1, mo_coeff.T)
    return dm1

def make_rdm1_demo_uhf(emb, ao_basis=False, with_mf=True, symmetrize=True):
    """Make democratically partitioned one-particle reduced density-matrix from fragment calculations.

    Warning: A democratically partitioned DM is only expected to yield reasonable results
    for full fragmentations (eg, Lowdin-AO or IAO+PAO fragmentation).

    Parameters
    ----------
    ao_basis: bool, optional
        Return the density-matrix in the AO basis. Default: False.
    with_mf: bool, optional
        Add the mean-field contribution to the density-matrix (double counting is accounted for).
        Is only used if `partition = 'dm'`. Default: False.
    symmetrize: bool, optional
        Symmetrize the density-matrix at the end of the calculation. Default: True.

    Returns
    -------
    dm1: tuple of (n, n) arrays
        Alpha- and beta one-particle reduced density matrix in AO (if `ao_basis=True`) or MO basis (default).
    """
    ovlp = emb.get_ovlp()
    mo_coeff = emb.mo_coeff
    dm1a = np.zeros((emb.nmo[0], emb.nmo[0]))
    dm1b = np.zeros((emb.nmo[1], emb.nmo[1]))
    if with_mf is True:
        dm1a[np.diag_indices(emb.nocc[0])] = 1
        dm1b[np.diag_indices(emb.nocc[1])] = 1
    for f in emb.fragments:
        emb.log.debugv("Now adding projected DM of fragment %s", f)
        ddma, ddmb = f.results.wf.make_rdm1(with_mf=False)
        cf = f.cluster.c_active
        rfa = dot(mo_coeff[0].T, ovlp, cf[0])
        rfb = dot(mo_coeff[1].T, ovlp, cf[1])
        pfa, pfb = f.get_fragment_projector(cf)
        dm1a += einsum('xi,ij,px,qj->pq', pfa, ddma, rfa, rfa)
        dm1b += einsum('xi,ij,px,qj->pq', pfb, ddmb, rfb, rfb)
    if symmetrize:
        dm1a = (dm1a + dm1a.T)/2
        dm1b = (dm1b + dm1b.T)/2
    if ao_basis:
        dm1a = dot(mo_coeff[0], dm1a, mo_coeff[0].T)
        dm1b = dot(mo_coeff[1], dm1b, mo_coeff[1].T)
    return (dm1a, dm1b)

# --- Two-particle
# ----------------

def make_rdm2_demo_rhf(emb, ao_basis=False, with_mf=True, with_dm1=True, approx_cumulant=True, dmet_dm2=True,
        symmetrize=True):
    """Make democratically partitioned two-particle reduced density-matrix from fragment calculations.

    Warning: A democratically partitioned DM is only expected to yield reasonable results
    for full fragmentations (eg. Lowdin-AO (SAO) or IAO+PAO fragmentation).

    Energies can be evaluated as follows from the 1-DM and 2-DM:

    1) Literature DMET energy:
    >>> hcore = mf.get_hcore()
    >>> eris = pyscf.ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape([mol.nao]*4)
    >>> ddm1 = emb.make_rdm1_demo(ao_basis=True, with_mf=False)
    >>> dm2 = emb.make_rdm2_demo(ao_basis=True, approx_cumulant=True, dmet_dm2=True)
    >>> e_tot = mf.e_tot + np.sum(hcore*ddm1_corr) + np.sum(eris*dm2)

    2) Improved DMET energy (same as `emb.get_dmet_energy(version=2)`):
    >>> dm1 = emb.make_rdm1_demo(ao_basis=True)
    >>> dm2 = emb.make_rdm2_demo(ao_basis=True, approx_cumulant=True, dmet_dm2=False)
    >>> e_tot = mol.energy_nuc() + np.sum(hcore*dm1) + np.sum(eris*dm2)/2

    ...or in terms of the (approximated) cumulant:
    >>> fock = mf.get_fock()
    >>> ddm1 = emb.make_rdm1_demo(ao_basis=True, with_mf=False)
    >>> ddm2 = emb.make_rdm2_demo(ao_basis=True, with_dm1=False, approx_cumulant=True, dmet_dm2=False)
    >>> e_tot = mf.e_tot + np.sum(fock*ddm1) + np.sum(eris*ddm2)/2

    3) Improved DMET energy with true cumulant (same as `emb.get_dmet_energy(version=2, approx_cumulant=False)`):
    >>> dm1 = emb.make_rdm1_demo(ao_basis=True)
    >>> dm2 = emb.make_rdm2_demo(ao_basis=True, approx_cumulant=False)
    >>> e_tot = mol.energy_nuc() + np.sum(hcore*dm1) + np.sum(eris*dm2)/2

    ...or in terms of the cumulant:
    >>> ddm2 = emb.make_rdm2_demo(ao_basis=True, with_dm1=False, approx_cumulant=False)
    >>> fcorr = mf.get_fock(dm=dm1)
    >>> e_tot = mol.energy_nuc() + np.sum((hcore+fcorr)*dm1)/2 + np.sum(eris*ddm2)/2


    Parameters
    ----------
    ao_basis: bool, optional
        Return the density-matrix in the AO basis. Default: False.
    with_dm1: bool, optional
        If True, the non-cumulant part of the 2-DM will be added. See also `approx_cumulant`. Default: False.
    approx_cumulant: bool or int, optional
        If True, the cumulant of the 2-DM will be approximated and contain the non-cumulant contribution
        "delta[DM1(corr)-DM1(MF)]^2". Default: True.
    dmet_dm2: bool, optional
        If True, the mixed non-cumulant contributions, "DM1(MF) * [DM1(corr)-DM1(MF)]", will be projected
        symmetrically between both factors. This will return a 2-DM will evaluates to the DMET-energy
        of the literature. If False, only the second factor will be projected. This will generally
        give better expectation values and is the recommended setting. This value will only make
        a difference of `approx_cumulant` is True. Default: True.
    symmetrize: bool, optional
        Symmetrize the density-matrix at the end of the calculation. Default: True.

    Returns
    -------
    dm2: (n, n, n, n) array
        Two-particle reduced density matrix in AO (if `ao_basis=True`) or MO basis (default).
    """
    dm2 = np.zeros((emb.nmo, emb.nmo, emb.nmo, emb.nmo))

    # Loop over fragments to get cumulant contributions + non-cumulant contributions,
    # if (approx_cumulant and dmet_dm2):
    for x in emb.fragments:
        rx = x.get_overlap('mo|cluster')
        px = x.get_overlap('cluster|frag|cluster')

        if not approx_cumulant:
            dm1x = x.results.wf.make_rdm1()
            dm2x = x.results.wf.make_rdm2()
            dm2x -= (einsum('ij,kl->ijkl', dm1x, dm1x)
                   - einsum('ij,kl->iklj', dm1x, dm1x)/2)
        elif (int(approx_cumulant) == 1):
            dm2x = x.results.wf.make_rdm2(with_dm1=False, approx_cumulant=1)
            if (with_dm1 and dmet_dm2):
                dm1x = x.results.wf.make_rdm1(with_mf=False)
                dm1x = dot(rx, dm1x, rx.T)
                p = x.get_overlap('mo|frag|mo')
                pdm1x = np.dot(p, dm1x)

                # Projected DM1(HF)
                p = x.get_overlap('mo|mo[occ]|frag|mo')
                dm2 += 2*einsum('ij,kl->ijkl', p, dm1x)
                dm2 -= einsum('ij,kl->iklj', p, dm1x)
                # Replacing the above with this would lead to the new DMET energy:
                #for i in range(emb.nocc):
                #    dm2[i,i] += 2*pdm1x
                #    dm2[i,:,:,i] -= pdm1x

                # Projected DM1(CC)
                for i in range(emb.nocc):
                    dm2[:,:,i,i] += 2*pdm1x
                    dm2[:,i,i,:] -= pdm1x
        # Warning: This will give bad results [worse than E(DMET)]:
        elif (approx_cumulant == 2):
            dm2x = x.results.wf.make_rdm2()
            for i in range(x.cluster.nocc_active):
                for j in range(x.cluster.nocc_active):
                    dm2x[i,i,j,j] -= 4
                    dm2x[i,j,j,i] += 2
        else:
            raise ValueError('Invalid value for approx_cumulant: %r' % approx_cumulant)

        dm2 += einsum('xi,ijkl,px,qj,rk,sl->pqrs', px, dm2x, rx, rx, rx, rx)

    if with_dm1:
        if not approx_cumulant:
            dm1 = make_rdm1_demo_rhf(emb)
            dm2 += (einsum('ij,kl->ijkl', dm1, dm1)
                  - einsum('ij,kl->iklj', dm1, dm1)/2)
        elif (int(approx_cumulant) == 1 and not dmet_dm2):
            ddm1 = make_rdm1_demo_rhf(emb, with_mf=False)
            ddm1[np.diag_indices(emb.nocc)] += 1
            for i in range(emb.nocc):
                dm2[i,i,:,:] += 2*ddm1
                dm2[:,:,i,i] += 2*ddm1
                dm2[:,i,i,:] -= ddm1
                dm2[i,:,:,i] -= ddm1.T
        elif (approx_cumulant == 2):
            for i in range(emb.nocc):
                for j in range(emb.nocc):
                    dm2[i,i,j,j] += 4
                    dm2[i,j,j,i] -= 2

    if symmetrize:
        dm2 = (dm2 + dm2.transpose(1,0,3,2))/2
    if ao_basis:
        dm2 = einsum('ijkl,pi,qj,rk,sl->pqrs', dm2, *(4*[emb.mo_coeff]))
    return dm2
