"""Democratically partitioned RDMs"""

import numpy as np
from vayesta.core.util import dot, einsum, with_doc
from vayesta.mpi import mpi


def _get_fragments(emb):
    return emb.get_fragments(contributes=True)


def make_rdm1_demo_rhf(emb, ao_basis=False, with_mf=True, symmetrize=True, mpi_target=None):
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
    mpi_target: int or None, optional
        If set to an integer, the result will only be available at the specified MPI rank.
        If set to None, an MPI allreduce will be performed and the result will be available
        at all MPI ranks. Default: None.    

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
    for x in emb.get_fragments(contributes=True, mpi_rank=mpi.rank):
        emb.log.debugv("Now adding projected DM of fragment %s", x)
        dm1x = x.results.wf.make_rdm1(with_mf=False)
        rx = x.get_overlap("mo|cluster")
        px = x.get_overlap("cluster|frag|cluster")
        dm1 += einsum("xi,ij,px,qj->pq", px, dm1x, rx, rx)
    # --- MPI
    if mpi:
        dm1 = mpi.nreduce(dm1, target=mpi_target, logfunc=emb.log.timingv)
    if symmetrize:
        dm1 = (dm1 + dm1.T) / 2
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
    for x in _get_fragments(emb):
        emb.log.debugv("Now adding projected DM of fragment %s", x)
        dm1xa, dm1xb = x.results.wf.make_rdm1(with_mf=False)
        rxa, rxb = x.get_overlap("mo|cluster")
        pxa, pxb = x.get_overlap("cluster|frag|cluster")
        dm1a += einsum("xi,ij,px,qj->pq", pxa, dm1xa, rxa, rxa)
        dm1b += einsum("xi,ij,px,qj->pq", pxb, dm1xb, rxb, rxb)
    if symmetrize:
        dm1a = (dm1a + dm1a.T) / 2
        dm1b = (dm1b + dm1b.T) / 2
    if ao_basis:
        dm1a = dot(mo_coeff[0], dm1a, mo_coeff[0].T)
        dm1b = dot(mo_coeff[1], dm1b, mo_coeff[1].T)
    return (dm1a, dm1b)


# --- Two-particle
# ----------------


def make_rdm2_demo_rhf(
    emb, ao_basis=False, with_mf=True, with_dm1=True, part_cumulant=True, approx_cumulant=True, symmetrize=True
):
    """Make democratically partitioned two-particle reduced density-matrix from fragment calculations.

    Warning: A democratically partitioned DM is only expected to yield reasonable results
    for full fragmentations (eg. Lowdin-AO (SAO) or IAO+PAO fragmentation).

    Energies can be evaluated as follows from the 1-DM and 2-DM:

    1) Literature DMET energy:
    >>> e_nuc = mol.energy_nuc()
    >>> hcore = mf.get_hcore()
    >>> eris = pyscf.ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape([mol.nao]*4)
    >>> dm1 = emb.make_rdm1_demo(ao_basis=True)
    >>> dm2 = emb.make_rdm2_demo(ao_basis=True, part_cumulant=False, approx_cumulant=True)
    >>> e_tot = e_nuc + np.sum(hcore*dm1) + np.sum(eris*dm2)

    ...or in terms of the (approximated) cumulant:
    >>> vhf = mf.get_veff()
    >>> ddm1 = 2*dm1 - mf.make_rdm1()
    >>> ddm2 = emb.make_rdm2_demo(ao_basis=True, with_dm1=False, part_cumulant=False, approx_cumulant=True)
    >>> e_tot = e_nuc + np.sum(hcore*dm1) + np.sum(eris*ddm2) + np.sum(vhf*ddm1)/2

    2) Improved DMET energy (same as `emb.get_dmet_energy(part_cumulant=True)`):
    >>> dm1 = emb.make_rdm1_demo(ao_basis=True)
    >>> dm2 = emb.make_rdm2_demo(ao_basis=True, part_cumulant=True, approx_cumulant=True)
    >>> e_tot = e_nuc + np.sum(hcore*dm1) + np.sum(eris*dm2)/2

    ...or in terms of the (approximated) cumulant:
    >>> fock = mf.get_fock()
    >>> ddm1 = emb.make_rdm1_demo(ao_basis=True, with_mf=False)
    >>> ddm2 = emb.make_rdm2_demo(ao_basis=True, with_dm1=False, part_cumulant=True, approx_cumulant=True)
    >>> e_tot = mf.e_tot + np.sum(fock*ddm1) + np.sum(eris*ddm2)/2

    3) Improved DMET energy with true cumulant
    (same as `emb.get_dmet_energy(part_cumulant=True, approx_cumulant=False)`):
    >>> dm1 = emb.make_rdm1_demo(ao_basis=True)
    >>> dm2 = emb.make_rdm2_demo(ao_basis=True, part_cumulant=True, approx_cumulant=False)
    >>> e_tot = e_nuc + np.sum(hcore*dm1) + np.sum(eris*dm2)/2

    ...or in terms of the cumulant:
    >>> ddm2 = emb.make_rdm2_demo(ao_basis=True, with_dm1=False, part_cumulant=True, approx_cumulant=False)
    >>> fcorr = mf.get_fock(dm=dm1)
    >>> e_tot = e_nuc + np.sum((hcore+fcorr)*dm1)/2 + np.sum(eris*ddm2)/2


    Parameters
    ----------
    ao_basis: bool, optional
        Return the density-matrix in the AO basis. Default: False.
    with_dm1: bool, optional
        If True, the non-cumulant part of the 2-DM will be added. See also `approx_cumulant`. Default: False.
    part_cumulant: bool, optional
        If False, the mixed non-cumulant contributions, "DM1(MF) * [DM1(corr)-DM1(MF)]", will be projected
        symmetrically between both factors. This will return a 2-DM will evaluates to the DMET-energy
        of the literature. If True, only the second factor will be projected. This will generally
        give better expectation values and is the recommended setting. Default: True.
    approx_cumulant: bool or int, optional
        If True, the cumulant of the 2-DM will be approximated and contain the non-cumulant contribution
        "delta[DM1(corr)-DM1(MF)]^2". This value is ignored if part_cumulant is False and with_dm1 is True.
        Default: True.
    symmetrize: bool, optional
        Symmetrize the density-matrix at the end of the calculation. Default: True.

    Returns
    -------
    dm2: (n, n, n, n) array
        Two-particle reduced density matrix in AO (if `ao_basis=True`) or MO basis (default).
    """
    dm2 = np.zeros((emb.nmo, emb.nmo, emb.nmo, emb.nmo))

    # Loop over fragments to get cumulant contributions + non-cumulant contributions,
    # if (approx_cumulant and part_cumulant):
    for x in _get_fragments(emb):
        rx = x.get_overlap("mo|cluster")
        px = x.get_overlap("cluster|frag|cluster")

        # Partitioned cumulant:
        if part_cumulant or not with_dm1:
            if approx_cumulant:
                dm2x = x.results.wf.make_rdm2(with_dm1=False, approx_cumulant=True)
            else:
                # Form exact cluster cumulant:
                try:
                    dm2x = x.results.wf.make_rdm2(with_dm1=False, approx_cumulant=False)
                except NotImplementedError:
                    dm1x = x.results.wf.make_rdm1()
                    dm2x = x.results.wf.make_rdm2()
                    dm2x -= einsum("ij,kl->ijkl", dm1x, dm1x) - einsum("ij,kl->iklj", dm1x, dm1x) / 2
        # Partitioned 2-DM:
        else:
            dm2x = x.results.wf.make_rdm2(with_dm1=False, approx_cumulant=True)
            dm1x = x.results.wf.make_rdm1(with_mf=False)
            dm1x = dot(rx, dm1x, rx.T)
            # Add MF (1/2, since two mixed products are added):
            dm1x[np.diag_indices(emb.nocc)] += 1
            # The below is equivalent to:
            # p = x.get_overlap('mo|frag|mo')
            # ddm2 = np.zeros_like(dm2)
            # for i in range(emb.nocc):
            #     ddm2[i,i,:,:] += 2*dm1x
            #     ddm2[:,:,i,i] += 2*dm1x
            #     ddm2[:,i,i,:] -= dm1x
            #     ddm2[i,:,:,i] -= dm1x
            # dm2 += einsum('xi,ijkl->xjkl', p, ddm2)

            p = x.get_overlap("mo|frag|mo")
            pdm1x = np.dot(p, dm1x)
            # Projected DM1(HF)
            p = x.get_overlap("mo|mo[occ]|frag|mo")
            dm2 += 2 * einsum("ij,kl->ijkl", p, dm1x)
            dm2 -= einsum("ij,kl->iklj", p, dm1x)
            # Replacing the above with this would lead to the new DMET energy:
            # for i in range(emb.nocc):
            #     dm2[i,i] += 2*pdm1x
            #     dm2[i,:,:,i] -= pdm1x

            # Projected DM1(CC)
            for i in range(emb.nocc):
                dm2[:, :, i, i] += 2 * pdm1x
                dm2[:, i, i, :] -= pdm1x

        dm2 += einsum("xi,ijkl,px,qj,rk,sl->pqrs", px, dm2x, rx, rx, rx, rx)

    # Add non-cumulant contribution (unless added above, for part_cumulant=False)
    if with_dm1 and part_cumulant:
        if approx_cumulant:
            ddm1 = make_rdm1_demo_rhf(emb, with_mf=False)
            ddm1[np.diag_indices(emb.nocc)] += 1
            for i in range(emb.nocc):
                dm2[i, i, :, :] += 2 * ddm1
                dm2[:, :, i, i] += 2 * ddm1
                dm2[:, i, i, :] -= ddm1
                dm2[i, :, :, i] -= ddm1
        else:
            dm1 = make_rdm1_demo_rhf(emb)
            dm2 += einsum("ij,kl->ijkl", dm1, dm1) - einsum("ij,kl->iklj", dm1, dm1) / 2

    if symmetrize:
        dm2 = (dm2 + dm2.transpose(1, 0, 3, 2)) / 2
    if ao_basis:
        dm2 = einsum("ijkl,pi,qj,rk,sl->pqrs", dm2, *(4 * [emb.mo_coeff]))
    return dm2


@with_doc(make_rdm2_demo_rhf)
def make_rdm2_demo_uhf(
    emb, ao_basis=False, with_mf=True, with_dm1=True, part_cumulant=True, approx_cumulant=True, symmetrize=True
):
    na, nb = emb.nmo
    dm2aa = np.zeros((na, na, na, na))
    dm2ab = np.zeros((na, na, nb, nb))
    dm2bb = np.zeros((nb, nb, nb, nb))

    # Loop over fragments to get cumulant contributions + non-cumulant contributions,
    # if (approx_cumulant and part_cumulant):
    for x in _get_fragments(emb):
        rxa, rxb = x.get_overlap("mo|cluster")
        pxa, pxb = x.get_overlap("cluster|frag|cluster")

        # Partitioned cumulant:
        if part_cumulant or not with_dm1:
            if approx_cumulant:
                dm2xaa, dm2xab, dm2xbb = x.results.wf.make_rdm2(with_dm1=False, approx_cumulant=True)
            else:
                dm1xa, dm1xb = x.results.wf.make_rdm1()
                dm2xaa, dm2xab, dm2xbb = x.results.wf.make_rdm2()
                dm2xaa -= einsum("ij,kl->ijkl", dm1xa, dm1xa) - einsum("ij,kl->iklj", dm1xa, dm1xa)
                dm2xab -= einsum("ij,kl->ijkl", dm1xa, dm1xb)
                dm2xbb -= einsum("ij,kl->ijkl", dm1xb, dm1xb) - einsum("ij,kl->iklj", dm1xb, dm1xb)
        # Partitioned 2-DM:
        else:
            dm2xaa, dm2xab, dm2xbb = x.results.wf.make_rdm2(with_dm1=False, approx_cumulant=True)
            dm1xa, dm1xb = x.results.wf.make_rdm1(with_mf=False)
            dm1xa = dot(rxa, dm1xa, rxa.T)
            dm1xb = dot(rxb, dm1xb, rxb.T)
            # Add MF (1/2, since two mixed products are added):
            dm1xa[np.diag_indices(emb.nocc[0])] += 0.5
            dm1xb[np.diag_indices(emb.nocc[1])] += 0.5

            pa, pb = x.get_overlap("mo|frag|mo")
            ddm2aa = np.zeros_like(dm2aa)
            ddm2ab = np.zeros_like(dm2ab)
            ddm2bb = np.zeros_like(dm2bb)
            for i in range(emb.nocc[0]):
                ddm2aa[i, i, :, :] += dm1xa
                ddm2aa[:, :, i, i] += dm1xa
                ddm2aa[:, i, i, :] -= dm1xa
                ddm2aa[i, :, :, i] -= dm1xa
                ddm2ab[i, i, :, :] += dm1xb
            for i in range(emb.nocc[1]):
                ddm2bb[i, i, :, :] += dm1xb
                ddm2bb[:, :, i, i] += dm1xb
                ddm2bb[:, i, i, :] -= dm1xb
                ddm2bb[i, :, :, i] -= dm1xb
                ddm2ab[:, :, i, i] += dm1xa
            dm2aa += einsum("xi,ijkl->xjkl", pa, ddm2aa)
            dm2bb += einsum("xi,ijkl->xjkl", pb, ddm2bb)
            dm2ab += (einsum("xi,ijkl->xjkl", pa, ddm2ab) + einsum("xk,ijkl->ijxl", pb, ddm2ab)) / 2

        dm2aa += einsum("xi,ijkl,px,qj,rk,sl->pqrs", pxa, dm2xaa, rxa, rxa, rxa, rxa)
        dm2bb += einsum("xi,ijkl,px,qj,rk,sl->pqrs", pxb, dm2xbb, rxb, rxb, rxb, rxb)
        dm2ab += (
            einsum("xi,ijkl,px,qj,rk,sl->pqrs", pxa, dm2xab, rxa, rxa, rxb, rxb)
            + einsum("xk,ijkl,pi,qj,rx,sl->pqrs", pxb, dm2xab, rxa, rxa, rxb, rxb)
        ) / 2

    if with_dm1 and part_cumulant:
        if approx_cumulant:
            ddm1a, ddm1b = make_rdm1_demo_uhf(emb, with_mf=False)
            ddm1a[np.diag_indices(emb.nocc[0])] += 0.5
            ddm1b[np.diag_indices(emb.nocc[1])] += 0.5
            for i in range(emb.nocc[0]):
                dm2aa[i, i, :, :] += ddm1a
                dm2aa[:, :, i, i] += ddm1a
                dm2aa[:, i, i, :] -= ddm1a
                dm2aa[i, :, :, i] -= ddm1a
                dm2ab[i, i, :, :] += ddm1b
            for i in range(emb.nocc[1]):
                dm2bb[i, i, :, :] += ddm1b
                dm2bb[:, :, i, i] += ddm1b
                dm2bb[:, i, i, :] -= ddm1b
                dm2bb[i, :, :, i] -= ddm1b
                dm2ab[:, :, i, i] += ddm1a
        else:
            dm1a, dm1b = make_rdm1_demo_uhf(emb)
            dm2aa += einsum("ij,kl->ijkl", dm1a, dm1a) - einsum("ij,kl->iklj", dm1a, dm1a)
            dm2bb += einsum("ij,kl->ijkl", dm1b, dm1b) - einsum("ij,kl->iklj", dm1b, dm1b)
            dm2ab += einsum("ij,kl->ijkl", dm1a, dm1b)

    if symmetrize:
        dm2aa = (dm2aa + dm2aa.transpose(1, 0, 3, 2)) / 2
        dm2bb = (dm2bb + dm2bb.transpose(1, 0, 3, 2)) / 2
    if ao_basis:
        dm2aa = einsum("ijkl,pi,qj,rk,sl->pqrs", dm2aa, *(4 * [emb.mo_coeff[0]]))
        dm2bb = einsum("ijkl,pi,qj,rk,sl->pqrs", dm2bb, *(4 * [emb.mo_coeff[1]]))
        dm2ab = einsum("ijkl,pi,qj,rk,sl->pqrs", dm2ab, *(2 * [emb.mo_coeff[0]] + 2 * [emb.mo_coeff[1]]))
    return (dm2aa, dm2ab, dm2bb)
