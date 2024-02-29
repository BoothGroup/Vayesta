import scipy.linalg

from vayesta.rpa import ssRPA
from .screening_moment import _get_target_rot
import copy
from vayesta.core.util import *
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import __config__


class cRPAError(RuntimeError):
    pass


def get_frag_W(mf, fragment, pcoupling=True, only_ov_screened=False, log=None):
    """Generates screened coulomb interaction due to screening at the level of cRPA.
    Note that this currently scales as O(N_frag N^6), so is not practical without further refinement.

    Parameters
    ----------
    mf : pyscf.scf object
        Mean-field instance.
    fragment : vayesta.qemb.Fragment subclass
        Fragments for the calculation, used to define local interaction space.
    log : logging.Logger, optional
        Logger object. If None, the logger of the `emb` object is used. Default: None.

    Returns
    -------
    freqs : np.array
        Effective bosonic frequencies for cRPA screening.
    couplings : np.array
        Effective bosonic couplings for cRPA screening.
    """

    log.info("Generating screened interaction via frequency dependent cRPA.")
    try:
        l_a, l_b, crpa = set_up_W_crpa(mf, fragment, pcoupling, only_ov_screened, log=log)
    except cRPAError as e:
        freqs = np.zeros((0,))
        nmo = mf.mo_coeff.shape[1]
        couplings = (np.zeros((0, nmo, nmo)), np.zeros((0, nmo, nmo)))
    else:
        freqs = crpa.freqs_ss
        couplings = (l_a, l_b)
    log.info("cRPA resulted in %d poles", len(freqs))

    return freqs, couplings


def get_frag_deltaW(mf, fragment, pcoupling=True, only_ov_screened=False, log=None):
    """Generates change in coulomb interaction due to screening at the level of static limit of cRPA.
    Note that this currently scales as O(N_frag N^6), so is not practical without further refinement.

    Parameters
    ----------
    mf : pyscf.scf object
        Mean-field instance.
    fragment : vayesta.qemb.Fragment subclass
        Fragments for the calculation, used to define local interaction space.
    log : logging.Logger, optional
        Logger object. If None, the logger of the `emb` object is used. Default: None.

    Returns
    -------
    deltaW : np.array
        Change in cluster local coulomb interaction due to cRPA screening.
    """

    log.info("Generating screened interaction via static limit of cRPA.")
    log.warning("This is poorly defined for non-CAS fragmentations.")
    log.warning("This implementation is expensive, with O(N^6) computational cost per cluster.")
    try:
        l_a, l_b, crpa = set_up_W_crpa(mf, fragment, pcoupling, only_ov_screened=only_ov_screened, log=log)
    except cRPAError as e:
        nmo = mf.mo_coeff.shape[1]
        delta_w = tuple([np.zeros([nmo] * 4)] * 4)
        crpa = None
    else:
        # Have a factor of -2 due to negative value of RPA dd response, and summation of
        # the excitation and deexcitation branches of the dd response.
        static_fac = -1.0 * (crpa.freqs_ss ** (-1))

        delta_w = (
            einsum("npq,n,nrs->pqrs", l_a, static_fac, l_a) + einsum("nqp,n,nsr->pqrs", l_a, static_fac, l_a),
            einsum("npq,n,nrs->pqrs", l_a, static_fac, l_b) + einsum("nqp,n,nsr->pqrs", l_a, static_fac, l_b),
            einsum("npq,n,nrs->pqrs", l_b, static_fac, l_b) + einsum("nqp,n,nsr->pqrs", l_b, static_fac, l_b),
        )
    return delta_w, crpa


def set_up_W_crpa(mf, fragment, pcoupling=True, only_ov_screened=False, log=None):
    is_rhf = np.ndim(mf.mo_coeff[1]) == 1
    if not hasattr(mf, "with_df"):
        raise NotImplementedError("Screened interactions require density-fitting.")
    crpa, rot_loc, rot_crpa = get_crpa(mf, fragment, log)
    # Now need to calculate interactions.
    nmo = mf.mo_coeff.shape[1]
    nocc = sum(mf.mo_occ.T > 0)
    c = mf.mo_coeff
    if is_rhf:
        nocc = (nocc, nocc)
        c = (c, c)

    # First calculate alpha contribution.
    lov_a = ao2mo(mf, mo_coeff=c[0], ijslice=(0, nocc[0], nocc[0], nmo)).reshape((-1, crpa.ova))
    lov_a = dot(lov_a, crpa.ov_rot[0].T)
    l_aux = dot(lov_a, crpa.XpY_ss[0])
    del lov_a
    lov_b = ao2mo(mf, mo_coeff=c[1], ijslice=(0, nocc[1], nocc[1], nmo)).reshape((-1, crpa.ovb))
    lov_b = dot(lov_b, crpa.ov_rot[1].T)

    # This is a decomposition of the cRPA reducible dd response in the auxilliary basis
    l_aux += dot(lov_b, crpa.XpY_ss[1])
    del lov_b

    # This is expensive, and we'd usually want to avoid doing it twice unnecessarily, but other things will be worse.
    # Now calculate the coupling back to the fragment itself.
    c_act = fragment.cluster.c_active
    if is_rhf:
        c_act = (c_act, c_act)

    # Calculating this quantity scales as O(N^3), rather than O(N^4) if we explicitly calculated the cRPA dd response
    # in the auxiliary basis.
    lpqa_loc = ao2mo(mf, mo_coeff=c_act[0])
    l_a = einsum("npq,nm->mpq", lpqa_loc, l_aux)
    del lpqa_loc
    lpqb_loc = ao2mo(mf, mo_coeff=c_act[1])
    l_b = einsum("npq,nm->mpq", lpqb_loc, l_aux)
    del lpqb_loc

    if pcoupling:
        # Need to calculate additional contribution to the couplings resulting from rotation of the irreducible
        # polarisability.
        # Generate the full-system matrix of orbital energy differences.
        eps = ssRPA(mf)._gen_eps()
        # First, generate epsilon couplings between cluster and crpa spaces.
        eps_fb = [einsum("p,qp,rp->qr", e, l, nl) for e, l, nl in zip(eps, rot_loc, crpa.ov_rot)]
        # Then generate X and Y values for this correction.
        x_crpa = [(p + m) / 2 for p, m in zip(crpa.XpY_ss, crpa.XmY_ss)]
        y_crpa = [(p - m) / 2 for p, m in zip(crpa.XpY_ss, crpa.XmY_ss)]
        # Contract with epsilon values
        a_fb = [dot(e, x) for x, e in zip(x_crpa, eps_fb)]
        b_fb = [dot(e, y) for y, e in zip(y_crpa, eps_fb)]
        no = fragment.cluster.nocc_active
        if isinstance(no, int):
            no = (no, no)
        nv = fragment.cluster.nvir_active
        if isinstance(nv, int):
            nv = (nv, nv)
        l_a[:, : no[0], no[0] :] += a_fb[0].T.reshape((a_fb[0].shape[-1], no[0], nv[0]))
        l_b[:, : no[1], no[1] :] += a_fb[1].T.reshape((a_fb[1].shape[-1], no[1], nv[1]))

        l_a[:, no[0] :, : no[0]] += b_fb[0].T.reshape((b_fb[0].shape[-1], no[0], nv[0])).transpose(0, 2, 1)
        l_b[:, no[1] :, : no[1]] += b_fb[1].T.reshape((b_fb[1].shape[-1], no[1], nv[1])).transpose(0, 2, 1)

    if only_ov_screened:
        # Zero out all contributions screening oo or vv contributions.
        no = fragment.cluster.nocc_active
        if isinstance(no, int):
            no = (no, no)
        l_a[:, no[0] :, no[0] :] = 0.0
        l_a[:, : no[0], : no[0]] = 0.0
        l_b[:, no[1] :, no[1] :] = 0.0
        l_b[:, : no[1], : no[1]] = 0.0
    return l_a, l_b, crpa


def get_crpa(orig_mf, f, log):
    def construct_loc_rot(f):
        """Constructs the rotation of the overall mean-field space into which"""
        ro = f.get_overlap("cluster[occ]|mo[occ]")
        rv = f.get_overlap("cluster[vir]|mo[vir]")

        if isinstance(ro, np.ndarray):
            ro = (ro, ro)
        if isinstance(rv, np.ndarray):
            rv = (rv, rv)

        rot_ova = einsum("Ij,Ab->IAjb", ro[0], rv[0])
        if rot_ova.size == 0:
            rot_ova = np.empty((0, ro[0].shape[1]*rv[0].shape[1]))
        else:
            rot_ova = rot_ova.reshape((rot_ova.shape[0] * rot_ova.shape[1], -1))

        rot_ovb = einsum("Ij,Ab->IAjb", ro[1], rv[1])
        if rot_ovb.size == 0:
            rot_ovb = np.empty((0, ro[1].shape[1]*rv[1].shape[1]))
        else:
            rot_ovb = rot_ovb.reshape((rot_ovb.shape[0] * rot_ovb.shape[1], -1))

        return rot_ova, rot_ovb

    rot_loc = construct_loc_rot(f)
    if rot_loc[0].size > 0:
        rot_ov_a = scipy.linalg.null_space(rot_loc[0]).T
    else:
        rot_ov_a = np.eye(rot_loc[0].shape[1])
    if rot_loc[1].size > 0:
        rot_ov_b = scipy.linalg.null_space(rot_loc[1]).T
    else:
        rot_ov_b = np.eye(rot_loc[1].shape[1])
    rot_ov = (rot_ov_a, rot_ov_b)
    if rot_ov[0].shape[0] == 0 and rot_ov[1].shape[0] == 0:
        log.warning("cRPA space contains no excitations! Interactions will be unscreened.")
        raise cRPAError("cRPA space contains no excitations!")
    # RPA calculation and new_mf will contain all required information for the response.
    crpa = ssRPA(orig_mf, ov_rot=rot_ov)
    crpa.kernel()
    return crpa, rot_loc, rot_ov


def ao2mo(mf, mo_coeff=None, ijslice=None):
    """Get MO basis density-fitted integrals."""

    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    nmo = mo_coeff.shape[1]
    naux = mf.with_df.get_naoaux()
    mem_incore = (2 * nmo**2 * naux) * 8 / 1e6
    mem_now = lib.current_memory()[0]

    mo = np.asarray(mo_coeff, order="F")
    if ijslice is None:
        ijslice = (0, nmo, 0, nmo)

    finshape = (naux, ijslice[1] - ijslice[0], ijslice[3] - ijslice[2])

    Lpq = None
    if (mem_incore + mem_now < 0.99 * mf.max_memory) or mf.mol.incore_anyway:
        Lpq = _ao2mo.nr_e2(mf.with_df._cderi, mo, ijslice, aosym="s2", out=Lpq)
        return Lpq.reshape(*finshape)
    else:
        logger.warn(mf, "Memory may not be enough!")
        raise NotImplementedError
