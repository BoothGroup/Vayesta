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


def get_frag_deltaW(mf, fragment, log=None):
    """Generates change in coulomb interaction due to screening at the level of cRPA.
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

    is_rhf = np.ndim(mf.mo_coeff[1]) == 1
    if not hasattr(mf, "with_df"):
        raise NotImplementedError("Screened interactions require density-fitting.")

    crpa = get_crpa(mf, fragment)
    # Now need to calculate interactions.
    nmo = mf.mo_coeff.shape[1]
    nocc = sum(mf.mo_occ.T > 0)
    c = mf.mo_coeff
    if is_rhf:
        nocc = (nocc, nocc)
        c = (c, c)
    # First calculate alpha contribution.
    lov_a = ao2mo(mf, mo_coeff=c[0], ijslice=(0, nocc[0], nocc[0], nmo[0])).reshape((-1, crpa.ova))
    lov_a = dot(lov_a, crpa.ov_rot[0].T)

    l_aux = dot(lov_a, crpa.XpY_ss[0])
    del lov_a
    lov_b = ao2mo(mf, mo_coeff=c[1], ijslice=(0, nocc[1], nocc[1], nmo[1])).reshape((-1, crpa.ovb))
    lov_b = dot(lov_b, crpa.ov_rot[1].T)

    # This is a decomposition of the cRPA reducible dd response in the auxilliary basis
    l_aux += dot(lov_b, crpa.XpY_ss[1])
    del lov_b

    # This is expensive, and we'd usually want to avoid doing it twice unnecessarily, but other things will be worse.
    # Now calculate the coupling back to the fragment itself.
    c_act = fragment.cluster.c_active
    if is_rhf:
        c_act = (c_act, c_act)

    lpqa_loc = ao2mo(mf, mo_coeff=c_act[0])
    lpqb_loc = ao2mo(mf, mo_coeff=c_act[1])

    # Calculating this quantity scales as O(N^3), rather than O(N^4) if we explicitly calculated the cRPA dd response
    # in the auxiliary basis.
    l_a = einsum("npq,nm->mpq", lpqa_loc, l_aux)
    del lov_a
    l_b = einsum("npq,nm->mpq", lpqb_loc, l_aux)
    del lov_b

    static_fac = crpa.freqs_ss ** (-1)

    delta_w = (
        einsum("npq,n,nrs->pqrs", l_a, static_fac, l_a),
        einsum("npq,n,nrs->pqrs", l_a, static_fac, l_b),
        einsum("npq,n,nrs->pqrs", l_b, static_fac, l_b),
    )
    return delta_w


def get_crpa(orig_mf, f):

    def construct_crpa_rot(f):
        """Constructs the rotation of the overall mean-field space into which """
        ro = f.get_overlap("cluster[occ]|mo[occ]")
        rv = f.get_overlap("cluster[vir]|mo[vir]")

        if isinstance(ro, np.ndarray):
            ro = (ro, ro)
        if isinstance(rv, np.ndarray):
            rv = (rv, rv)

        rot_ova = einsum("Ij,Ab->IAjb", ro[0], rv[0])
        rot_ova = rot_ova.reshape((rot_ova.shape[0] * rot_ova.shape[1], -1))

        rot_ovb = einsum("Ij,Ab->IAjb", ro[1], rv[1])
        rot_ovb = rot_ovb.reshape((rot_ovb.shape[0] * rot_ovb.shape[1], -1))
        return scipy.linalg.null_space(rot_ova).T, scipy.linalg.null_space(rot_ovb).T

    rot_ov = construct_crpa_rot(f)
    # RPA calculation and new_mf will contain all required information for the response.
    crpa = ssRPA(orig_mf, ov_rot=rot_ov)
    crpa.kernel()

    return crpa


def ao2mo(mf, mo_coeff=None, ijslice=None):
    """Get MO basis density-fitted integrals.
    """

    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    nmo = mo_coeff.shape[1]
    naux = mf.with_df.get_naoaux()
    mem_incore = (2 * nmo ** 2 * naux) * 8 / 1e6
    mem_now = lib.current_memory()[0]

    mo = np.asarray(mo_coeff, order='F')
    if ijslice is None:
        ijslice = (0, nmo, 0, nmo)

    finshape = (naux, ijslice[1] - ijslice[0], ijslice[3] - ijslice[2])

    Lpq = None
    if (mem_incore + mem_now < 0.99 * mf.max_memory) or mf.mol.incore_anyway:
        Lpq = _ao2mo.nr_e2(mf.with_df._cderi, mo, ijslice, aosym='s2', out=Lpq)
        return Lpq.reshape(*finshape)
    else:
        logger.warn(mf, 'Memory may not be enough!')
        raise NotImplementedError
