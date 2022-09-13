from vayesta.rpa import ssRPA
from .screening_moment import _get_target_rot
import copy
from vayesta.core.util import *
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import __config__


def get_crpa_deltaW(emb, fragments=None, calc_delta_e=True, log=None):
    """Generates change in coulomb interaction due to screening at the level of cRPA.
    Note that this currently scales as O(N_frag N^6), so is not practical without further refinement.

    Parameters
    ----------
    emb : Embedding
        Embedding instance.
    fragments : list of vayesta.qemb.Fragment subclasses, optional
        List of fragments for the calculation, used to define local interaction spaces.
        If None, `emb.get_fragments(sym_parent=None)` is used. Default: None.
    calc_delta_e : bool, optional.
        Whether to calculate a nonlocal energy correction at the level of RPA
    log : logging.Logger, optional
        Logger object. If None, the logger of the `emb` object is used. Default: None.

    Returns
    -------
    seris_ov : list of tuples of np.array
        List of spin-dependent screened (ov|ov), for each fragment provided.
    delta_e: float
        Delta RPA correction computed as difference between full system RPA energy and
        cluster correlation energies; currently only functional in CAS fragmentations.
    """
    if fragments is None:
        fragments = emb.get_fragments(sym_parent=None)
    if emb.spinsym != 'unrestricted':
        raise NotImplementedError("Screened interactions require a spin-unrestricted formalism.")
    if emb.df is None:
        raise NotImplementedError("Screened interactions require density-fitting.")

    wc = []

    for f in fragments:
        # Compute cRPA response.
        new_mf, crpa = get_crpa_chi(emb.mf, f)

        # Apply static approximation to interaction.
        static_factor = crpa.freqs_ss ** (-1)
        # Now need to calculate interactions.

        nmo = new_mf.mo_coeff.shape[1]
        nocc = sum(new_mf.mo_occ.T > 0)

        Lov_aenv = ao2mo(new_mf, mo_coeff=new_mf.mo_coeff[0], ijslice=(0, nocc[0], nocc[0], nmo[0])).reshape(
            (-1, crpa.ova))
        Lov_benv = ao2mo(new_mf, mo_coeff=new_mf.mo_coeff[1], ijslice=(0, nocc[1], nocc[1], nmo[1])).reshape(
            (-1, crpa.ovb))

        # This is the coefficient of the cRPA reducible dd response in the auxilliary basis
        L_aux = dot(Lov_aenv, crpa.XpY_ss[0]) + dot(Lov_benv, crpa.XpY_ss[1])
        del Lov_aenv, Lov_benv
        # This is the static approximation for the screened coulomb interaction in the auxiliary basis.
        chi_aux = einsum("nx,x,mx->nm", L_aux, crpa.freqs_ss ** (-1), L_aux)
        # This is expensive, and we'd usually want to avoid doing it twice unnecessarily, but other things will be worse.
        # Now calculate the coupling back to the fragment itself.
        Lpqa_loc = ao2mo(new_mf, mo_coeff=f.cluster.c_active[0])
        Lpqb_loc = ao2mo(new_mf, mo_coeff=f.cluster.c_active[1])

        deltaW = (
            einsum("npq,nm,mrs->pqrs", Lpqa_loc, chi_aux, Lpqa_loc),
            einsum("npq,nm,mrs->pqrs", Lpqa_loc, chi_aux, Lpqb_loc),
            einsum("npq,nm,mrs->pqrs", Lpqb_loc, chi_aux, Lpqb_loc),
        )
        wc += [deltaW]
    return wc


def get_frag_deltaW(mf, fragment, log=None):
    new_mf, crpa = get_crpa_chi(mf, fragment)

    # Apply static approximation to interaction.
    static_factor = crpa.freqs_ss ** (-1)
    # Now need to calculate interactions.

    nmo = new_mf.mo_coeff.shape[1]
    nocc = sum(new_mf.mo_occ.T > 0)

    Lov_aenv = ao2mo(new_mf, mo_coeff=new_mf.mo_coeff[0], ijslice=(0, nocc[0], nocc[0], nmo[0])).reshape((-1, crpa.ova))
    Lov_benv = ao2mo(new_mf, mo_coeff=new_mf.mo_coeff[1], ijslice=(0, nocc[1], nocc[1], nmo[1])).reshape((-1, crpa.ovb))

    # This is the coefficient of the cRPA reducible dd response in the auxilliary basis
    L_aux = dot(Lov_aenv, crpa.XpY_ss[0]) + dot(Lov_benv, crpa.XpY_ss[1])
    del Lov_aenv, Lov_benv
    # This is the static approximation for the screened coulomb interaction in the auxiliary basis.
    chi_aux = einsum("nx,x,mx->nm", L_aux, crpa.freqs_ss ** (-1), L_aux)
    # This is expensive, and we'd usually want to avoid doing it twice unnecessarily, but other things will be worse.
    # Now calculate the coupling back to the fragment itself.
    Lpqa_loc = ao2mo(new_mf, mo_coeff=fragment.cluster.c_active[0])
    Lpqb_loc = ao2mo(new_mf, mo_coeff=fragment.cluster.c_active[1])

    deltaW = (
        einsum("npq,nm,mrs->pqrs", Lpqa_loc, chi_aux, Lpqa_loc),
        einsum("npq,nm,mrs->pqrs", Lpqa_loc, chi_aux, Lpqb_loc),
        einsum("npq,nm,mrs->pqrs", Lpqb_loc, chi_aux, Lpqb_loc),
    )
    return deltaW


def get_crpa_chi(orig_mf, f):
    def construct_crpa_mf(orig_mf, f):
        """Construct mean-field object upon which an RPA calculation returns the cRPA response for a cluster."""
        s = orig_mf.get_ovlp()

        def get_canon_env(c, orig_e):
            # First define rotation of initial orbitals.
            r = dot(orig_mf.mo_coeff.T, s, c)
            # Then get environmental orbital energies
            eenv = einsum("n,np,nq->pq", orig_e, r, r)
            # Diagonalise to get canonicalised equivalent.
            mo_energy, c_eig = np.linalg.eigh(eenv)
            return mo_energy, dot(c, c_eig)

        # Do seperately to ensure no changing of mean-field.
        if np.ndim(orig_mf.mo_coeff[0]) == 1:
            eo, co = get_canon_env(f.cluster.c_frozen_occ, f.mo_energy)
            ev, cv = get_canon_env(f.cluster.c_frozen_vir, f.mo_energy)
            e = np.concatenate((eo, ev))
            c = np.concatenate((co, cv), axis=1)
            occ = np.zeros_like(e)
            occ[:f.cluster.nocc_frozen] = 2.0
        else:
            eoa, coa = get_canon_env(f.cluster.c_frozen_occ[0], f.mo_energy[0])
            eob, cob = get_canon_env(f.cluster.c_frozen_occ[1], f.mo_energy[1])
            eva, cva = get_canon_env(f.cluster.c_frozen_vir[0], f.mo_energy[0])
            evb, cvb = get_canon_env(f.cluster.c_frozen_vir[1], f.mo_energy[1])

            ea = np.concatenate([eoa, eva])
            eb = np.concatenate([eob, evb])
            e = np.array((ea, eb))

            ca = np.concatenate([coa, cva], axis=1)
            cb = np.concatenate([cob, cvb], axis=1)
            c = np.array((ca, cb))
            occ = np.zeros_like(e)
            occ[0, :f.cluster.nocc_frozen[0]] = 1.0
            occ[1, :f.cluster.nocc_frozen[1]] = 1.0

        new_mf = copy.copy(orig_mf)
        new_mf.mo_coeff = c
        new_mf.mo_energy = e
        new_mf.mo_occ = occ
        return new_mf

    new_mf = construct_crpa_mf(orig_mf, f)
    # RPA calculation and new_mf will contain all required information for the response.
    crpa = ssRPA(new_mf)
    crpa.kernel()

    return new_mf, crpa


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
