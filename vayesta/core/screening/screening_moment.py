import logging
import numpy as np
import scipy
import scipy.linalg
from vayesta.rpa import ssRIRPA, ssRPA
from vayesta.core.util import dot, einsum
from vayesta.mpi import mpi


def build_screened_eris(emb, fragments=None, cderi_ov=None, store_m0=True, npoints=48, log=None):
    """Generates renormalised coulomb interactions for use in local cluster calculations.
    Currently requires unrestricted system.

    Parameters
    ----------
    emb : Embedding
        Embedding instance.
    fragments : list of vayesta.qemb.Fragment subclasses, optional
        List of fragments for the calculation, used to define local interaction spaces.
        If None, `emb.get_fragments(sym_parent=None)` is used. Default: None.
    cderi_ov : np.array or tuple of np.array, optional.
        Cholesky-decomposed ERIs in the particle-hole basis of mf. If mf is unrestricted
        this should be a list of arrays corresponding to the different spin channels.
    store_m0 : bool, optional.
        Whether to store the local zeroth moment in the fragment class for use later.
    npoints : int, optional
        Number of points for numerical integration. Default: 48.
    log : logging.Logger, optional
        Logger object. If None, the logger of the `emb` object is used. Default: None.

    Returns
    -------
    seris_ov : list of tuples of np.array
        List of spin-dependent screened (ov|ov), for each fragment provided.
    erpa: float
        Delta RPA correction computed as difference between full system RPA energy and
        cluster correlation energies; currently only functional in CAS fragmentations.
    """
    if log is None:
        log = log or emb.log or logging.getLogger(__name__)
    log.info("Calculating screened Coulomb interactions")
    log.info("-----------------------------------------")

    # --- Setup
    if fragments is None:
        fragments = emb.get_fragments(active=True, sym_parent=None, mpi_rank=mpi.rank)
    fragments = [f for f in fragments if f.opts.screening == 'mrpa']
    if emb.df is None:
        raise NotImplementedError("Screened interactions require density-fitting.")
    r_occs = [f.get_overlap('mo[occ]|cluster[occ]') for f in fragments]
    r_virs = [f.get_overlap('mo[vir]|cluster[vir]') for f in fragments]
    target_rots, ovs_active = _get_target_rot(r_occs, r_virs)

    local_moments = calc_moms_RIRPA(emb.mf, target_rots, ovs_active, log, cderi_ov, npoints)
    # Could generate moments using N^6 moments instead, but just for debugging.
    #local_moments, erpa = calc_moms_RPA(emb.mf, target_rots, ovs_active, log, cderi_ov, calc_e, npoints)

    # Then construct the RPA coupling matrix A-B, given by the diagonal matrix of energy differences.
    no = np.array(sum(emb.mf.mo_occ.T > 0))
    if no.size == 1: no = np.array([int(no), int(no)])
    norb = emb.mo_coeff[0].shape[0]
    nv = norb - no

    mo_e = emb.mf.mo_energy

    if isinstance(mo_e[0], float):
        mo_e = np.array([mo_e, mo_e])

    def get_eps_singlespin(no_, nv_, mo_energy):
        eps = np.zeros((no_, nv_))
        eps = eps + mo_energy[no_:]
        eps = (eps.T - mo_energy[:no_]).T
        eps = eps.reshape(-1)
        return eps
    eps = np.concatenate([get_eps_singlespin(no[0], nv[0], mo_e[0]),
                          get_eps_singlespin(no[1], nv[1], mo_e[1])])

    # And use this to perform inversion to calculate interaction in cluster.
    seris_ov = []
    for i, (f, rot, mom, (ova, ovb)) in enumerate(zip(fragments, target_rots, local_moments, ovs_active)):
        amb = einsum("pn,qn,n->pq", rot, rot, eps)  # O(N^2 N_clus^4)
        # Everything from here on is independent of system size, scaling at most as O(N_clus^6)
        # (arrays have side length equal to number of cluster single-particle excitations).
        e, c = np.linalg.eigh(mom)
        if min(e) < 1e-4:
            log.warning("Small eigenvalue of local rpa moment in %s: %e", f.name, min(e))

        mominv = einsum("pn,n,qn->pq", c, e**(-1), c)
        apb = dot(mominv, amb, mominv)

        # This is the renormalised coulomb kernel in the cluster.
        # Note that this is only defined in the particle-hole space, but has the same 8-fold symmetry
        # as the (assumed real) coulomb interaction.
        kc = 0.5 * (apb - amb)
        # Now need to strip out spin components of the renormalised interaction, and change to 4 orbital
        # indices.
        no = f.cluster.nocc_active
        nv = f.cluster.nvir_active
        if isinstance(no, int):
            no = (no, no)
            nv = (nv, nv)

        kcaa = kc[:ova, :ova].reshape((no[0], nv[0], no[0], nv[0]))
        kcab = kc[:ova, ova:].reshape((no[0], nv[0], no[1], nv[1]))
        kcbb = kc[ova:, ova:].reshape((no[1], nv[1], no[1], nv[1]))

        kc = (kcaa, kcab, kcbb)
        f._seris_ov = (kc, mom, amb) if store_m0 else (kc,)
        seris_ov.append(kc)


def calc_moms_RIRPA(mf, target_rots, ovs_active, log, cderi_ov, npoints):
    rpa = ssRIRPA(mf, log=log, lov=cderi_ov)
    tr = np.concatenate(target_rots, axis=0)
    if sum(sum(ovs_active)) > 0:
        # Computation scales as O(N^4)
        moms_interact, est_errors = rpa.kernel_moms(0, tr, npoints=npoints)
        momzero_interact = moms_interact[0]
    else:
        momzero_interact = np.zeros_like(np.concatenate(tr, axis=0))

    # Now need to separate into local contributions
    n = 0
    local_moments = []
    for nov, rot in zip(ovs_active, target_rots):
        # Computation costs O(N^2 N_clus^2)
        # Get corresponding section of overall moment, then project to just local contribution.
        mom = dot(momzero_interact[n:n+sum(nov)], rot.T)
        # This isn't exactly symmetric due to numerical integration, so enforce here.
        mom = (mom + mom.T) / 2
        local_moments += [mom]
        n += sum(nov)

    return local_moments

def calc_moms_RPA(mf, target_rots, ovs_active, log, cderi_ov, calc_e, npoints):
    rpa = ssRPA(mf, log=log)
    erpa = rpa.kernel()
    mom0 = rpa.gen_moms(0)[0]
    local_moments = []
    for rot in target_rots:
        local_moments += [dot(rot, mom0, rot.T)]
    return local_moments, erpa

def get_screened_eris_full(eris, seris_ov, copy=True, log=None):
    """Build full array of screened ERIs, given the bare ERIs and screening."""

    log.info("Generating screened interaction to conserve zeroth moment of the dd response.")

    def replace_ov(full, ov, spins):
        out = full.copy() if copy else full
        no1, no2 = ov.shape[0], ov.shape[2]
        o1, v1 = np.s_[:no1], np.s_[no1:]
        o2, v2 = np.s_[:no2], np.s_[no2:]
        out[o1,v1,o2,v2] = ov
        out[v1,o1,o2,v2] = ov.transpose([1, 0, 2, 3])
        out[o1,v1,v2,o2] = ov.transpose([0, 1, 3, 2])
        out[v1,o1,v2,o2] = ov.transpose([1, 0, 3, 2])
        return out

    if isinstance(eris, np.ndarray):
        eris = (eris, eris, eris)

    seris = (replace_ov(eris[0], seris_ov[0], 'aa'),
             replace_ov(eris[1], seris_ov[1], 'ab'),
             replace_ov(eris[2], seris_ov[2], 'bb'))
    return seris

def get_screened_eris_ccsd(eris, seris_ov, add_restore_bare=True, log=None):

    if add_restore_bare:
        gaa = eris.ovov[:]
        gab = eris.ovOV[:]
        gbb = eris.OVOV[:]

    saa, sab, sbb = seris_ov
    # Alpha-alpha
    eris.ovov = saa
    eris.ovvo = saa.transpose([0,1,3,2])
    # Alpha-beta
    eris.ovOV = sab
    eris.ovVO = sab.transpose([0,1,3,2])
    # Beta-beta
    eris.OVOV = sbb
    eris.OVVO = sbb.transpose([0,1,3,2])
    # Beta-alpha
    eris.OVvo = sab.transpose([2,3,1,0])

    # Add restore_bare function to remove screening later on
    if add_restore_bare:
        def get_bare(eris):
            return (gaa, gab, gbb)
        def restore_bare(eris):
            eris = get_screened_eris_ccsd(eris, eris.get_bare(), add_restore_bare=False)
            del eris.get_bare, eris.restore_bare
            return eris

        eris.get_bare = get_bare.__get__(eris)
        eris.restore_bare = restore_bare.__get__(eris)

    return eris

def _get_target_rot(r_active_occs, r_active_virs):
    """Given the definitions of our cluster spaces in terms of rotations of the occupied and virtual
    orbitals, define the equivalent rotation of the full-system particle-hole excitation space.

    Parameters
    ----------
    mf : pyscf.scf.SCF
        PySCF mean-field object.
    fragments : list of vayesta.qemb.Fragment subclasses
        List of fragments for the calculation, used to define local interaction spaces.

    Returns
    -------
    target_rots : list of np.array
        Rotations of particle-hole excitation space defining clusters.
    ovs_active : list of int
        Total number of local particle-hole excitations for each cluster.
    """

    def get_target_rot_spat(ro, rv):
        """
        Parameters
        ----------
        ro : list of tuples of np.array
            List of occupied orbital rotations defining clusters, with separate spin channels.
        rv : list of tuples of np.array
            List of virtual orbital rotations defining clusters, with separate spin channels.

        Returns
        -------
        rot : np.array
            Rotation of system particle-hole excitation space into cluster particle-hole excitation space.
        """
        no = ro.shape[1]
        nv = rv.shape[1]
        ov_active = no * nv
        rot = einsum("iJ,aB->JBia", ro, rv).reshape((ov_active, -1))
        return rot, ov_active

    nfrag = len(r_active_occs)
    assert(nfrag == len(r_active_virs))
    ovs_active = np.full((nfrag, 2), fill_value=0)

    target_rots = []

    for i, (r_o, r_v) in enumerate(zip(r_active_occs, r_active_virs)):

        if isinstance(r_o, np.ndarray):
            r_o = (r_o, r_o)
            r_v = (r_v, r_v)

        arot, ova = get_target_rot_spat(r_o[0], r_v[0])
        brot, ovb = get_target_rot_spat(r_o[1], r_v[1])

        spinorb_active_rot = scipy.linalg.block_diag(arot, brot)
        target_rots += [spinorb_active_rot]
        ovs_active[i] = np.array([ova, ovb])

    return target_rots, ovs_active
