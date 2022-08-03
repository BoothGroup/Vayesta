from vayesta.rpa import ssRIRPA
from vayesta.core.util import dot, einsum
import numpy as np
import scipy.linalg
import logging


def get_renorm_coulomb_interaction(mf, fragments, log=None, cderi_ov=None, loc_eris=None, calc_deltae=True):
    """Generates renormalised coulomb interactions for use in local cluster calculations.
    Currently requires unrestricted system.

    Parameters
    ----------
    mf : pyscf.scf.SCF
        PySCF mean-field object.
    fragments : list of vayesta.qemb.Fragment subclasses
        List of fragments for the calculation, used to define local interaction spaces.
    log : logging.Logger, optional.
        Logger object. If None, the logger of the `base` object is used. Default: None.
    cderi_ov : np.array or tuple of np.array, optional.
        Cholesky-decomposed ERIs in the particle-hole basis of mf. If mf is unrestricted
        this should be a list of arrays corresponding to the different spin channels.
    loc_eris : list of np.arrays or list of tuples of np.arrays, optional.
        List of ERIs in the particle-hole basis' of the fragments provided. If mf is
        unrestricted separated into spin channels.
    calc_ecorrection : bool, optional.
        Whether to calculate a nonlocal energy correction at the level of RPA
    :return:
    renorm_eris : list of tuples of np.array
        Spin-dependent renormalised ERI, for each fragment provided.
    loc_eris : list of tuples of np.array
        Local ERI for each fragment provided.
    deltae_rpa : float
        Delta RPA correction computed as difference between full system RPA energy and
        cluster correlation energies; currently only functional in CAS fragmentations.
    """
    if log is None:
        log = logging.getLogger(__name__)

    log.info("Renormalising Local Coulomb Interactions")
    log.info("----------------------------------------")

    if fragments[0].base.is_rhf:
        raise NotImplementedError("Currently, renormalised interactions require an unrestricted formalism "
                                  "due to spin dependence.")
    r_occs = [f.get_overlap('mo[occ]|cluster[occ]') for f in fragments]
    r_virs = [f.get_overlap('mo[vir]|cluster[vir]') for f in fragments]
    target_rots, ovs_active = get_target_rot(r_occs, r_virs)

    rpa = ssRIRPA(mf, log=log, Lpq=cderi_ov)

    if calc_deltae:
        deltae_rpa, energy_error = rpa.kernel_energy(correction="linear")
    else:
        deltae_rpa = None

    tr = np.concatenate(target_rots, axis=0)
    if sum(sum(ovs_active)) > 0:
        moms_interact, est_errors = rpa.kernel_moms(0, tr, npoints=48)
        momzero_interact = moms_interact[0]
    else:
        moms_interact = np.zeros_like(np.concatenate(tr, axis=0))

    # Now need to separate into local contributions
    n = 0
    local_moments = []
    for nov, rot in zip(ovs_active, target_rots):
        # Get corresponding section of overall moment, then project to just local contribution.
        local_moments += [dot(momzero_interact[n:n+sum(nov)], rot.T)]
        n += sum(nov)

    # Then construct the RPA coupling matrix A-B, given by the diagonal matrix of energy differences.
    no = np.array(sum(mf.mo_occ.T > 0))
    norb = mf.mo_coeff.shape[1]
    nv = norb - no

    def get_eps_singlespin(no_, nv_, mo_energy):
        eps = np.zeros((no_, nv_))
        eps = eps + mo_energy[no_:]
        eps = (eps.T - mo_energy[:no_]).T
        eps = eps.reshape(-1)
        return eps
    eps = np.concatenate([get_eps_singlespin(no[0], nv[0], mf.mo_energy[0]),
                          get_eps_singlespin(no[1], nv[1], mf.mo_energy[1])])

    # And use this to perform inversion to calculate interaction in cluster.

    calc_eris = loc_eris is None

    renorm_eris = []
    if loc_eris is None:
        def get_leris(f):
            coeff = f.cluster.c_active
            return (f.base.get_eris_array(coeff[0]),
                    f.base.get_eris_array((coeff[0], coeff[0], coeff[1], coeff[1])),
                    f.base.get_eris_array(coeff[1]))
        loc_eris = [get_leris(f) for f in fragments]

    for i, (f, rot, mom, (ova, ovb), leri) in enumerate(zip(fragments, target_rots, local_moments, ovs_active, loc_eris)):
        amb = einsum("pn,qn,n->pq", rot, rot, eps)
        mominv = np.linalg.inv(mom)
        apb = dot(mominv, amb, mominv)

        if calc_deltae:
            # Calculate the effective local correlation energy.
            loc_erpa = 0.5 * (dot(mom, apb).trace() - (amb.trace() + apb.trace())/2)
            # and deduct from total rpa energy to get nonlocal contribution.
            deltae_rpa -= loc_erpa

        # This is the renormalised coulomb kernel in the cluster.
        # Note that this is only defined in the particle-hole space, but has the same 8-fold symmetry
        # as the (assumed real) coulomb interaction.
        kc = 0.5 * (apb - amb)
        # Now need to strip out spin components of the renormalised interaction, and change to 4 orbital
        # indices.
        no = f.cluster.nocc_active
        nv = f.cluster.nvir_active
        kcaa = kc[:ova, :ova].reshape((no[0], nv[0], no[0], nv[0]))
        kcab = kc[:ova, ova:].reshape((no[0], nv[0], no[1], nv[1]))
        kcbb = kc[ova:, ova:].reshape((no[1], nv[1], no[1], nv[1]))

        log.info("Maximum spin symmetry breaking: %e \n"
                 "           and spin dependence; %e", abs(kcaa - kcbb).max(), abs(kcaa - kcab).max())

        def replace_ph(full, ph_rep):
            res = full.copy()
            no1 = ph_rep.shape[0]
            no2 = ph_rep.shape[2]
            res[:no1, no1:, :no2, no2:] = ph_rep
            res[no1:, :no1, :no2, no2:] = ph_rep.transpose([1, 0, 2, 3])
            res[:no1, no1:, no2:, :no2] = ph_rep.transpose([0, 1, 3, 2])
            res[no1:, :no1, no2:, :no2] = ph_rep.transpose([1, 0, 3, 2])

            log.info("Maximum ERI change due to renormalisation: %e (vs maximum eri value %e)",
                     abs(full - res).max(), abs(full[:no1, no1:, :no2, no2:]).max())
            return res


        renorm_eri = tuple([replace_ph(full, ph_rep) for full, ph_rep in zip(leri, [kcaa, kcab, kcbb])])
        renorm_eris += [renorm_eri]
    return renorm_eris, loc_eris, deltae_rpa

def get_target_rot(r_active_occs, r_active_virs):
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

        arot, ova = get_target_rot_spat(r_o[0], r_v[0])
        brot, ovb = get_target_rot_spat(r_o[1], r_v[1])

        spinorb_active_rot = scipy.linalg.block_diag(arot, brot)
        target_rots += [spinorb_active_rot]
        ovs_active[i] = np.array([ova, ovb])

    return target_rots, ovs_active
