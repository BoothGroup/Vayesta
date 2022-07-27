from vayesta.rpa import ssRIRPA
from vayesta.core.util import dot, einsum
import numpy as np
import scipy.linalg

def get_renorm_coulomb_interaction(mf, fragments, log=None, cderi=None):
    r_occs = [f.get_overlap('mo[occ]|cluster[occ]') for f in fragments]
    r_virs = [f.get_overlap('mo[vir]|cluster[vir]') for f in fragments]
    target_rots, ovs_active = get_target_rot(r_occs, r_virs)

    rpa = ssRIRPA(mf, log=log, Lpq=cderi)

    if sum(ovs_active) > 0:
        moms_interact, est_errors = rpa.kernel_moms(0, np.concatenate(target_rots, axis=0), npoints=48)
    else:
        moms_interact = np.zeros_like(np.concatenate(target_rots, axis=0))

    # Now need to separate into local contributions
    n = 0
    local_moments = []
    for nov, rot in zip(ovs_active, target_rots):
        local_moments += [dot(moms_interact[n:n+nov], rot.T)]
        n += nov

    # Then construct the RPA coupling matrix A-B, given by the diagonal matrix of energy differences.
    no = sum(mf.mo_occ > 0)
    nv = mf.mo_occ.shape[1] - no
    eps = np.zeros((no, nv))
    eps = eps + mf.mo_energy[no:]
    eps = (eps.T - mf.mo_energy[:no]).T
    eps = eps.reshape(-1)
    eps = np.concatenate(eps)
    # And use this to perform inversion to calculate interaction in cluster.
    renorm_coulomb = []
    for f, rot, mom in zip(fragments, target_rots, local_moments):
        amb = einsum("pn,qn,n->pq", rot, rot, eps)

        mominv = np.linalg.inv(mom)
        apb = dot(mominv, amb, mominv)
        # This is the renormalised coulomb kernel in the cluster.
        kc = 0.5 * (apb - amb)

def get_target_rot(r_active_occs, r_active_virs):
    # This essentially performs the function of edmet.fragment.get_rot_to_mf_ov, converting the active orbtals of each
    # cluster into a rotation of the overall mean-field particle-hole excitations.

    nfrag = len(r_active_occs)
    assert(nfrag == len(r_active_virs))
    ovs_active = np.zeros((nfrag,))

    target_rots = []

    for i, (r_o, r_v) in enumerate(zip(r_active_occs, r_active_virs)):
        no = r_o.shape[1]
        nv = r_v.shape[1]
        ov_active = no * nv

        spat_active_rot = einsum("iJ,aB->JBia", r_o, r_v).reshape((ov_active, ov_mf))
        spinorb_active_rot = scipy.linalg.block_diag(spat_active_rot, spat_active_rot)
        target_rots += [spinorb_active_rot]
        ovs_active[i] = 2*ov_active

    return target_rots, ovs_active
