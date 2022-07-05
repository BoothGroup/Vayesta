"""Expectation values for quantum embedding methods."""

import functools
import numpy as np
import vayesta
from vayesta.core.util import *
from vayesta.misc import corrfunc


def get_corrfunc_mf(emb, kind, dm1=None, atoms=None, projection='sao'):
    """dm1 in MO basis"""
    if dm1 is None:
        dm1 = np.zeros((emb.nmo, emb.nmo))
        dm1[np.diag_indices(emb.nocc)] = 2
    func = {
            'n,n': functools.partial(corrfunc.chargecharge, subtract_indep=False),
            'dn,dn': functools.partial(corrfunc.chargecharge, subtract_indep=True),
            'sz,sz': corrfunc.spinspin_z,
            }.get(kind.lower())
    if func is None:
        raise ValueError(kind)
    atoms1, atoms2, proj = emb._get_atom_projectors(atoms, projection)
    corr = np.zeros((len(atoms1), len(atoms2)))
    for a, atom1 in enumerate(atoms1):
        for b, atom2 in enumerate(atoms2):
            corr[a,b] = func(dm1, None, proj1=proj[atom1], proj2=proj[atom2])
    return corr

def get_corrfunc(emb, kind, dm1=None, dm2=None, atoms=None, projection='sao', dm2_with_dm1=None, use_symmetry=True):
    """Get expectation values <P(A) S_z P(B) S_z>, where P(X) are projectors onto atoms X.

    TODO: MPI

    Parameters
    ----------
    atoms : list[int] or list[list[int]], optional
        Atom indices for which the spin-spin correlation function should be evaluated.
        If set to None (default), all atoms of the system will be considered.
        If a list is given, all atom pairs formed from this list will be considered.
        If a list of two lists is given, the first list contains the indices of atom A,
        and the second of atom B, for which <Sz(A) Sz(B)> will be evaluated.
        This is useful in cases where one is only interested in the correlation to
        a small subset of atoms. Default: None

    Returns
    -------
    corr : array(N,M)
        Atom projected correlation function.
    """
    kind = kind.lower()
    if kind not in ('n,n', 'dn,dn', 'sz,sz'):
        raise ValueError(kind)

    # --- Setup
    f1, f2, f22 = {
            'n,n': (1, 2, 1),
            'dn,dn': (1, 2, 1),
            'sz,sz': (1/4, 1/2, 1/2),
            }[kind]
    if dm2_with_dm1 is None:
        dm2_with_dm1 = False
        if dm2 is not None:
            # Determine if DM2 contains DM1 by calculating norm
            norm = einsum('iikk->', dm2)
            ne2 = emb.mol.nelectron*(emb.mol.nelectron-1)
            dm2_with_dm1 = (norm > ne2/2)
    atoms1, atoms2, proj = emb._get_atom_projectors(atoms, projection)
    corr = np.zeros((len(atoms1), len(atoms2)))

    # 1-DM contribution:
    with log_time(emb.log.timing, "Time for 1-DM contribution: %s"):
        if dm1 is None:
            dm1 = emb.make_rdm1()
        for a, atom1 in enumerate(atoms1):
            tmp = np.dot(proj[atom1], dm1)
            for b, atom2 in enumerate(atoms2):
                corr[a,b] = f1*np.sum(tmp*proj[atom2])

    # Non-(approximate cumulant) DM2 contribution:
    if not dm2_with_dm1:
        with log_time(emb.log.timing, "Time for non-cumulant 2-DM contribution: %s"):
            occ = np.s_[:emb.nocc]
            occdiag = np.diag_indices(emb.nocc)
            ddm1 = dm1.copy()
            ddm1[occdiag] -= 1
            for a, atom1 in enumerate(atoms1):
                tmp = np.dot(proj[atom1], ddm1)
                for b, atom2 in enumerate(atoms2):
                    corr[a,b] -= f2*np.sum(tmp[occ] * proj[atom2][occ])       # N_atom^2 * N^2 scaling
            if kind in ('n,n', 'dn,dn'):
                # These terms are zero for Sz,Sz (but not in UHF)
                # Traces of projector*DM(HF) and projector*[DM(CC)+DM(HF)/2]:
                tr1 = {a: np.trace(p[occ,occ]) for a, p in proj.items()}  # DM(HF)
                tr2 = {a: np.sum(p * ddm1) for a, p in proj.items()}      # DM(CC) + DM(HF)/2
                for a, atom1 in enumerate(atoms1):
                    for b, atom2 in enumerate(atoms2):
                        corr[a,b] += f2*(tr1[atom1]*tr2[atom2] + tr1[atom2]*tr2[atom1])

    with log_time(emb.log.timing, "Time for cumulant 2-DM contribution: %s"):
        if dm2 is not None:
            # DM2(aa)               = (DM2 - DM2.transpose(0,3,2,1))/6
            # DM2(ab)               = DM2/2 - DM2(aa)
            # DM2(aa) - DM2(ab)]    = 2*DM2(aa) - DM2/2
            #                       = DM2/3 - DM2.transpose(0,3,2,1)/3 - DM2/2
            #                       = -DM2/6 - DM2.transpose(0,3,2,1)/3
            if kind in ('n,n', 'dn,dn'):
                pass
            elif kind == 'sz,sz':
                # DM2 is not needed anymore, so we can overwrite:
                dm2 = -(dm2/6 + dm2.transpose(0,3,2,1)/3)
            for a, atom1 in enumerate(atoms1):
                tmp = np.tensordot(proj[atom1], dm2)
                for b, atom2 in enumerate(atoms2):
                    corr[a,b] += f22*np.sum(tmp*proj[atom2])
        else:
            # Cumulant DM2 contribution:
            ffilter = dict(sym_parent=None) if use_symmetry else {}
            maxgen = None if use_symmetry else 0
            cst = np.dot(emb.get_ovlp(), emb.mo_coeff)
            for fx in emb.get_fragments(active=True, **ffilter):
                # Currently only defined for EWF
                # (but could also be defined for a democratically partitioned cumulant):
                dm2 = fx.make_fragment_dm2cumulant()
                if kind in ('n,n', 'dn,dn'):
                    pass
                # DM2(aa)               = (DM2 - DM2.transpose(0,3,2,1))/6
                # DM2(ab)               = DM2/2 - DM2(aa)
                # DM2(aa) - DM2(ab)]    = 2*DM2(aa) - DM2/2
                #                       = DM2/3 - DM2.transpose(0,3,2,1)/3 - DM2/2
                #                       = -DM2/6 - DM2.transpose(0,3,2,1)/3
                elif kind == 'sz,sz':
                    dm2 = -(dm2/6 + dm2.transpose(0,3,2,1)/3)

                for fx2, cx2_coeff in fx.loop_symmetry_children([fx.cluster.coeff], include_self=True, maxgen=maxgen):
                    rx = np.dot(cx2_coeff.T, cst)
                    projx = {atom: dot(rx, p_atom, rx.T) for (atom, p_atom) in proj.items()}
                    for a, atom1 in enumerate(atoms1):
                        tmp = np.tensordot(projx[atom1], dm2)
                        for b, atom2 in enumerate(atoms2):
                            corr[a,b] += f22*np.sum(tmp*projx[atom2])

    # Remove independent particle [P(A).DM1 * P(B).DM1] contribution
    if kind == 'dn,dn':
        for a, atom1 in enumerate(atoms1):
            for b, atom2 in enumerate(atoms2):
                corr[a,b] -= np.sum(dm1*proj[atom1]) * np.sum(dm1*proj[atom2])

    return corr
