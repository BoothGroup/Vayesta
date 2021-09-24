"""Routines to generate reduced density-matrices (RDMs) from quantum embedding calculations."""

import numpy as np

import pyscf
import pyscf.cc

from vayesta.core.util import *


def make_rdm1_demo(self, ao_basis=False, add_mf=True, symmetrize=True):
    """Make democratically partitioned one-particle reduced density-matrix from fragment calculations.

    Warning: A democratically partitioned DM is only expected to yield reasonable results
    for full fragmentations (eg, Lowdin-AO or IAO+PAO fragmentation).

    Parameters
    ----------
    ao_basis: bool, optional
        Return the density-matrix in the AO basis. Default: False.
    add_mf: bool, optional
        Add the mean-field contribution to the density-matrix (double counting is accounted for).
        Is only used if `partition = 'dm'`. Default: True.
    symmetrize: bool, optional
        Symmetrize the density-matrix at the end of the calculation. Default: True.

    Returns
    -------
    dm1: (n, n) array
        One-particle reduced density matrix in AO (if `ao_basis=True`) or MO basis (default).
    """
    if add_mf:
        sc = np.dot(self.get_ovlp(), self.mo_coeff)
        dm1_mf = dot(sc.T, self.mf.make_rdm1(), sc)
        dm1 = dm1_mf.copy()
    else:
        dm1 = np.zeros((self.nmo, self.nmo))
    for f in self.fragments:
        self.log.debugv("Now adding projected DM of fragment %s", f)
        if f.results.dm1 is None:
            raise RuntimeError("DM1 not calculated for fragment %s!" % f)
        if self.opts.dm_with_frozen:
            cf = f.mo_coeff
        else:
            cf = f.c_active
        rf = dot(self.mo_coeff.T, self.get_ovlp(), cf)
        if add_mf:
            # Subtract double counting:
            ddm = (f.results.dm1 - dot(rf.T, dm1_mf, rf))
        else:
            ddm = f.results.dm1
        pf = f.get_fragment_projector(cf)
        dm1 += einsum('xi,ij,px,qj->pq', pf, ddm, rf, rf)
    if ao_basis:
        dm1 = dot(self.mo_coeff, dm1, self.mo_coeff.T)
    if symmetrize:
        dm1 = (dm1 + dm1.T)/2
    return dm1

def make_rdm1_ccsd(self, ao_basis=False, symmetrize=True, partition=None, t_as_lambda=False, slow=False):
    """Make one-particle reduced density-matrix from partitioned fragment CCSD wave functions.

    Parameters
    ----------
    ao_basis: bool, optional
        Return the density-matrix in the AO basis. Default: False.
    symmetrize: bool, optional
        Symmetrize the density-matrix at the end of the calculation. Default: True.
    partition: ['first-occ', 'first-vir', 'democratic']
    t_as_lambda: bool, optional
        Use T-amplitudes instead of Lambda-amplitudes for CCSD density matrix. Default: False.
    slow: bool, optional
        Combine to global CCSD wave function first, then build density matrix.
        Equivalent, but does not scale well. Default: False

    Returns
    -------
    dm1: (n, n) array
        One-particle reduced density matrix in AO (if `ao_basis=True`) or MO basis (default).
    """

    if slow:
        t1, t2 = self.get_t12(partition=partition)
        cc = pyscf.cc.ccsd.CCSD(self.mf)
        #cc.conv_tol = 1e-12
        #cc.conv_tol_normt = 1e-10
        #if 'full-l' in partition:
        #    l1 = l2 = None
        if t_as_lambda:
            l1, l2 = t1, t2
        else:
            l1, l2 = self.get_t12(get_lambda=True, partition=partition)
        dm1 = cc.make_rdm1(t1=t1, t2=t2, l1=l1, l2=l2, with_frozen=False)

    else:
        # T1/L1-amplitudes can be summed directly
        t1 = self.get_t1(partition=partition)
        l1 = (t1 if t_as_lambda else self.get_t1(get_lambda=True, partition=partition))

        # --- Preconstruct some C^T.S.C rotation matrices:
        # Fragment orbital projectors
        pf = []
        # Fragment to mean-field occupied/virtual
        f2mfo = []
        f2mfv = []
        # Fragment to other fragment occupied/virtual
        f2fo = [[] for i in range(self.nfrag)]
        f2fv = [[] for i in range(self.nfrag)]
        ovlp = self.get_ovlp()
        for i1, f1 in enumerate(self.fragments):
            pf.append(f1.get_fragment_projector(f1.c_active_occ))
            cso = np.dot(f1.c_active_occ.T, ovlp)
            csv = np.dot(f1.c_active_vir.T, ovlp)
            f2mfo.append(np.dot(cso, self.mo_coeff_occ))
            f2mfv.append(np.dot(csv, self.mo_coeff_vir))
            for i2, f2 in enumerate(self.fragments):
                f2fo[i1].append(np.dot(cso, f2.c_active_occ))
                f2fv[i1].append(np.dot(csv, f2.c_active_vir))

        # --- Loop over pairs of fragments and add projected density-matrix contributions
        nocc, nvir = t1.shape
        doo = np.zeros((nocc, nocc))
        dvv = np.zeros((nvir, nvir))
        dov = (t1 + l1 - einsum('ie,me,ma->ia', t1, l1, t1))
        for i1, f1 in enumerate(self.fragments):
            theta = f1.results.get_t2()
            theta = (2*theta - theta.transpose(0,1,3,2))
            theta = f1.project_amplitude_to_fragment(theta, partition=partition)
            # Intermediates [leave left index in cluster basis]:
            doo_f1 = np.zeros((f1.n_active_occ, nocc))
            dvv_f1 = np.zeros((f1.n_active_vir, nvir))
            dov += einsum('imae,ip,mM,aq,eE,ME->pq', theta, f2mfo[i1], f2mfo[i1], f2mfv[i1], f2mfv[i1], l1)
            for i2, f2 in enumerate(self.fragments):
                #l2 = (f2.results.t2 if t_as_lambda else f2.results.l2)
                l2 = (f2.results.get_t2() if t_as_lambda else f2.results.l2)
                l2 = f2.project_amplitude_to_fragment(l2, partition=partition)
                ## Theta_jk^ab * l_ik^ab -> ij
                #doo -= einsum('jkab,IKAB,jp,kK,aA,bB,Iq->pq', theta_f1, l2_f2,
                #        f2mfo[i1], f2fo[i1][i2], f2fv[i1][i2], f2fv[i1][i2], f2mfo[i2])
                ## Theta_ji^ca * l_ji^cb -> ab
                #dvv += einsum('jica,JICB,jJ,iI,cC,ap,Bq->pq', theta_f1, l2_f2,
                #        f2fo[i1][i2], f2fo[i1][i2], f2fv[i1][i2], f2mfv[i1], f2mfv[i2])

                # Theta_jk^ab * l_ik^ab -> ij
                doo_f1 -= einsum('jkab,IKAB,kK,aA,bB,Iq->jq', theta, l2,
                        f2fo[i1][i2], f2fv[i1][i2], f2fv[i1][i2], f2mfo[i2])
                # Theta_ji^ca * l_ji^cb -> ab
                dvv_f1 += einsum('jica,JICB,jJ,iI,cC,Bq->aq', theta, l2,
                        f2fo[i1][i2], f2fo[i1][i2], f2fv[i1][i2], f2mfv[i2])

                # Theta_jk^ab * l_ik^ab -> ij
                #doo -= einsum('jx,Iy,jkab,IKAB,xp,kK,aA,bB,yq->pq',
                #        pf[i1], pf[i2], theta_f1, l2_f2,
                #        f2mfo[i1], f2fo[i1][i2], f2fv[i1][i2], f2fv[i1][i2], f2mfo[i2])
                ### Theta_ji^ca * l_ji^cb -> ab
                #dvv += einsum('jx,Jy,jica,JICB,xy,iI,cC,ap,Bq->pq',
                #        pf[i1], pf[i2], theta_f1, l2_f2,
                #        f2fo[i1][i2], f2fo[i1][i2], f2fv[i1][i2], f2mfv[i1], f2mfv[i2])
            doo += np.dot(f2mfo[i1].T, doo_f1)
            dvv += np.dot(f2mfv[i1].T, dvv_f1)

        dov += einsum('im,ma->ia', doo, t1)
        dov -= einsum('ie,ae->ia', t1, dvv)
        doo -= einsum('ja,ia->ij', t1, l1)
        dvv += einsum('ia,ib->ab', t1, l1)

        nmo = (nocc + nvir)
        occ, vir = np.s_[:nocc], np.s_[nocc:]
        dm1 = np.zeros((nmo, nmo))
        dm1[occ,occ] = (doo + doo.T)
        dm1[vir,vir] = (dvv + dvv.T)
        dm1[occ,vir] = dov
        dm1[vir,occ] = dov.T
        dm1[np.diag_indices(nocc)] += 2

    if ao_basis:
        dm1 = dot(self.mo_coeff, dm1, self.mo_coeff.T)
    if symmetrize:
        dm1 = (dm1 + dm1.T)/2
    return dm1

def make_rdm2_demo(self, ao_basis=False, add_mf=True, symmetrize=True):
    """Make democratically partitioned two-particle reduced density-matrix from fragment calculations.

    Warning: A democratically partitioned DM is only expected to yield reasonable results
    for full fragmentations (eg, Lowdin-AO or IAO+PAO fragmentation).

    Parameters
    ----------
    ao_basis: bool, optional
        Return the density-matrix in the AO basis. Default: False.
    add_mf: bool, optional
        Add the mean-field contribution to the density-matrix (double counting is accounted for).
        Is only used if `partition = 'dm'`. Default: True.
    symmetrize: bool, optional
        Symmetrize the density-matrix at the end of the calculation. Default: True.

    Returns
    -------
    dm2: (n, n, n, n) array
        Two-particle reduced density matrix in AO (if `ao_basis=True`) or MO basis (default).
    """
    if add_mf:
        #dm2_mf = np.zeros(4*[self.nmo])
        #for i in range(self.nocc):
        #    for j in range(self.nocc):
        #        dm2_mf[i,i,j,j] += 4.0
        #        dm2_mf[i,j,j,i] -= 2.0
        sc = np.dot(self.get_ovlp(), self.mo_coeff)
        dm1_mf = np.linalg.multi_dot((sc.T, self.mf.make_rdm1(), sc))
        dm2_mf = einsum('ij,kl->ijkl', dm1_mf, dm1_mf) - einsum('il,jk->ijkl', dm1_mf, dm1_mf)/2
        dm2 = dm2_mf.copy()
    else:
        dm2 = np.zeros((self.nmo, self.nmo, self.nmo, self.nmo))

    for f in self.fragments:
        if f.results.dm2 is None:
            raise RuntimeError("DM2 not calculated for fragment %s!" % f)
        if self.opts.dm_with_frozen:
            cf = f.mo_coeff
        else:
            cf = f.c_active
        rf = np.linalg.multi_dot((self.mo_coeff.T, self.get_ovlp(), cf))
        if add_mf:
            # Subtract double counting:
            ddm = (f.results.dm2 - einsum('IJKL,Ii,Jj,Kk,Ll->ijkl', dm2_mf, rf, rf, rf, rf))
        else:
            ddm = f.results.dm2
        pf = f.get_fragment_projector(cf)
        dm2 += einsum('xi,ijkl,px,qj,rk,sl->pqrs', pf, ddm, rf, rf, rf, rf)
    if ao_basis:
        dm2 = einsum('ijkl,pi,qj,rk,sl->pqrs', dm2, *(4*[self.mo_coeff]))
    if symmetrize:
        dm2 = (dm2 + dm2.transpose(1,0,3,2))/2
    return dm2


def make_rdm2_ccsd(self, ao_basis=False, symmetrize=True, partition=None, t_as_lambda=False, slow=True):
    """Recreate global two-particle reduced density-matrix from fragment calculations.

    Parameters
    ----------
    ao_basis: bool, optional
        Return the density-matrix in the AO basis. Default: False.
    symmetrize: bool, optional
        Symmetrize the density-matrix at the end of the calculation. Default: True.
    partition: ['first-occ', 'first-vir', 'democratic']
    t_as_lambda: bool, optional
        Use T-amplitudes instead of Lambda-amplitudes for CCSD density matrix. Default: False.
    slow: bool, optional
        Combine to global CCSD wave function first, then build density matrix.
        Equivalent, but does not scale well. Default: True.

    Returns
    -------
    dm2: (n, n, n, n) array
        Two-particle reduced density matrix in AO (if `ao_basis=True`) or MO basis (default).
    """

    if slow:
        t1, t2 = self.get_t12(partition=partition)
        cc = pyscf.cc.ccsd.CCSD(self.mf)
        #if 'l12_full' in partition:
        #    l1 = l2 = None
        if t_as_lambda:
            l1, l2 = t1, t2
        else:
            l1, l2 = self.get_t12(get_lambda=True, partition=partition)
        dm2 = cc.make_rdm2(t1=t1, t2=t2, l1=l1, l2=l2, with_frozen=False)
    else:
        raise NotImplementedError()
    if ao_basis:
        dm2 = einsum('ijkl,pi,qj,rk,sl->pqrs', dm2, *(4*[self.mo_coeff]))
    if symmetrize:
        dm2 = (dm2 + dm2.transpose(1,0,3,2))/2
    return dm2
