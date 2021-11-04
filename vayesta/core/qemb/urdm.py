"""Routines to generate reduced density-matrices (RDMs) from spin-unrestricted quantum embedding calculations."""

import numpy as np

import pyscf
import pyscf.cc

from vayesta.core.util import *


def make_rdm1_demo(emb, ao_basis=False, add_mf=False, symmetrize=True):
    """Make democratically partitioned one-particle reduced density-matrix from fragment calculations.

    Warning: A democratically partitioned DM is only expected to yield reasonable results
    for full fragmentations (eg, Lowdin-AO or IAO+PAO fragmentation).

    Parameters
    ----------
    ao_basis: bool, optional
        Return the density-matrix in the AO basis. Default: False.
    add_mf: bool, optional
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
    if add_mf:
        sca = np.dot(ovlp, mo_coeff[0])
        scb = np.dot(ovlp, mo_coeff[1])
        dm1a_mf, dm1b_mf = emb.mf.make_rdm1()
        dm1a_mf = dot(sca.T, dm1a_mf, sca)
        dm1b_mf = dot(scb.T, dm1b_mf, scb)
        dm1a = dm1a_mf.copy()
        dm1b = dm1b_mf.copy()
    else:
        dm1a = np.zeros((emb.nmo[0], emb.nmo[0]))
        dm1b = np.zeros((emb.nmo[1], emb.nmo[1]))
    for f in emb.fragments:
        emb.log.debugv("Now adding projected DM of fragment %s", f)
        if f.results.dm1 is None:
            raise RuntimeError("DM1 not calculated for fragment %s!" % f)
        if emb.opts.dm_with_frozen:
            cf = f.mo_coeff
        else:
            cf = f.c_active
        rfa = dot(mo_coeff[0].T, ovlp, cf[0])
        rfb = dot(mo_coeff[1].T, ovlp, cf[1])
        if add_mf:
            # Subtract double counting:
            ddma = (f.results.dm1[0] - dot(rfa.T, dm1a_mf, rfa))
            ddmb = (f.results.dm1[1] - dot(rfb.T, dm1b_mf, rfb))
        else:
            ddma, ddmb = f.results.dm1
        pfa, pfb = f.get_fragment_projector(cf)
        dm1a += einsum('xi,ij,px,qj->pq', pfa, ddma, rfa, rfa)
        dm1b += einsum('xi,ij,px,qj->pq', pfb, ddmb, rfb, rfb)
    if ao_basis:
        dm1a = dot(mo_coeff[0], dm1a, mo_coeff[0].T)
        dm1b = dot(mo_coeff[1], dm1b, mo_coeff[1].T)
    if symmetrize:
        dm1a = (dm1a + dm1a.T)/2
        dm1b = (dm1b + dm1b.T)/2
    return (dm1a, dm1b)

def make_rdm1_uccsd(emb, ao_basis=False, partition=None, t_as_lambda=False, slow=False, faster=False):

    """Make one-particle reduced density-matrices from partitioned fragment CCSD wave functions.

    Parameters
    ----------
    ao_basis: bool, optional
        Return the density-matrix in the AO basis. Default: False.
    partition: ['first-occ', 'first-vir', 'democratic']
    t_as_lambda: bool, optional
        Use T-amplitudes instead of Lambda-amplitudes for CCSD density matrix. Default: False.
    slow: bool, optional
        Combine to global CCSD wave function first, then build density matrix.
        Equivalent, but does not scale well. Default: False

    Returns
    -------
    dm1a, dm1b: 2 (n, n) arrays
        One-particle reduced density matrices in AO (if `ao_basis=True`) or MO basis (default).
    """

    if slow:
        t1, t2 = emb.get_t12(partition=partition)
        #t1a, t1b = t1
        #t2aa, t2ab, t2bb = t2

        cc = pyscf.cc.uccsd.UCCSD(emb.mf)

        if t_as_lambda:
            l1= t1
            l2 = t2
        else:
            l1, l2 = emb.get_t12(get_lambda=True, partition=partition)
            #l1a, l1b = l1
            #l2aa, l2ab, l2bb = l2

        #t2 = (t2[0], np.zeros(t2[1].shape) , t2[2])
        #t2 = (np.zeros(t2[0].shape), t2[1], np.zeros(t2[2].shape))
        #t2 = (np.zeros(t2[0].shape), np.zeros(t2[1].shape), np.zeros(t2[2].shape))
        dm1a, dm1b = cc.make_rdm1(t1, t2, l1, l2, with_frozen=False)
        return dm1a, dm1b


    elif faster:
        raise NotImplementedError()
        t1a, t1b = emb.get_t1(partition=partition)
        l1a, l1b = ((t1a, t1b) if t_as_lambda else emb.get_t1(get_lambda=True, partition=partition))

        #Create lists of rotations from fragment to fragment and from fragment to mean field orbitals
        f2mfoa, f2mfob = [], []
        f2mfva, f2mfvb = [], []

        f2foa = [[] for i in range(emb.nfrag)]
        f2fob = [[] for i in range(emb.nfrag)]
        f2fva = [[] for i in range(emb.nfrag)]
        f2fvb = [[] for i in range(emb.nfrag)]

        ovlp = emb.get_ovlp()

        for i1, f1 in enumerate(emb.fragments):
            #pf.append(f1.get_fragment_projector())

            csoa = np.dot(f1.c_active_occ[0].T, ovlp)
            csob = np.dot(f1.c_active_occ[1].T, ovlp)

            csva = np.dot(f1.c_active_vir[0].T, ovlp)
            csvb = np.dot(f1.c_active_vir[1].T, ovlp)

            f2mfoa.append(np.dot(csoa, emb.mo_coeff_occ[0]))
            f2mfob.append(np.dot(csob, emb.mo_coeff_occ[1]))

            f2mfva.append(np.dot(csva, emb.mo_coeff_vir[0]))
            f2mfvb.append(np.dot(csvb, emb.mo_coeff_vir[1]))

            for i2, f2 in enumerate(emb.fragments):
                f2foa[i1].append(np.dot(csoa, f2.c_active_occ[0]))
                f2fob[i1].append(np.dot(csob, f2.c_active_occ[1]))
                f2fva[i1].append(np.dot(csva, f2.c_active_vir[0]))
                f2fvb[i1].append(np.dot(csvb, f2.c_active_vir[1]))


        nocca, nvira = t1a.shape
        noccb, nvirb = t1b.shape

        dooa = np.zeros((nocca, nocca))
        doob = np.zeros((noccb, noccb))
        dvva = np.zeros((nvira, nvira))
        dvvb = np.zeros((nvirb, nvirb))
        dova = np.zeros((nocca, nvira))
        dovb = np.zeros((noccb, nvirb))
        dvoa = np.zeros((nvira, nocca))
        dvob = np.zeros((nvirb, noccb))

        xt1a = np.zeros((nvira, nocca))
        xt1b = np.zeros((nvirb, noccb))
        xt2a = np.zeros((nvira, nocca))
        xt2b = np.zeros((nvirb, noccb))

        #Iterate over pairs of fragments and calculate their contribution to the 1RDMs
        for i1, f1 in enumerate(emb.fragments):
            t2 = f1.results.get_t2()
            t2aa, t2ab, t2bb = f1.project_amplitude_to_fragment(t2)


            dooa_f1 = np.zeros((f1.n_active_occ[0], f1.n_active_occ[0]))
            doob_f1 = np.zeros((f1.n_active_occ[1], f1.n_active_occ[1]))

            dvva_f1 = np.zeros((f1.n_active_vir[0], f1.n_active_vir[0]))
            dvvb_f1 = np.zeros((f1.n_active_vir[1], f1.n_active_vir[1]))

            dvoa_f1 = np.zeros((f1.n_active_vir[0], f1.n_active_occ[0]))
            dvob_f1 = np.zeros((f1.n_active_vir[1], f1.n_active_occ[1]))

            xt1a_f1 = np.zeros((f1.n_active_occ[0], f1.n_active_occ[0]))
            xt1b_f1 = np.zeros((f1.n_active_occ[1], f1.n_active_occ[1]))
            xt2a_f1 = np.zeros((f1.n_active_vir[0], f1.n_active_vir[0]))
            xt2b_f1 = np.zeros((f1.n_active_vir[1], f1.n_active_vir[1]))

            for i2, f2 in enumerate(emb.fragments):
                l1 = (f2.results.get_t1() if t_as_lambda else f2.results.l1)
                l2 = (f2.results.get_t2() if t_as_lambda else f2.results.l2)

                l1a_f2, l1b_f2 = f2.project_amplitude_to_fragment(l1)
                l2aa, l2ab, l2bb = f2.project_amplitude_to_fragment(l2)

                #Rotate f2 amplitudes to f1 basis
                l2aa = einsum('IJAB,iI,jJ,aA,bB->ijab', l2aa, f2foa[i1][i2], f2foa[i1][i2], f2fva[i1][i2], f2fva[i1][i2])

                l2ab = einsum('IJAB,iI,jJ,aA,bB->ijab', l2ab, f2foa[i1][i2], f2fob[i1][i2], f2fva[i1][i2], f2fvb[i1][i2])

                l2bb = einsum('IJAB,iI,jJ,aA,bB->ijab', l2bb, f2fob[i1][i2], f2fob[i1][i2], f2fvb[i1][i2], f2fvb[i1][i2])

                l1a_f2 = einsum('IA,iI,aA', l1a_f2, f2foa[i1][i2], f2foa[i1][i2])
                l1b_f2 = einsum('IA,iI,aA', l1a_f2, f2fob[i1][i2], f2fob[i1][i2])

                #Calculate fragment pair's contribution to 1RDMs, block by block
                #Following pyscf's notation in cc.uccsd_rdm.py

                # Occupied Occupied Blocks
                dooa_f1 -= einsum('imef,jmef->ij', l2aa, t2aa) * 0.5
                dooa_f1 -= einsum('imef,jmef->ij', l2ab, t2ab)

                doob_f1 -= einsum('imef,jmef->ij', l2bb, t2bb) * 0.5
                doob_f1 -= einsum('mief,mjef->ij', l2ab, t2ab)

                # Virtual Virtual Blocks
                dvva_f1 += einsum('mnae,mnbe->ab', l2ab, t2ab)
                dvva_f1 += einsum('mnae,mnbe->ab', l2aa, t2aa) * 0.5

                dvvb_f1 += einsum('mnea,mneb->ab', l2ab, t2ab)
                dvvb_f1 += einsum('mnae,mnbe->ab', l2bb, t2bb) * 0.5

                #Virtual Occupied Blocks
                xt1a_f1 += einsum('mnef,inef->mi', t2aa, l2aa) * 0.5
                xt1a_f1 += einsum('mnef,inef->mi', t2ab, l2ab)
                xt2a_f1 += einsum('mnaf,mnef->ae', t2aa, l2aa) * 0.5
                xt2a_f1 += einsum('mnaf,mnef->ae', t2ab, l2ab)

                dvoa_f1 += einsum('imae,me->ai', t2aa, l1a_f2)
                dvoa_f1 += einsum('imae,me->ai', t2ab, l1b_f2)

                xt1b_f1 += einsum('mnef,inef->mi', t2bb, l2bb) * 0.5
                xt1b_f1 += einsum('nmef,nief->mi', t2ab, l2ab)
                xt2b_f1 += einsum('mnaf,mnef->ae', t2bb, l2bb) * 0.5
                xt2b_f1 += einsum('mnfa,mnfe->ae', t2ab, l2ab)

                dvob_f1 += einsum('imae,me->ai', t2aa, l1a_f2)
                dvob_f1 += einsum('imae,me->ai', t2ab, l1b_f2)

            #Rotate fragment contributions to MO basis
            dooa += einsum('IJ,Ii,Jj->ij', dooa_f1, f2mfoa[i1], f2mfoa[i1])
            doob += einsum('IJ,Ii,Jj->ij', doob_f1, f2mfob[i1], f2mfob[i1])

            dvva += einsum('IJ,Ii,Jj->ij', dvva_f1, f2mfva[i1], f2mfva[i1])
            dvvb += einsum('IJ,Ii,Jj->ij', dvvb_f1, f2mfvb[i1], f2mfvb[i1])

            xt1a += einsum('IJ,Ii,Jj->ij', xt1a_f1, f2mfoa[i1], f2mfoa[i1])
            xt2a += einsum('IJ,Ii,Jj->ij', xt2a_f1, f2mfva[i1], f2mfva[i1])
            dvoa += einsum('ij,jk,kl->il', f2mfva[i1].T, dvoa_f1, f2mfoa[i1])

            #dvob += np.dot(f2mfvb[i1].T, dvob_f1)
            xt1b += einsum('IJ,Ii,Jj->ij', xt1b_f1, f2mfob[i1], f2mfob[i1])
            xt2b += einsum('IJ,Ii,Jj->ij', xt2b_f1, f2mfvb[i1], f2mfvb[i1])
            dvob += einsum('ij,jk,kl->il', f2mfvb[i1].T, dvob_f1, f2mfob[i1])

        dooa -= einsum('ie,je->ij', l1a, t1a)
        doob -= einsum('ie,je->ij', l1b, t1b)

        dvva += einsum('ma,mb->ab', t1a, l1a)
        dvvb += einsum('ma,mb->ab', t1b, l1b)

        xt2a += einsum('ma,me->ae', t1a, l1a)
        dvoa -= einsum('mi,ma->ai', xt1a, t1a)
        dvoa -= einsum('ie,ae->ai', t1a, xt2a)
        dvoa += t1a.T

        xt2b += einsum('ma,me->ae', t1b, l1b)
        dvob -= einsum('mi,ma->ai', xt1b, t1b)
        dvob -= einsum('ie,ae->ai', t1b, xt2b)
        dvob += t1b.T

        dova = l1a
        dovb = l1b

        nmoa = nocca + nvira
        dm1a = np.zeros((nmoa,nmoa))
        dm1a[:nocca,:nocca] = dooa + dooa.conj().T
        dm1a[:nocca,nocca:] = (dova + dvoa.conj().T )
        dm1a[nocca:,:nocca] = (dm1a[:nocca,nocca:].conj().T)
        dm1a[nocca:,nocca:] = dvva + dvva.conj().T
        dm1a *= 0.5
        #dm1a = (dm1a + dm1a.T)/2
        dm1a[np.diag_indices(nocca)] += 1

        nmob = noccb + nvirb
        dm1b = np.zeros((nmob, nmob))
        dm1b[:noccb,:noccb] = doob + doob.conj().T
        dm1b[:noccb,noccb:] = (dovb + dvob.conj().T )
        dm1b[noccb:,:noccb] = (dm1b[:noccb,noccb:].conj().T)
        dm1b[noccb:,noccb:] = dvvb + dvvb.conj().T
        dm1b *= 0.5
        #dm1a = (dm1a + dm1a.T)/2
        dm1b[np.diag_indices(noccb)] += 1

        return dm1a, dm1b


    else:
        t1a, t1b = emb.get_t1(partition=partition)
        l1a, l1b = ((t1a, t1b) if t_as_lambda else emb.get_t1(get_lambda=True, partition=partition))

        #Create lists of rotations from fragment to fragment and from fragment to mean field orbitals
        f2mfoa, f2mfob = [], []
        f2mfva, f2mfvb = [], []

        f2foa = [[] for i in range(emb.nfrag)]
        f2fob = [[] for i in range(emb.nfrag)]
        f2fva = [[] for i in range(emb.nfrag)]
        f2fvb = [[] for i in range(emb.nfrag)]

        ovlp = emb.get_ovlp()

        for i1, f1 in enumerate(emb.fragments):
            #pf.append(f1.get_fragment_projector())

            csoa = np.dot(f1.c_active_occ[0].T, ovlp)
            csob = np.dot(f1.c_active_occ[1].T, ovlp)

            csva = np.dot(f1.c_active_vir[0].T, ovlp)
            csvb = np.dot(f1.c_active_vir[1].T, ovlp)

            f2mfoa.append(np.dot(csoa, emb.mo_coeff_occ[0]))
            f2mfob.append(np.dot(csob, emb.mo_coeff_occ[1]))

            f2mfva.append(np.dot(csva, emb.mo_coeff_vir[0]))
            f2mfvb.append(np.dot(csvb, emb.mo_coeff_vir[1]))

            for i2, f2 in enumerate(emb.fragments):
                f2foa[i1].append(np.dot(csoa, f2.c_active_occ[0]))
                f2fob[i1].append(np.dot(csob, f2.c_active_occ[1]))
                f2fva[i1].append(np.dot(csva, f2.c_active_vir[0]))
                f2fvb[i1].append(np.dot(csvb, f2.c_active_vir[1]))


        nocca, nvira = t1a.shape
        noccb, nvirb = t1b.shape

        dooa = np.zeros((nocca, nocca))
        doob = np.zeros((noccb, noccb))
        dvva = np.zeros((nvira, nvira))
        dvvb = np.zeros((nvirb, nvirb))
        dova = np.zeros((nocca, nvira))
        dovb = np.zeros((noccb, nvirb))
        dvoa = np.zeros((nvira, nocca))
        dvob = np.zeros((nvirb, noccb))

        xt1a = np.zeros((nvira, nocca))
        xt1b = np.zeros((nvirb, noccb))
        xt2a = np.zeros((nvira, nocca))
        xt2b = np.zeros((nvirb, noccb))

        #Iterate over pairs of fragments and calculate their contribution to the 1RDMs
        for i1, f1 in enumerate(emb.fragments):
            t2 = f1.results.get_t2()
            t2aa, t2ab, t2bb = f1.project_amplitude_to_fragment(t2)


            dooa_f1 = np.zeros((f1.n_active_occ[0], nocca))
            doob_f1 = np.zeros((f1.n_active_occ[1], noccb))

            dvva_f1 = np.zeros((f1.n_active_vir[0], nvira))
            dvvb_f1 = np.zeros((f1.n_active_vir[1], nvirb))

            dvoa_f1 = np.zeros((f1.n_active_vir[0], f1.n_active_occ[0]))
            dvob_f1 = np.zeros((f1.n_active_vir[1], f1.n_active_occ[1]))

            xt1a_f1 = np.zeros((f1.n_active_vir[0], nocca))
            xt1b_f1 = np.zeros((f1.n_active_vir[1], nocca))
            xt2a_f1 = np.zeros((f1.n_active_vir[0], nocca))
            xt2b_f1 = np.zeros((f1.n_active_vir[1], nocca))

            for i2, f2 in enumerate(emb.fragments):
                l1 = (f2.results.get_t1() if t_as_lambda else f2.results.l1)
                l2 = (f2.results.get_t2() if t_as_lambda else f2.results.l2)

                l1a_f2, l1b_f2 = f2.project_amplitude_to_fragment(l1)
                l2aa, l2ab, l2bb = f2.project_amplitude_to_fragment(l2)

                """
                print(t2aa.shape)
                print(l2aa.shape)
                print(f2foa[i1][i2].shape)
                print(f2fva[i1][i2].shape)
                print(f2fva[i1][i2].shape)
                print(f2mfoa[i2].shape)
                """
                #Calculate fragment pair's contribution to 1RDMs, block by block
                #Following pyscf's notation in cc.uccsd_rdm.py

                # Occupied Occupied Blocks
                dooa_f1 -= 0.5 * einsum('imef,JMEF,mM,eE,fF,Jj->ij', t2aa, l2aa, f2foa[i1][i2], f2fva[i1][i2], f2fva[i1][i2], f2mfoa[i2])

                dooa_f1 -= einsum('imef,JMEF,mM,eE,fF,Jj->ij', t2ab, l2ab, f2fob[i1][i2], f2fva[i1][i2], f2fvb[i1][i2], f2mfoa[i2])

                doob_f1 -= 0.5 * einsum('imef,JMEF,mM,eE,fF,Jj->ij', t2bb, l2bb, f2fob[i1][i2], f2fvb[i1][i2], f2fvb[i1][i2], f2mfob[i2] )

                doob_f1 -= einsum('mief,MJEF,mM,eE,fF,Jj->ij', t2ab, l2ab, f2foa[i1][i2], f2fva[i1][i2], f2fvb[i1][i2], f2mfob[i2])

                #Virtual Virtual Blocks
                dvva_f1 += 0.5 * einsum('mnae,MNBE,mM,nN,eE,Bb->ab', t2aa, l2aa, f2foa[i1][i2], f2foa[i1][i2], f2fva[i1][i2], f2mfva[i2])

                dvva_f1 += einsum('mnae,MNBE,mM,nN,eE,Bb->ab', t2ab, l2ab, f2foa[i1][i2], f2fob[i1][i2], f2fvb[i1][i2], f2mfva[i2])

                dvvb_f1 += 0.5 * einsum('mnea,MNEB,mM,nN,eE,Bb->ab', t2bb, l2bb, f2fob[i1][i2], f2fob[i1][i2], f2fvb[i1][i2], f2mfvb[i2])

                dvvb_f1 += einsum('mnea,MNEB,mM,nN,eE,Bb->ab', t2ab, l2ab, f2foa[i1][i2], f2fob[i1][i2], f2fva[i1][i2], f2mfvb[i2])

                #Virtual Occupied Blocks

                #note can combine sum over f,n in xt1a-1 and xt2a-1
                xt1a_f1 += 0.5 * einsum('mnef,INEF,nN,eE,fF,Ii->mi', t2aa, l2aa, f2foa[i1][i2], f2fva[i1][i2], f2fva[i1][i2], f2mfoa[i2])

                xt1a_f1 += einsum('mnef,INEF,nN,eE,fF,Ii->mi', t2ab, l2ab, f2fob[i1][i2], f2fva[i1][i2], f2fvb[i1][i2], f2mfoa[i2])

                xt2a_f1 += 0.5 * einsum('mnaf,MNEF,mM,nN,fF,Ee->ae', t2aa, l2aa, f2foa[i1][i2], f2foa[i1][i2], f2fva[i1][i2], f2mfva[i2])

                xt2a_f1 += einsum('mnaf,MNEF,mM,nN,fF,Ee->ae', t2ab, l2ab, f2foa[i1][i2], f2fob[i1][i2], f2fvb[i1][i2], f2mfva[i2])

                dvoa_f1 += einsum('imae,ME,mM,eE->ai', t2aa, l1a_f2, f2foa[i1][i2], f2fva[i1][i2])

                dvoa_f1 += einsum('imae,ME,mM,eE->ai', t2ab, l1b_f2, f2fob[i1][i2], f2fvb[i1][i2])


                xt1b_f1 += 0.5 * einsum('mnef,INEF,nN,eE,fF,Ii->mi', t2bb, l2bb, f2fob[i1][i2], f2fvb[i1][i2], f2fvb[i1][i2], f2mfob[i2])

                xt1b_f1 += einsum('nmef,NIEF,nN,eE,fF,Ii->mi', t2ab, l2ab, f2foa[i1][i2], f2fva[i1][i2], f2fvb[i1][i2], f2mfob[i2])

                xt2b_f1 += 0.5 * einsum('mnaf,MNEF,mM,nN,fF,Ee->ae', t2bb, l2bb, f2fob[i1][i2], f2fob[i1][i2], f2fvb[i1][i2], f2mfvb[i2])

                xt2b_f1 += einsum('mnfa,MNFE,mM,nN,fF,Ee->ae', t2ab, l2ab, f2foa[i1][i2], f2fob[i1][i2], f2fva[i1][i2], f2mfvb[i2])

                dvob_f1 += einsum('imae,ME,mM,eE->ai', t2bb, l1b_f2, f2fob[i1][i2], f2fvb[i1][i2])

                dvob_f1 += einsum('miea,ME,mM,eE->ai', t2ab, l1a_f2, f2foa[i1][i2], f2fva[i1][i2])


            #Rotate fragment contributions to MO basis
            dooa += np.dot(f2mfoa[i1].T, dooa_f1)
            doob += np.dot(f2mfob[i1].T, doob_f1)

            dvva += np.dot(f2mfva[i1].T, dvva_f1)
            dvvb += np.dot(f2mfvb[i1].T, dvvb_f1)

            xt1a += np.dot(f2mfoa[i1].T, xt1a_f1)
            xt2a += np.dot(f2mfva[i1].T, xt2a_f1)
            dvoa += einsum('ij,jk,kl->il', f2mfva[i1].T, dvoa_f1, f2mfoa[i1])

            #dvob += np.dot(f2mfvb[i1].T, dvob_f1)
            xt1b += np.dot(f2mfob[i1].T, xt1b_f1)
            xt2b += np.dot(f2mfvb[i1].T, xt2b_f1)
            dvob += einsum('ij,jk,kl->il', f2mfvb[i1].T, dvob_f1, f2mfob[i1])

        dooa -= einsum('ie,je->ij', l1a, t1a)
        doob -= einsum('ie,je->ij', l1b, t1b)

        dvva += einsum('ma,mb->ab', t1a, l1a)
        dvvb += einsum('ma,mb->ab', t1b, l1b)

        xt2a += einsum('ma,me->ae', t1a, l1a)
        dvoa -= einsum('mi,ma->ai', xt1a, t1a)
        dvoa -= einsum('ie,ae->ai', t1a, xt2a)
        dvoa += t1a.T

        xt2b += einsum('ma,me->ae', t1b, l1b)
        dvob -= einsum('mi,ma->ai', xt1b, t1b)
        dvob -= einsum('ie,ae->ai', t1b, xt2b)
        dvob += t1b.T

        dova = l1a
        dovb = l1b

        nmoa = nocca + nvira
        dm1a = np.zeros((nmoa,nmoa))
        dm1a[:nocca,:nocca] = dooa + dooa.conj().T
        dm1a[:nocca,nocca:] = (dova + dvoa.conj().T )
        dm1a[nocca:,:nocca] = (dm1a[:nocca,nocca:].conj().T)
        dm1a[nocca:,nocca:] = dvva + dvva.conj().T
        dm1a *= 0.5
        #dm1a = (dm1a + dm1a.T)/2
        dm1a[np.diag_indices(nocca)] += 1

        nmob = noccb + nvirb
        dm1b = np.zeros((nmob, nmob))
        dm1b[:noccb,:noccb] = doob + doob.conj().T
        dm1b[:noccb,noccb:] = (dovb + dvob.conj().T )
        dm1b[noccb:,:noccb] = (dm1b[:noccb,noccb:].conj().T)
        dm1b[noccb:,noccb:] = dvvb + dvvb.conj().T
        dm1b *= 0.5
        #dm1a = (dm1a + dm1a.T)/2
        dm1b[np.diag_indices(noccb)] += 1

        return dm1a, dm1b
