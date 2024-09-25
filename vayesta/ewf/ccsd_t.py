import numpy as np
import pyscf.lib


def calc_fragment_ccsd_t_energy(fragment, **kwargs):
    if fragment.base.is_rhf:
        return calc_fragment_rccsd_t_energy(fragment, **kwargs)
    elif fragment.base.is_uhf:
        return calc_fragment_uccsd_t_energy(fragment, **kwargs)
    else:
        raise NotImplementedError()

def calc_fragment_rccsd_t_energy(fragment, t1=None, t2=None, eris=None, project='w', global_t1=False):
    """
    Calculates a fragment CCSD(T) energy contribution.

    Modified expressions obtained from pyscf.cc.ccsd_t_slow, and
    JCP 94, 442 (1991); DOI:10.1063/1.460359.  Error in Eq (1), should be [ia] >= [jb] >= [kc]
    
    """

    einsum = pyscf.lib.einsum


    if global_t1 and (t1 is None):
        t1 = fragment.base.get_global_t1()
        c_occ, c_vir = fragment.get_overlap('mo[occ]|cluster[occ]'), fragment.get_overlap('mo[vir]|cluster[vir]')
        t1 = c_occ.T @ t1 @ c_vir
        t1T = t1.T
    elif (not global_t1) and (t1 is None):
        t1T = fragment.results.wf.as_ccsd().t1.T
    elif t1 is not None:
        t1T = t1.T

    nvir, nocc = t1T.shape

    if t2 is None:
        t2T = fragment.results.wf.as_ccsd().t2.transpose(2,3,0,1)
        

    fvo = fragment.hamil.get_fock(with_exxdiv=True)[nocc:,:nocc]

    mo_e = fragment.hamil.get_clus_mf_info(with_exxdiv=True)[2]
    e_occ, e_vir = mo_e[:nocc], mo_e[nocc:]
    eijk = pyscf.lib.direct_sum('i,j,k->ijk', e_occ, e_occ, e_occ)


    # NOTE: These transpositions are for indexing the outer loop, they are NOT the vvov, vooo and vvoo blocks of the ERIS
    if eris is not None:
        eris_vvov = eris.get_ovvv().conj().transpose(1,3,0,2)
        eris_vooo = np.asarray(eris.ovoo).conj().transpose(1,0,2,3)
        eris_vvoo = np.asarray(eris.ovov).conj().transpose(1,3,0,2)
    else:
        eris_vvov = fragment.hamil.get_eris_bare(block='ovvv').conj().transpose(1,3,0,2)
        eris_vooo = fragment.hamil.get_eris_bare(block='ovoo').conj().transpose(1,0,2,3)
        eris_vvoo = fragment.hamil.get_eris_bare(block='ovov').conj().transpose(1,3,0,2)
        
    def get_w(a, b, c):
        w = einsum('if,fkj->ijk', eris_vvov[a,b], t2T[c,:])
        w-= einsum('ijm,mk->ijk', eris_vooo[a,:], t2T[b,c])
        return w
    def get_v(a, b, c):
        v = einsum('ij,k->ijk', eris_vvoo[a,b], t1T[c])
        v+= einsum('ij,k->ijk', t2T[a,b], fvo[c])
        return v
    
    def sym_proj(expr):
        assert len(expr.shape) == 3
        cf = fragment.get_overlap('cluster[occ]|frag[occ]')
        cfc = cf @ cf.T

        sym_expr =  einsum('iI,Ijk->ijk', cfc, expr)
        sym_expr += einsum('jJ,iJk->ijk', cfc, expr)
        sym_expr += einsum('kK,ijK->ijk', cfc, expr)

        return  1/3 * sym_expr

    et = 0
    for a in range(nvir):
        for b in range(a+1):
            for c in range(b+1):
                d3 = eijk - e_vir[a] - e_vir[b] - e_vir[c]
                if a == c:  # a == b == c
                    d3 *= 6
                elif a == b or b == c:
                    d3 *= 2


                wabc = get_w(a, b, c)
                wacb = get_w(a, c, b)
                wbac = get_w(b, a, c)
                wbca = get_w(b, c, a)
                wcab = get_w(c, a, b)
                wcba = get_w(c, b, a)
                
                vabc = get_v(a, b, c)
                vacb = get_v(a, c, b)
                vbac = get_v(b, a, c)
                vbca = get_v(b, c, a)
                vcab = get_v(c, a, b)
                vcba = get_v(c, b, a)

                zabc = r3(wabc + .5 * vabc) / d3
                zacb = r3(wacb + .5 * vacb) / d3
                zbac = r3(wbac + .5 * vbac) / d3
                zbca = r3(wbca + .5 * vbca) / d3
                zcab = r3(wcab + .5 * vcab) / d3
                zcba = r3(wcba + .5 * vcba) / d3

                if project == 'w':
                    wabc = sym_proj(wabc)
                    wacb = sym_proj(wacb)
                    wbac = sym_proj(wbac)
                    wbca = sym_proj(wbca)
                    wcab = sym_proj(wcab)
                    wcba = sym_proj(wcba)
                # elif project == 'v':
                #     vabc = get_v(a, b, c)
                #     vacb = get_v(a, c, b)
                #     vbac = get_v(b, a, c)
                #     vbca = get_v(b, c, a)
                #     vcab = get_v(c, a, b)
                #     vcba = get_v(c, b, a)
                elif project == 'z':
                    zabc = sym_proj(zabc)
                    zacb = sym_proj(zacb)
                    zbac = sym_proj(zbac)
                    zbca = sym_proj(zbca)
                    zcab = sym_proj(zcab)
                    zcba = sym_proj(zcba)
                else:
                    raise NotImplementedError()

                et+= einsum('ijk,ijk', wabc, zabc.conj())
                et+= einsum('ikj,ijk', wacb, zabc.conj())
                et+= einsum('jik,ijk', wbac, zabc.conj())
                et+= einsum('jki,ijk', wbca, zabc.conj())
                et+= einsum('kij,ijk', wcab, zabc.conj())
                et+= einsum('kji,ijk', wcba, zabc.conj())

                et+= einsum('ijk,ijk', wacb, zacb.conj())
                et+= einsum('ikj,ijk', wabc, zacb.conj())
                et+= einsum('jik,ijk', wcab, zacb.conj())
                et+= einsum('jki,ijk', wcba, zacb.conj())
                et+= einsum('kij,ijk', wbac, zacb.conj())
                et+= einsum('kji,ijk', wbca, zacb.conj())

                et+= einsum('ijk,ijk', wbac, zbac.conj())
                et+= einsum('ikj,ijk', wbca, zbac.conj())
                et+= einsum('jik,ijk', wabc, zbac.conj())
                et+= einsum('jki,ijk', wacb, zbac.conj())
                et+= einsum('kij,ijk', wcba, zbac.conj())
                et+= einsum('kji,ijk', wcab, zbac.conj())

                et+= einsum('ijk,ijk', wbca, zbca.conj())
                et+= einsum('ikj,ijk', wbac, zbca.conj())
                et+= einsum('jik,ijk', wcba, zbca.conj())
                et+= einsum('jki,ijk', wcab, zbca.conj())
                et+= einsum('kij,ijk', wabc, zbca.conj())
                et+= einsum('kji,ijk', wacb, zbca.conj())

                et+= einsum('ijk,ijk', wcab, zcab.conj())
                et+= einsum('ikj,ijk', wcba, zcab.conj())
                et+= einsum('jik,ijk', wacb, zcab.conj())
                et+= einsum('jki,ijk', wabc, zcab.conj())
                et+= einsum('kij,ijk', wbca, zcab.conj())
                et+= einsum('kji,ijk', wbac, zcab.conj())

                et+= einsum('ijk,ijk', wcba, zcba.conj())
                et+= einsum('ikj,ijk', wcab, zcba.conj())
                et+= einsum('jik,ijk', wbca, zcba.conj())
                et+= einsum('jki,ijk', wbac, zcba.conj())
                et+= einsum('kij,ijk', wacb, zcba.conj())
                et+= einsum('kji,ijk', wabc, zcba.conj())
    et *= 2
    #log.info('CCSD(T) correction = %.15g', et)
    return et

def r3(w):
    return (4 * w + w.transpose(1,2,0) + w.transpose(2,0,1)
            - 2 * w.transpose(2,1,0) - 2 * w.transpose(0,2,1)
            - 2 * w.transpose(1,0,2))


def calc_fragment_uccsd_t_energy(fragment, t1=None, t2=None, eris=None, project='w', global_t1=False):

    einsum = pyscf.lib.einsum
    lib = pyscf.lib

    def p6(t):
        return (t + t.transpose(1,2,0,4,5,3) +
                t.transpose(2,0,1,5,3,4) + t.transpose(0,2,1,3,5,4) +
                t.transpose(2,1,0,5,4,3) + t.transpose(1,0,2,4,3,5))
    def r6(w):
        return (w + w.transpose(2,0,1,3,4,5) + w.transpose(1,2,0,3,4,5)
                - w.transpose(2,1,0,3,4,5) - w.transpose(0,2,1,3,4,5)
                - w.transpose(1,0,2,3,4,5))
    
    def sym_proj(expr, spin):
        assert len(expr.shape) == 6
        cfa, cfb = fragment.get_overlap('cluster[occ]|frag[occ]')
        cfca, cfcb = cfa @ cfa.T, cfb @ cfb.T
        cfc = dict(a=cfca, b=cfcb)
        sym_expr =  einsum('iI,Ijkabc->ijkabc', cfc[spin[0]], expr)
        sym_expr += einsum('jJ,iJkabc->ijkabc', cfc[spin[1]], expr)
        sym_expr += einsum('kK,ijKabc->ijkabc', cfc[spin[2]], expr)

        return  1/3 * sym_expr


    if global_t1 and (t1 is None):
        t1a, t1b = fragment.base.get_global_t1()
        c_occ, c_vir = fragment.get_overlap('mo[occ]|cluster[occ]'), fragment.get_overlap('mo[vir]|cluster[vir]')
        t1a = c_occ[0].T @ t1a @ c_vir[0]
        t1b = c_occ[1].T @ t1b @ c_vir[1]
    elif (not global_t1) and (t1 is None):
        t1a, t1b = fragment.results.wf.as_ccsd().t1
    elif t1 is not None:
        t1a, t1b = t1

    if t2 is None:
        t2aa, t2ab, t2bb = fragment.results.wf.as_ccsd().t2
    else:
        t2aa, t2ab, t2bb = t2

    nocca, noccb = t2ab.shape[:2]
    mo_ea, mo_eb = fragment.hamil.get_clus_mf_info(with_exxdiv=True)[2]
    eia = mo_ea[:nocca,None] - mo_ea[nocca:]
    eIA = mo_eb[:noccb,None] - mo_eb[noccb:]
    focka, fockb = fragment.hamil.get_fock(with_exxdiv=True)
    fvo = focka[nocca:,:nocca]
    fVO = fockb[noccb:,:noccb]

    if eris is not None:
        eris_ovvv = numpy.asarray(eris.get_ovvv()).conj()
        eris_ovoo = numpy.asarray(eris.ovoo).conj()
        eris_ovov = numpy.asarray(eris.ovov).conj()
        eris_OVVV = numpy.asarray(eris.get_OVVV()).conj()
        eris_OVOO = numpy.asarray(eris.OVOO).conj()
        eris_OVOV = numpy.asarray(eris.OVOV).conj()
        eris_ovVV = numpy.asarray(eris.get_ovVV()).conj()
        eris_OVvv = numpy.asarray(eris.get_OVvv()).conj()
        eris_ovOO = numpy.asarray(eris.ovOO).conj()
        eris_OVoo = numpy.asarray(eris.OVoo).conj()
        eris_ovOV = numpy.asarray(eris.ovOV).conj()
    else:
        eris_ovvv = fragment.hamil.get_eris_bare(block='ovvv').conj()
        eris_ovoo = fragment.hamil.get_eris_bare(block='ovoo').conj()
        eris_ovov = fragment.hamil.get_eris_bare(block='ovov').conj()
        eris_OVVV = fragment.hamil.get_eris_bare(block='OVVV').conj()
        eris_OVOO = fragment.hamil.get_eris_bare(block='OVOO').conj()
        eris_OVOV = fragment.hamil.get_eris_bare(block='OVOV').conj()
        eris_ovVV = fragment.hamil.get_eris_bare(block='ovVV').conj()
        eris_OVvv = fragment.hamil.get_eris_bare(block='OVvv').conj()
        eris_ovOO = fragment.hamil.get_eris_bare(block='ovOO').conj()
        eris_OVoo = fragment.hamil.get_eris_bare(block='OVoo').conj()
        eris_ovOV = fragment.hamil.get_eris_bare(block='ovOV').conj()
        
    # aaa
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eia, eia, eia)
    w = einsum('ijae,kceb->ijkabc', t2aa, eris_ovvv)
    w-= einsum('mkbc,iajm->ijkabc', t2aa, eris_ovoo)
    r = r6(w)
    v = einsum('jbkc,ia->ijkabc', eris_ovov, t1a)
    v+= einsum('jkbc,ai->ijkabc', t2aa, fvo) * .5
    wvd = p6(w + v) / d3

    if project == 'w':
        wvd = sym_proj(wvd, 'aaa')
    elif project == 'r':
        r = sym_proj(r, 'aaa')
    else:
        raise NotImplementedError()
    et = einsum('ijkabc,ijkabc', wvd.conj(), r)

    # bbb
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eIA, eIA)
    w = einsum('ijae,kceb->ijkabc', t2bb, eris_OVVV)
    w-= einsum('imab,kcjm->ijkabc', t2bb, eris_OVOO)
    r = r6(w)
    v = einsum('jbkc,ia->ijkabc', eris_OVOV, t1b)
    v+= einsum('jkbc,ai->ijkabc', t2bb, fVO) * .5
    wvd = p6(w + v) / d3

    if project == 'w':
        wvd = sym_proj(wvd, 'bbb')
    elif project == 'r':
        r = sym_proj(r, 'bbb')
    else:
        raise NotImplementedError()
    et += einsum('ijkabc,ijkabc', wvd.conj(), r)

    # baa
    w  = einsum('jIeA,kceb->IjkAbc', t2ab, eris_ovvv) * 2
    w += einsum('jIbE,kcEA->IjkAbc', t2ab, eris_ovVV) * 2
    w += einsum('jkbe,IAec->IjkAbc', t2aa, eris_OVvv)
    w -= einsum('mIbA,kcjm->IjkAbc', t2ab, eris_ovoo) * 2
    w -= einsum('jMbA,kcIM->IjkAbc', t2ab, eris_ovOO) * 2
    w -= einsum('jmbc,IAkm->IjkAbc', t2aa, eris_OVoo)
    r = w - w.transpose(0,2,1,3,4,5)
    r = r + r.transpose(0,2,1,3,5,4)
    v  = einsum('jbkc,IA->IjkAbc', eris_ovov, t1b)
    v += einsum('kcIA,jb->IjkAbc', eris_ovOV, t1a)
    v += einsum('kcIA,jb->IjkAbc', eris_ovOV, t1a)
    v += einsum('jkbc,AI->IjkAbc', t2aa, fVO) * .5
    v += einsum('kIcA,bj->IjkAbc', t2ab, fvo) * 2
    w += v
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eia, eia)
    r /= d3

    if project == 'w':
        w = sym_proj(w, 'baa')
    elif project == 'r':
        r = sym_proj(r, 'baa')
    else:
        raise NotImplementedError()
    et += einsum('ijkabc,ijkabc', w.conj(), r)

    # bba
    w  = einsum('ijae,kceb->ijkabc', t2ab, eris_OVVV) * 2
    w += einsum('ijeb,kcea->ijkabc', t2ab, eris_OVvv) * 2
    w += einsum('jkbe,iaec->ijkabc', t2bb, eris_ovVV)
    w -= einsum('imab,kcjm->ijkabc', t2ab, eris_OVOO) * 2
    w -= einsum('mjab,kcim->ijkabc', t2ab, eris_OVoo) * 2
    w -= einsum('jmbc,iakm->ijkabc', t2bb, eris_ovOO)
    r = w - w.transpose(0,2,1,3,4,5)
    r = r + r.transpose(0,2,1,3,5,4)
    v  = einsum('jbkc,ia->ijkabc', eris_OVOV, t1a)
    v += einsum('iakc,jb->ijkabc', eris_ovOV, t1b)
    v += einsum('iakc,jb->ijkabc', eris_ovOV, t1b)
    v += einsum('JKBC,ai->iJKaBC', t2bb, fvo) * .5
    v += einsum('iKaC,BJ->iJKaBC', t2ab, fVO) * 2
    w += v
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eia, eIA, eIA)
    r /= d3

    if project == 'w':
        w = sym_proj(w, 'abb')
    elif project == 'r':
        r = sym_proj(r, 'abb')
    else:
        raise NotImplementedError()
    et += einsum('ijkabc,ijkabc', w.conj(), r)

    et *= .25
    return et