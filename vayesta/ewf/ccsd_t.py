import numpy as np
import pyscf.lib

def calc_fragment_ccsd_t_energy(fragment, t1, t2, eris=None, project='w'):
    """
    Calculates a fragment CCSD(T) energy contribution.

    Modified expressions obtained from pyscf.cc.ccsd_t_slow, and
    JCP 94, 442 (1991); DOI:10.1063/1.460359.  Error in Eq (1), should be [ia] >= [jb] >= [kc]
    
    """

    einsum = pyscf.lib.einsum

    t1T = t1.T
    t2T = t2.transpose(2,3,0,1)

    nocc, nvir = t1.shape
    mo_e = fragment.hamil.get_clus_mf_info(with_vext=True)[2]
    e_occ, e_vir = mo_e[:nocc], mo_e[nocc:]
    eijk = pyscf.lib.direct_sum('i,j,k->ijk', e_occ, e_occ, e_occ)

    if eris is not None:
        eris_vvov = eris.get_ovvv().conj().transpose(1,3,0,2)
        eris_vooo = np.asarray(eris.ovoo).conj().transpose(1,0,2,3)
        eris_vvoo = np.asarray(eris.ovov).conj().transpose(1,3,0,2)
    else:
        eris_vvov = fragment.hamil.get_eris_bare(block='ovvv').conj().transpose(1,3,0,2)
        eris_vooo = fragment.hamil.get_eris_bare(block='ovoo').conj().transpose(1,0,2,3)
        eris_vvoo = fragment.hamil.get_eris_bare(block='ovov').conj().transpose(1,3,0,2)

    fvo = fragment.hamil.get_fock(with_vext=True)[nocc:,:nocc]
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
