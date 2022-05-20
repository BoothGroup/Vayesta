import matplotlib.pyplot as plt

from pyscf import ao2mo
from vayesta.core.util import *
import numpy as np
from scipy.linalg import block_diag



def dump_results(res, filename, method = "emb", write_exact = True):
    def write_cluster_result(clusres, filename, method, write_exact=True):
        clus_id, clusres = clusres[0], clusres[1:]
        fname_exact = filename + "_exact"
        fname_method = filename + "_" + method

        def get_result_string(id, ob, tb_c, tb_a):
            return ("  {:4d}    {:10s}" + "   {:12.6e}" * len(ob) + "      "
                    + "   {:12.6e}" * len(tb_c) + "      " + "   {:12.6e}" * len(tb_a)
                    +"\n").format(*id, *ob, *tb_c, *tb_a)
        if write_exact:
            with open(fname_exact, "a") as f:
                f.write(get_result_string(clus_id, clusres[0][0], clusres[0][1][1:], clusres[0][2][1:]))
                #print(f"For cluster {clus_id[1]} total correlation energies are {sum(clusres[0][0]) + clusres[0][1][0]}"
                #      f"and {sum(clusres[0][0]) + clusres[0][2][0]}")
        with open(fname_method, "a") as f:
            f.write(get_result_string(clus_id, *clusres[1]))

    fname_exact = filename + "_exact"
    fname_method = filename + "_" + method

    header_string = [("#   frag_id      frag_name   " + "{:^27s}       {:^60s}       {:^60s}\n").format("onebody", "twobody_coulomb", "twobody_antisym"),
        ("# " + " "* len("  frag_id      frag_name   ") + "   {:<12s}" * 2 + "      " + "   {:<12s}" * 4 + "      " + "   {:<12s}" * 4 + "\n").format(
        "loc", "nl",
        "loc", "nl_a", "nl_b", "nl_c",
        "loc", "nl_a", "nl_b", "nl_c"
    )
    ]

    if write_exact:
        with open(fname_exact, "a") as f:
            for x in header_string:
                f.write(x)
    with open(fname_method, "a") as f:
        for x in header_string:
            f.write(x)

    for x in res:
        write_cluster_result(x, filename, method, write_exact)


def get_energy_decomp(emb, dm1, dm2):
    c = emb.mo_coeff
    eris_aa = emb.get_eris_array(c[0])
    eris_ab = emb.get_eris_array((c[0], c[0], c[1], c[1]))
    eris_bb = emb.get_eris_array(c[1])
    eris = [eris_aa, eris_ab, eris_bb]

    nao = c.shape[1]

    hcore = tuple([dot(c[x].T, emb.mf.get_hcore(), c[x]) for x in [0, 1]])

    res_exact = []
    res_emb = []

    sc = [dot(emb.get_ovlp(), x) for x in emb.mo_coeff]
    # Difference from HF rdms.
    mf_dm1 = np.array([dot(x.T, y, x) for x, y in zip(sc, emb.mf.make_rdm1())])
    #print(mf_dm1)
    dm1 = [x - y for (x,y) in zip(dm1, mf_dm1)]
    #print(mf_dm1)
    #print(dm1)
    dm2 = list(dm2)
    dm2[0] = dm2[0] - einsum("pq,rs->pqrs", mf_dm1[0], mf_dm1[0]) + einsum("pq,rs->psrq", mf_dm1[0], mf_dm1[0])
    dm2[2] = dm2[2] - einsum("pq,rs->pqrs", mf_dm1[1], mf_dm1[1]) + einsum("pq,rs->psrq", mf_dm1[1], mf_dm1[1])
    dm2[1] = dm2[1] - einsum("pq,rs->pqrs", mf_dm1[0], mf_dm1[1])

    #print(abs(einsum("npprr->n", np.array(dm2))).max(),"!")

    for f in emb.fragments:

        p_frag = f.get_fragment_projector(c)

        p_act = f.get_fragment_projector(c, f.cluster.c_active)
        #print("Testing projectors:")
        #print(np.linalg.eigvalsh(p_frag))
        #print(np.linalg.eigvalsh(p_act))
        try:
            r_bosa, r_bosb = f.get_rbos_split()
        except AttributeError:
            noa, nob = emb.nocc
            nva, nvb = emb.nvir

            r_bosa = np.zeros((0, noa, nva))
            r_bosb = np.zeros((0, nob, nvb))

        def get_pinv(r):
            nbos = r.shape[0]
            if nbos == 0:
                return np.zeros_like(r)
            r2 = r.reshape((nbos, -1))
            return np.linalg.pinv(r2).reshape((r.shape[1], r.shape[2], nbos)).transpose(2,0,1)

        p_bos = [einsum("nia,njb->iajb", get_pinv(r_bosa), r_bosa), einsum("nia,njb->iajb", get_pinv(r_bosa), r_bosb),
                 einsum("nia,njb->iajb", get_pinv(r_bosb), r_bosb)]

        def map_ov_to_full(mat):
            no1, nv1, no2, nv2 = mat.shape
            n1, n2 = no1+nv1, no2+nv2
            res = np.zeros((n1, n1, n2, n2))
            res[:no1, no1:, :no2, no2:] = mat
            res[no1:, :no1, no2:, :no2] = mat.transpose(1,0,3,2)
            #eigs = np.linalg.eigvals(res.reshape((n1**2, n2**2)))
            res = res.transpose(0,1,3,2)
            #print(abs(eigs.imag).max())
            #print("^^^^", sorted(eigs.real))
            return res#.transpose(0,1,3,2)
        p_bos = tuple([map_ov_to_full(x) for x in p_bos])

        e1_exact, e2_c_exact, e2_as_exact = get_energy_decomp_exact(np.eye(nao), hcore, eris, dm1, dm2,
                                                                    p_frag, p_act, p_bos)
        res_exact += [e1_exact, e2_c_exact, e2_as_exact]

        e1_emb, e2_c_emb, e2_as_emb = get_energy_decomp_emb(f, eris)
        res_emb += [e1_emb, e2_c_emb, e2_as_emb]
        yield ((f.id, f.id_name), (e1_exact, e2_c_exact, e2_as_exact), (e1_emb, e2_c_emb, e2_as_emb))

    # return res_exact, res_emb


def get_energy_decomp_exact(ovlp, hcore, eri, dm1, dm2, p_frag, p_act, p_bos):
    """Note that we require the correlation-induced change in the rdms here, not the actual rdms.
    """

    p_nl = ovlp - p_act
    #print(np.linalg.eigvalsh(p_act))
    #print("!",p_nl)

    #print("Eonebody test:", sum([dot(x, y).trace() for x,y in zip(dm1, hcore)]))

    e1_tot = sum([dot(x, y, z).trace() for x,y,z in zip(p_frag, dm1, hcore)])

    e1_loc = sum([dot(p_frag[x], dm1[x], p_act[x], hcore[x]).trace() for x in [0,1]])

    e1_nl = e1_tot - e1_loc

    e2_coulomb = get_twobody(dm2, p_frag, p_act, p_bos, eri, p_nl, False)
    e2_antisym = get_twobody(dm2, p_frag, p_act, p_bos, eri, p_nl, True)

    return (e1_loc, e1_nl), e2_coulomb, e2_antisym


def get_twobody(dm2, p_frag, p_act, p_bos, eri, p_nl, antisym=False):
    fac = 2.0
    if antisym:
        eri = [x.copy() for x in eri]
        # Deduct exchange contributions where appropriate.
        eri[0] = eri[0] - eri[0].transpose(0, 3, 2, 1)
        eri[2] = eri[2] - eri[2].transpose(0, 3, 2, 1)

        #fac = 4.0
    #print(np.linalg.eigvalsh(p_frag))
    #print(np.linalg.eigvalsh(p_act))
    #print("Etwobody test:",
    #      sum([einsum("pqrs,pqsr->", x, y) for x,y in zip([*eri, eri[1].transpose(2,3,0,1)], [*dm2, dm2[1].transpose(2,3,0,1)])])
    #      )


    # Total twobody energy.
    fident = [np.eye(p_frag[x].shape[0]) for x in range(2)]
    bident = [einsum("pq,rs->prqs", x, y) for (x, y) in [(fident[0], fident[0]), (fident[0], fident[1]),
                                                        (fident[1], fident[1])]]

    b_act = [einsum("pq,rs->prqs", x, y) for (x, y) in [(p_act[0], p_act[0]), (p_act[0], p_act[1]),
                                                        (p_act[1], p_act[1])]]

    def get_twobody_contrib(dm2, eri, p1, p2=None, verbose = False):
        if p2 is None:
            p2 = [np.eye(x.shape[0]) for x in p1]
        if verbose:
            print("1")
            print(p1)
            print("2")
            print(p2)

        val = (einsum("pt,qu,pqsr,tuvw,rv,sw->", p1[0], p2[0], dm2[0], eri[0], p2[0], p2[0]) +  # aa
               einsum("pt,qu,pqsr,tuvw,rv,sw->", p1[1], p2[1], dm2[2], eri[2], p2[1], p2[1]) +  # bb
               einsum("pt,qu,pqsr,tuvw,rv,sw->", p1[0], p2[0], dm2[1], eri[1], p2[1], p2[1]) +  # ab
               einsum("pt,qu,pqsr,tuvw,rv,sw->", p2[0], p2[0], dm2[1], eri[1], p1[1], p2[1])  # ba
               )
        return val

    def get_twobody_contrib_bos(dm2, eri, p1, p2=None, p3=None):
        if p2 is None:
            p2 = [np.eye(p1[x].shape[0]) for x in range(2)]

        if p3 is None:
            p3 = bident

        val = (einsum("pt,qu,pqsr,tuvw,rsvw->", p1[0], p2[0], dm2[0], eri[0], p3[0]) +  # aa
               einsum("pt,qu,pqsr,tuvw,rsvw->", p1[1], p2[1], dm2[2], eri[2], p3[2]) +  # bb
               einsum("pt,qu,pqsr,tuvw,rsvw->", p1[0], p2[0], dm2[1], eri[1], p3[2]) +  # ab
               einsum("pqtu,pqsr,tuvw,rv,sw->", p3[0], dm2[1], eri[1], p1[1], p2[1])  # ba
               )
        return val

    e2_tot = get_twobody_contrib(dm2, eri, p_frag, verbose=False) / fac

    e2_loc = get_twobody_contrib(dm2, eri, p_frag, p_act) / fac

    #print(
    #    "!!!",
    #    e2_tot,
    #    get_twobody_contrib_bos(dm2, eri, p_frag) / fac,
    #    get_twobody_contrib_bos(dm2, eri, p_frag, p2=p_act) / fac +
    #        get_twobody_contrib_bos(dm2, eri, p_frag, p2=p_nl) / fac,
    #    get_twobody_contrib_bos(dm2, eri, p_frag, p2=p_nl) / fac +
    #    get_twobody_contrib_bos(dm2, eri, p_frag, p2=p_act, p3=p_bos) / fac +
    #    get_twobody_contrib_bos(dm2, eri, p_frag, p_act, [x - y for (x, y) in zip(bident, p_bos)]) / fac
    #)
    #print(
    #    get_twobody_contrib(dm2, eri, p_frag, p_act) / fac -
    #        get_twobody_contrib_bos(dm2, eri, p_frag, p2=p_act, p3=b_act) / fac,
    #    get_twobody_contrib_bos(dm2, eri, p_frag, p2=p_nl) / fac,
    #    get_twobody_contrib_bos(dm2, eri, p_frag, p2=p_act, p3=p_bos) / fac,
    #    get_twobody_contrib_bos(dm2, eri, p_frag, p2=p_act, p3=[x - y for (x, y) in zip(bident, p_bos)]) / fac,
    #)

    assert(abs(get_twobody_contrib(dm2, eri, p_frag, p_act) / fac -
            get_twobody_contrib_bos(dm2, eri, p_frag, p2=p_act, p3=b_act) / fac) < 1e-8)

    e2_nl_a = get_twobody_contrib_bos(dm2, eri, p_frag, p_nl, bident) / fac

    e2_nl_b = get_twobody_contrib_bos(dm2, eri, p_frag, p_act, p_bos) / fac


    e2_nl_c = get_twobody_contrib_bos(dm2, eri, p_frag, p_act, [x - y - z for (x,y,z) in zip(bident, p_bos, b_act)])/fac

    #print(e2_tot, e2_loc, e2_nl_a, e2_nl_b, e2_nl_c)

    return e2_tot, e2_loc, e2_nl_a, e2_nl_b, e2_nl_c


def get_energy_decomp_emb(frag, eris=None):
    sc = [dot(frag.base.get_ovlp(), x) for x in frag.cluster.c_active]
    # Difference from HF rdms.
    mf_dm1 = np.array([dot(x.T, y, x) for x, y in zip(sc, frag.mf.make_rdm1())])
    # dms in active orbitals
    dm1 = np.array(frag.results.dm1) - mf_dm1
    dm2 = np.array(frag.results.dm2)
    # Deduct different spin components from dms.
    dm2[0] = dm2[0] - einsum("pq,rs->pqrs", mf_dm1[0], mf_dm1[0]) + einsum("pq,rs->psrq", mf_dm1[0], mf_dm1[0])
    dm2[1] = dm2[1] - einsum("pq,rs->pqrs", mf_dm1[0], mf_dm1[1])
    dm2[2] = dm2[2] - einsum("pq,rs->pqrs", mf_dm1[1], mf_dm1[1]) + einsum("pq,rs->psrq", mf_dm1[1], mf_dm1[1])

    c_act = frag.cluster.c_active
    # onebody hamiltonian in active orbitals.
    hcore_loc = np.array([dot(x.T, frag.base.get_hcore(), x) for x in c_act])
    # fragment projector in active orbitals.
    p_frag = np.array(frag.get_fragment_projector(c_act))
    if eris is None:
        c_full = frag.base.mo_coeff
        eris_aa = frag.base.get_eris_array(c_full[0])
        eris_ab = frag.base.get_eris_array((c_full[0], c_full[0], c_full[1], c_full[1]))
        eris_bb = frag.base.get_eris_array(c_full[1])
        eris = (eris_aa, eris_ab, eris_bb)

    # Can compute this in the cluster space easily.
    e1_loc = einsum("npq,nqr,nrp->", p_frag, dm1, hcore_loc)
    # Zero by construction.
    e1_nl = 0.0

    e2_coulomb = get_twobody_emb(frag, p_frag, eris, dm2, False)
    e2_antisym = get_twobody_emb(frag, p_frag, eris, dm2, True)

    return (e1_loc, e1_nl), e2_coulomb, e2_antisym


def get_twobody_emb(frag, p_frag, eri, dm2_loc, antisym=False):
    c_act = frag.cluster.c_active
    eris_aa = frag.base.get_eris_array(c_act[0])
    eris_ab = frag.base.get_eris_array((c_act[0], c_act[0], c_act[1], c_act[1]))
    eris_bb = frag.base.get_eris_array(c_act[1])
    eri_loc = np.array([eris_aa, eris_ab, eris_bb])

    fac = 2.0
    eri = list(eri)
    if antisym:

        # Deduct exchange contributions where appropriate.
        eri[0] = eri[0] - eri[0].transpose(0, 3, 2, 1)
        eri[2] = eri[2] - eri[2].transpose(0, 3, 2, 1)

        eri_loc[0] = eri_loc[0] - eri_loc[0].transpose(0, 3, 2, 1)
        eri_loc[2] = eri_loc[2] - eri_loc[2].transpose(0, 3, 2, 1)

        #fac = 4.0

    # Local correlated 2rdm.
    e2_loc = (einsum("pt,pqsr,tqrs->", p_frag[0], dm2_loc[0], eri_loc[0]) +
              einsum("pt,pqsr,tqrs->", p_frag[0], dm2_loc[1], eri_loc[1]) +
              einsum("rt,qprs,pqts->", p_frag[1], dm2_loc[1], eri_loc[1]) +
              einsum("pt,pqsr,tqrs->", p_frag[1], dm2_loc[2], eri_loc[2])
              ) / fac

    # Approximate coupling of nonlocal excitations to the rest of the space at mean-field level.
    e2_nl_a = 0.0

    try:
        r_bosa, r_bosb = frag.get_rbos_split()
    except AttributeError:
        noa, nob = frag.base.nocc
        nva, nvb = frag.base.nvir

        na, nb = frag.cluster.norb_active

        r_bosa = np.zeros((0, noa, nva))
        r_bosb = np.zeros((0, nob, nvb))

        dm_eb = np.array([np.zeros((na, na, 0)), np.zeros((nb, nb, 0))])

    else:
        # This is in the active cluster basis.
        dm_eb = np.array(frag.results.dm_eb)

    # Bosonic coupling treats couplings of local excitations to the rest of the space, which would
    # otherwise be neglected.
    # Have contributions from all possible combos of spin pairs. Note that since bosons only contain same-spin
    # excitations don't have to worry about spin-flip contributions.
    # First, bosonic excitation couplings of all possible spin pairings.
    noa, nob = frag.base.nocc

    r = frag.get_overlap_m2c()

    ra = block_diag(r[0][0], r[1][0])
    rb = block_diag(r[0][1], r[1][1])

    # Can generate bosonic couplings first if want...


    #print("&&&&",
    #      np.linalg.svd(
    #          np.concatenate(
    #              [r_bosa.reshape((r_bosa.shape[0], -1)), r_bosb.reshape((r_bosb.shape[0], -1))]
    #              , axis=1
    #          )
    #      )[1])

    # Coupling from bosonic excitations to alpha spins
    va = einsum("nia,pqia->pqn", r_bosa, eri[0][:, :, :noa, noa:]) + \
          einsum("nia,pqia->pqn", r_bosb, eri[1][:, :, :nob, nob:])
    vb = einsum("nia,pqia->pqn", r_bosb, eri[2][:, :, :nob, nob:]) + \
          einsum("nia,iapq->pqn", r_bosa, eri[1][:noa, noa:, :, :])

    v = [einsum("pqn,pr,qs->rsn", x, y, y) for x,y in zip((va, vb), (ra, rb))]

    #print(einsum("pqn->pq", sum(dm_eb)))
    print(einsum("pqn->pq", sum(v)))
    #print(einsum("npq->pq", sum(frag.couplings)))

    #print(np.linalg.eigvals(p_frag[0]))
    #print(np.linalg.eigvals(p_frag[1]))
    #print(p_frag[0])
    #print(p_frag[1])
    print(einsum("pqn->",v[0]),einsum("pqn->", dm_eb[0]),einsum("pqn->",v[1]),einsum("pqn->", dm_eb[1]),
          einsum("pqn,pqn->", v[0], dm_eb[0]), einsum("pqn,pqn->",v[1], dm_eb[1]))

    #print(abs(v[0] - frag.couplings[0].transpose(1,2,0)).max(), abs(v[1] - frag.couplings[1].transpose(1,2,0)).max())
    #v = [x.transpose(1,2,0) for x in frag.couplings]
    e2_nl_b2 = (einsum("pr,pqn,rqn", p_frag[0], dm_eb[0], v[0]) +
                einsum("pr,pqn,rqn", p_frag[1], dm_eb[1], v[1])) / fac

    e2_nl_b2 += (einsum("pr,qpn,qrn", p_frag[0], dm_eb[0], v[0]) +
                 einsum("pr,qpn,qrn", p_frag[1], dm_eb[1], v[1])) / fac
    # excitation portion first.
    e2_nl_b = (einsum("tp,pqn,nrs,tqrs", p_frag[0], dm_eb[0], r_bosa,
                      einsum("qprs,pt,qu->utrs", eri[0][:, :, :noa, noa:], ra, ra)) +  # aa
               einsum("tp,pqn,nrs,tqrs", p_frag[0], dm_eb[0], r_bosb,
                      einsum("qprs,pt,qu->utrs", eri[1][:, :, :nob, nob:], ra, ra)) +  # ab
               einsum("tp,pqn,nrs,tqrs", p_frag[1], dm_eb[1], r_bosa,
                      einsum("qprs,pt,qu->utrs", eri[2][:, :, :nob, nob:], rb, rb)) +  # bb
               einsum("tp,pqn,nrs,tqrs", p_frag[1], dm_eb[1], r_bosa,
                      einsum("rsqp,pt,qu->utrs", eri[1][:noa, noa:, :, :], rb, rb))  # ba
               ) / fac
    # Now dexcitation portion; just a swap of the indices on the electron-boson dm.
    e2_nl_b += (einsum("tp,qpn,nrs,tqrs", p_frag[0], dm_eb[0], r_bosa,
                      einsum("qprs,pt,qu->turs", eri[0][:, :, :noa, noa:], ra, ra)) +  # aa
               einsum("tp,qpn,nrs,tqrs", p_frag[0], dm_eb[0], r_bosb,
                      einsum("qprs,pt,qu->turs", eri[1][:, :, :nob, nob:], ra, ra)) +  # ab
               einsum("tp,qpn,nrs,tqrs", p_frag[1], dm_eb[1], r_bosa,
                      einsum("qprs,pt,qu->turs", eri[2][:, :, :nob, nob:], rb, rb)) +  # bb
               einsum("tp,qpn,nrs,tqrs", p_frag[1], dm_eb[1], r_bosa,
                      einsum("rsqp,pt,qu->turs", eri[1][:noa, noa:, :, :], rb, rb))  # ba
               ) / fac
    print("%%", e2_nl_b, e2_nl_b2, e2_nl_b - e2_nl_b2)

    # Coupling between local excitations and non-bosonic interactions.
    # In this case this is just the mean-field contribution.
    e2_nl_c = 0.0

    return e2_loc, e2_nl_a, e2_nl_b2, e2_nl_c


def get_comparison(basis, cardinality, res_file):
    import vayesta.edmet
    import vayesta.dmet
    from pyscf import cc, gto, scf
    from pyscf.tools import ring

    mol = gto.Mole()
    mol.atom = [('H %f %f %f' % xyz) for xyz in ring.make(10, 1.0)]
    mol.basis = basis

    rmf = scf.RHF(mol)
    rdfmf = rmf.density_fit()
    rdfmf.conv_tol = 1e-10
    rdfmf.kernel()

    myccsd = cc.CCSD(rdfmf)
    myccsd.kernel()
    mf_dm1 = rdfmf.make_rdm1()
    dm1 = myccsd.make_rdm1(ao_repr=True) - mf_dm1
    dm2 = myccsd.make_rdm2(ao_repr=True) - einsum("pq,rs->pqrs", mf_dm1, mf_dm1) + einsum("pq,rs->psrq", mf_dm1, mf_dm1)

    # Now need to convert to
    rdfedmet = vayesta.edmet.EDMET(rdfmf, oneshot=True, make_dd_moments=False, solver="EBCCSD", dmet_threshold=1e-12,
                                   bosonic_interaction="qba_bos_ex")
    rdfedmet.iao_fragmentation()
    rdfedmet.add_atomic_fragment([0])
    rdfedmet.kernel()

    res = next(get_energy_decomp(rdfedmet, dm1, dm2))
    del rdfedmet
    rdfdmet = vayesta.dmet.DMET(rdfmf, oneshot=True, solver="CCSD", dmet_threshold=1e-12)
    rdfdmet.iao_fragmentation();
    rdfdmet.add_atomic_fragment([0])
    rdfdmet.kernel()
    resdmet = next(get_energy_decomp(rdfdmet, dm1, dm2))

    with open(res_file, "a") as f:
        f.write((" {:d}   " + "   {:16.8e}" * 10).format(cardinality, res[0], res[1], res[4], res[5], resdmet[0],
                                                         resdmet[1], resdmet[2], resdmet[3], resdmet[4], resdmet[5]))


def run_full_comparison():
    res_file = "dm_comparison.out"
    basis_sets = ["STO-3g", "cc-pvdz", "cc-pvtz", "cc-pvqz"]

    with open(res_file, "a") as f:
        f.write(
            " #   Cardinality                                   EDMET                                                  DMET                                                            Exact")
        f.write(
            " #                     DM1_prop_err      DM2_prop_err    E_onebody     E_twobody            DM1_prop_err      DM2_prop_err    E_onebody     E_twobody           E_onebody     E_twobody")

    for i, bas in enumerate(basis_sets):
        get_comparison(bas, i + 1, res_file)


def plot_decomposition(files, labels, clusind=0, only_edmet_contribs=False, antisym=True):

    res = [np.genfromtxt(x)[clusind] for x in files]
    aoff = 4 * int(antisym)

    inds = [2,3] + list(range(4+aoff, 8+aoff))
    dat = [x[inds] for x in res]

    if only_edmet_contribs:
        npoints = 4
        def sum_unwanted(x):
            resx = np.zeros(4)
            resx[0] = x[0]
            resx[1] = x[2]
            resx[2] = x[4]
            resx[3] = x[1] + x[3] + x[5]
            return resx
        dat = [sum_unwanted(x) for x in dat]
    else:
        npoints = 6

    nfiles = len(files)

    # Want total width to be 120, arbitrarily, so width of each energy category is given by
    w = 120 / npoints
    rawx = [w * i for i in range(npoints)]
    # Now need offset between different file datapoints; arbitrarily set spacing between
    # different points to 0.2 of their total width.
    space = 0.8 * w / nfiles

    # Need centre of each set of bars for labelling.
    centres = [x + (space * nfiles / 2) for x in rawx]
    # Now just to plot.

    for i, (v, lab) in enumerate(zip(dat, labels)):
        x = [y+i*space for y in rawx]
        print(x)
        print(v)
        plt.bar(x=x, height=v, label=lab, width=space)

    ax = plt.gca()
    ax.set_xticks(centres)
    if only_edmet_contribs:
        ax.set_xticklabels([
            "$E1_\\textrm{loc}$", "$E2_\\textrm{loc}$", "$E2_\\textrm{b}$", "Neglected"
        ])
    else:
        ax.set_xticklabels([
            "$E1_\\textrm{loc}$", "$E1_\\textrm{nl}$", "$E2_\\textrm{loc}$", "$E2_\\textrm{b}$", "$E2_\\text{c}$"
        ])

    plt.show(block=False)
