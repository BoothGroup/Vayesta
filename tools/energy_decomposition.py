from pyscf import ao2mo
from vayesta.core.util import *
import numpy as np






def get_energy_decomp(emb, dm1, dm2, bosonic_exchange_coupling=True):
    # Given DMs in AO basis, going to generate local contribution to energy for each cluster.
    # This can then be fragmented into four contributions within the fragmented approach.
    # The contribution from degrees of freedom corresponding to the fermionic cluster, those corresponding to the
    # coupling between the fermionic and bosonic degrees of freedom within the cluster, and the one- and two-body
    # contributions which cannot be represented within the cluster. The latter of these can be approximated via the
    # delta RPA correction, while the former cannot currently be approximated.
    for frag in emb.fragments:

        clus = frag.cluster

        c_act = clus.c_active
        c_act_mo = dot(emb.mo_coeff.T, emb.get_ovlp(), clus.c_active)
        co_fr = clus.c_frozen_occ

        bosons_present = True
        try:
            r_bosa, r_bosb = frag.r_bos_ao
        except AttributeError:
            bosons_present = False

        p_frag = frag.get_fragment_projector(c_act)

        eris = emb.get_eris_array(c_act)

        exact_dm1_loc = dot(c_act_mo.T, dm1, c_act_mo)
        exact_dm2_loc = ao2mo.full(dm2, c_act_mo)

        loc_e1_exact = dot(p_frag, exact_dm1_loc, dot(c_act.T, emb.get_hcore(), c_act)).trace()
        loc_e2_exact = einsum("pt,pqrs,qtsr->", p_frag, eris, exact_dm2_loc)/2

        err_dm1 = frag.results.dm1 - exact_dm1_loc
        err_dm2 = frag.results.dm2 - exact_dm2_loc

        if bosons_present:
            mo_c = dot(emb.get_ovlp(), emb.mo_coeff)

            r_bosa_mo = einsum("npq,pi,qa->nia", r_bosa, mo_c, mo_c)
            r_bosb_mo = einsum("npq,pi,qa->nia", r_bosb, mo_c, mo_c)

            exact_ebdm_loca = einsum("pqrs,pt,qu,nrs->tun", dm2, c_act_mo, c_act_mo, r_bosa_mo)
            exact_ebdm_locb = einsum("pqrs,pt,qu,nrs->tun", dm2, c_act_mo, c_act_mo, r_bosb_mo)

            err_ebdma = frag.results.dm_eb[0] - exact_ebdm_loca
            err_ebdmb = frag.results.dm_eb[1] - exact_ebdm_locb

            # Can calculate equivalent contributions for EDMET straightforwardly.
            e1_contrib, e2_contrib, efb_contrib = frag.get_edmet_energy_contrib()

        else:
            e1_contrib, e2_contrib = frag.get_dmet_energy_contrib()

        loc_e1 = dot(p_frag, frag.results.dm1, dot(c_act.T, emb.get_hcore(), c_act)).trace()
        static_2body = e1_contrib - loc_e1
        loc_e2 = einsum("pt,pqrs,qtsr->", p_frag, eris, frag.results.dm2) / 2.0

        print(abs(exact_dm1_loc - frag.results.dm1).max(), loc_e2 - e2_contrib)

        if bosons_present:
            yield (np.linalg.norm(err_dm1) / np.linalg.norm(exact_dm1_loc),
                   np.linalg.norm(err_dm2)/np.linalg.norm(exact_dm2_loc),
                   exact_ebdm_loca, exact_ebdm_locb,
                   #np.linalg.norm(err_ebdma)/np.linalg.norm(exact_ebdm_loca), np.linalg.norm(err_ebdmb)/np.linalg.norm(exact_ebdm_locb),
                   loc_e1, loc_e2, loc_e1_exact, loc_e2_exact)
        else:
            yield (np.linalg.norm(err_dm1) / np.linalg.norm(exact_dm1_loc),
                   np.linalg.norm(err_dm2) / np.linalg.norm(exact_dm2_loc),
                   loc_e1, loc_e2, loc_e1_exact, loc_e2_exact)


def get_comparison(basis, cardinality, res_file):
    import vayesta.edmet
    import vayesta.dmet
    from pyscf import cc, gto, scf
    from vayesta.misc import molstructs

    mol = gto.Mole()
    mol.atom = molstructs.arene(6)
    mol.basis

    rmf = scf.RHF(mol)
    rdfmf = rmf.density_fit()
    rdfmf.conv_tol=1e-10
    rdfmf.kernel()

    myccsd = cc.CCSD(rdfmf)
    myccsd.kernel()
    dm1 = myccsd.make_rdm1()
    dm2 = myccsd.make_rdm2()

    rdfedmet = vayesta.edmet.EDMET(rdfmf, oneshot=True, make_dd_moments=False, solver = "EBCCSD", dmet_threshold=1e-12, bosonic_interaction="qba_bos_ex"); rdfedmet.iao_fragmentation(); rdfedmet.add_atomic_fragment([0,2,4,6,8,10], orbital_filter=["2pz"])
    rdfedmet.kernel()

    res = next(ed.get_energy_decomp(rdfedmet, dm1, dm2)
    del rdfedmet
    rdfdmet = vayesta.dmet.DMET(rdfmf, oneshot=True, solver = "CCSD", dmet_threshold=1e-12); rdfdmet.iao_fragmentation(); rdfdmet.add_atomic_fragment([0,2,4,6,8,10], orbital_filter=["2pz"])
    rdfdmet.kernel()
    resdmet = next(ed.get_energy_decomp(rdfdmet, dm1, dm2)

    with open(res_file, "a") as f:
        f.write((" {:d}   "+"   {:16.8e}"*10).format(cardinality, res[0], res[1], res[4], res[5], resdmet[0], resdmet[1], resdmet[2], resdmet[3], resdmet[4], resdmet[5])


def run_full_comparison():
    res_file = "dm_comparison.out"
    basis_sets = ["STO-3g", "cc-pvdz", "cc-pvtz", "cc-pvqz"]

    with open(res_file, "a") as f:
        f.write(" #   Cardinality                                   EDMET                          |                DMET                                          |             Exact")
        f.write(" #                     DM1_prop_err      DM2_prop_err    E_onebody     E_twobody  |   DM1_prop_err      DM2_prop_err    E_onebody     E_twobody  |   E_onebody     E_twobody")


    for i, bas in enumerate(basis_sets):
        get_comparison(bas, i+1, res_file)
