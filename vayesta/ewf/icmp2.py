import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.tools

from vayesta.core.util import *
from vayesta.mpi import mpi
from vayesta.mpi import RMA_Dict

class ClusterRHF:
    """Helper class"""

    def __init__(self, fragment, coll):
        # The following attributes can be used from the symmetry parent, without modification:
        f0id = fragment.get_symmetry_parent().id
        self.p_frag = coll[f0id, 'p_frag']
        self.e_occ = coll[f0id, 'e_occ']
        self.e_vir = coll[f0id, 'e_vir']
        # These attributes are different for every fragment:
        fid = fragment.id
        self.c_vir = coll[fid, 'c_vir']
        self.cderi = coll[fid, 'cderi']
        if (fid, 'cderi_neg') in coll:
            self.cderi_neg = coll[fid, 'cderi_neg']
        else:
            self.cderi_neg = None

class ClusterUHF:
    """Helper class"""

    def __init__(self, fragment, coll):
        # From symmetry parent:
        f0id = fragment.get_symmetry_parent().id
        self.p_frag = (coll[f0id, 'p_frag_a'], coll[f0id, 'p_frag_b'])
        self.e_occ = (coll[f0id, 'e_occ_a'], coll[f0id, 'e_occ_b'])
        self.e_vir = (coll[f0id, 'e_vir_a'], coll[f0id, 'e_vir_b'])
        # Own:
        fid = fragment.id
        self.c_vir = (coll[fid, 'c_vir_a'], coll[fid, 'c_vir_b'])
        self.cderi = (coll[fid, 'cderi_a'], coll[fid, 'cderi_b'])
        if (fid, 'cderi_a_neg') in coll:
            self.cderi_neg = (coll[fid, 'cderi_a_neg'], coll[fid, 'cderi_b_neg'])
        else:
            self.cderi_neg = None


def _get_icmp2_fragments(emb, **kwargs):
        return emb.get_fragments(active=True, options=dict(icmp2_active=True), **kwargs)


def get_intercluster_mp2_energy_rhf(emb, bno_threshold_occ=None, bno_threshold_vir=1e-9,
        direct=True, exchange=True, fragments=None, project_dc='occ', vers=1, diagonal=True):
    """Get long-range, inter-cluster energy contribution on the MP2 level.

    This constructs T2 amplitudes over two clusters, X and Y, as

    .. math::

        t_ij^ab = \sum_L (ia|L)(L|j'b') / (ei + ej' - ea - eb)

    where i,a are in cluster X and j,b are in cluster Y.

    Parameters
    ----------
    bno_threshold: float, optional
        Threshold for BNO space. Default: 1e-9.
    bno_threshold_occ: float, optional
        Threshold for occupied BNO space. Default: None.
    bno_threshold_vir: float, optional
        Threshold for virtual BNO space. Default: None.
    direct: bool, optional
        Calculate energy contribution from the second-order direct MP2 term. Default: True.
    exchange: bool, optional
        Calculate energy contribution from the second-order exchange MP2 term. Default: True.

    Returns
    -------
    e_icmp2: float
        Intercluster MP2 energy contribution.
    """

    if project_dc not in ('occ', 'vir', 'both', None):
        raise ValueError()
    if not emb.has_df:
        raise RuntimeError("Intercluster MP2 energy requires density-fitting.")

    e_direct = e_exchange = 0.0
    with log_time(emb.log.timing, "Time for intercluster MP2 energy: %s"):
        ovlp = emb.get_ovlp()
        if emb.kdf is not None:
            # We need the supercell auxiliary cell here:
            cellmesh = emb.mf.subcellmesh
            auxmol = pyscf.pbc.tools.super_cell(emb.kdf.auxcell, cellmesh)
        else:
            try:
                auxmol = emb.df.auxmol
            except AttributeError:
                auxmol = emb.df.auxcell
        auxsym = type(emb.symmetry)(auxmol)

        with log_time(emb.log.timing, "Time for intercluster MP2 energy setup: %s"):
            coll = {}
            # Loop over symmetry unique fragments X
            for x in _get_icmp2_fragments(emb, mpi_rank=mpi.rank, sym_parent=None):
                # Occupied orbitals:
                if bno_threshold_occ is None:
                    c_occ = x._dmet_bath.c_cluster_occ
                else:
                    if x._bath_factory_occ is not None:
                        bath = x._bath_factory_occ
                        c_bath_occ = bath.get_bath(bno_threshold=bno_threshold_occ, verbose=False)[0]
                        c_occ = x.canonicalize_mo(bath.c_cluster_occ, c_bath_occ)[0]
                    else:
                        c_bath_occ = x.bath.get_occupied_bath(bno_threshold=bno_threshold_occ, verbose=False)[0]
                        c_occ = x.canonicalize_mo(x.bath.c_cluster_occ, c_bath_occ)[0]
                # Virtual orbitals:
                if bno_threshold_vir is None:
                    c_vir = x._dmet_bath.c_cluster_vir
                else:
                    if x._bath_factory_vir is not None:
                        bath = x._bath_factory_vir
                        c_bath_vir = bath.get_bath(bno_threshold=bno_threshold_vir, verbose=False)[0]
                        c_vir = x.canonicalize_mo(bath.c_cluster_vir, c_bath_vir)[0]
                    else:
                        c_bath_vir = x.bath.get_virtual_bath(bno_threshold=bno_threshold_vir, verbose=False)[0]
                        c_vir = x.canonicalize_mo(x.bath.c_cluster_vir, c_bath_vir)[0]
                # Three-center integrals:
                cderi, cderi_neg = emb.get_cderi((c_occ, c_vir))   # TODO: Reuse BNO
                # Store required quantities:
                coll[x.id, 'p_frag'] = dot(x.c_proj.T, ovlp, c_occ)
                coll[x.id, 'c_vir'] = c_vir
                coll[x.id, 'e_occ'] = x.get_fragment_mo_energy(c_occ)
                coll[x.id, 'e_vir'] = x.get_fragment_mo_energy(c_vir)
                coll[x.id, 'cderi'] = cderi
                # TODO: Test 2D
                if cderi_neg is not None:
                    coll[x.id, 'cderi_neg'] = cderi_neg
                # Fragments Y, which are symmetry related to X
                for y in _get_icmp2_fragments(emb, sym_parent=x):
                    sym_op = y.get_symmetry_operation()
                    coll[y.id, 'c_vir'] = sym_op(c_vir)
                    # TODO: Why do we need to invert the atom reordering with argsort?
                    sym_op_aux = type(sym_op)(auxsym, vector=sym_op.vector, atom_reorder=np.argsort(sym_op.atom_reorder))
                    coll[y.id, 'cderi'] = sym_op_aux(cderi)
                    #cderi_y2 = emb.get_cderi((sym_op(c_occ), sym_op(c_vir)))[0]
                    if cderi_neg is not None:
                        coll[y.id, 'cderi_neg'] = cderi_neg
            # Convert into remote memory access (RMA) dictionary:
            if mpi:
                coll = mpi.create_rma_dict(coll)

        for ix, x in enumerate(_get_icmp2_fragments(emb, fragments=fragments, mpi_rank=mpi.rank, sym_parent=None)):
            cx = ClusterRHF(x, coll)

            eia_x = cx.e_occ[:,None] - cx.e_vir[None,:]

            # Already contract these parts of P_dc and S_vir, to avoid having the n(AO)^2 overlap matrix in the n(Frag)^2 loop:
            if project_dc in ('occ', 'both'):
                pdco0 = np.dot(x.cluster.c_active.T, ovlp)
            if project_dc in ('vir', 'both'):
                pdcv0 = np.dot(x.cluster.c_active_vir.T, ovlp)
            #if exchange:
            svir0 = np.dot(cx.c_vir.T, ovlp)

            # Loop over all other fragments
            for iy, y in enumerate(_get_icmp2_fragments(emb)):
                cy = ClusterRHF(y, coll)

                # TESTING
                if diagonal == 'only' and x.id != y.id:
                    continue
                if not diagonal and x.id == y.id:
                    continue

                eia_y = cy.e_occ[:,None] - cy.e_vir[None,:]

                # Make T2
                # TODO: save memory by blocked loop
                # OR: write C function (also useful for BNO build)
                eris = einsum('Lia,Ljb->ijab', cx.cderi, cy.cderi) # O(n(frag)^2) * O(naux)
                if cx.cderi_neg is not None:
                    eris -= einsum('Lia,Ljb->ijab', cx.cderi_neg, cy.cderi_neg)
                eijab = (eia_x[:,None,:,None] + eia_y[None,:,None,:])
                t2 = (eris / eijab)
                # Project i onto F(x) and j onto F(y):
                t2 = einsum('xi,yj,ijab->xyab', cx.p_frag, cy.p_frag, t2)
                eris = einsum('xi,yj,ijab->xyab', cx.p_frag, cy.p_frag, eris)

                #if exchange:
                #    # Overlap of virtual space between X and Y
                svir = np.dot(svir0, cy.c_vir)

                # Projector to remove double counting with intracluster energy
                ed = ex = 0
                if project_dc == 'occ':
                    pdco = np.dot(pdco0, y.c_proj)
                    pdco = (np.eye(pdco.shape[-1]) - dot(pdco.T, pdco))
                    if direct:
                        if vers == 1:
                            ed = 2*einsum('ijab,iJab,jJ->', t2, eris, pdco)
                        elif vers == 2:
                            ed = 2*einsum('ijab,iJAb,jJ,aC,AC->', t2, eris, pdco, svir, svir)
                    if exchange:
                        ex = -einsum('ijaB,iJbA,jJ,aA,bB->', t2, eris, pdco, svir, svir)
                elif project_dc == 'vir':
                    pdcv = np.dot(pdcv0, cy.c_vir)
                    pdcv = (np.eye(pdcv.shape[-1]) - dot(pdcv.T, pdcv))
                    if direct:
                        if vers == 1:
                            ed = 2*einsum('ijab,ijaB,bB->', t2, eris, pdcv)
                        elif vers == 2:
                            ed = 2*einsum('ijab,ijAB,bB,aC,AC->', t2, eris, pdcv, svir, svir)
                    if exchange:
                        ex = -einsum('ijaB,ijbC,CA,aA,bB->', t2, eris, pdcv, svir, svir)
                elif project_dc == 'both':
                    pdco = np.dot(pdco0, y.c_proj)
                    pdco = (np.eye(pdco.shape[-1]) - dot(pdco.T, pdco))
                    pdcv = np.dot(pdcv0, cy.c_vir)
                    pdcv = (np.eye(pdcv.shape[-1]) - dot(pdcv.T, pdcv))
                    if direct:
                        ed = 2*einsum('ijab,iJaB,jJ,bB->', t2, eris, pdco, pdcv)
                    if exchange:
                        ex = -einsum('ijaB,iJbC,jJ,CA,aA,bB->', t2, eris, pdco, pdcv, svir, svir)
                elif project_dc is None:
                    if direct:
                        ed = 2*einsum('ijab,ijab->', t2, eris)
                    if exchange:
                        ex = -einsum('ijaB,ijbA,aA,bB->', t2, eris, svir, svir)

                prefac = x.sym_factor * x.symmetry_factor * y.sym_factor
                e_direct += prefac * ed
                e_exchange += prefac * ex

                if ix+iy == 0:
                    emb.log.debugv("Intercluster MP2 energies:")
                xystr = '%s <- %s:' % (x.id_name, y.id_name)
                estr = energy_string
                emb.log.debugv("  %-12s  direct= %s  exchange= %s  total= %s", xystr, estr(ed), estr(ex), estr(ed+ex))

        if mpi:
            e_direct = mpi.world.allreduce(e_direct)
            e_exchange = mpi.world.allreduce(e_exchange)
        e_direct /= emb.ncells
        e_exchange /= emb.ncells
        e_icmp2 = e_direct + e_exchange
        if mpi.is_master:
            emb.log.info("  %-12s  direct= %s  exchange= %s  total= %s", "Total:", estr(e_direct), estr(e_exchange), estr(e_icmp2))
        coll.clear()    # Important in order to not run out of MPI communicators

    return e_icmp2


def get_intercluster_mp2_energy_uhf(emb, bno_threshold=1e-9, direct=True, exchange=True, project_dc='vir'):
    """Get long-range, inter-cluster energy contribution on the MP2 level.

    This constructs T2 amplitudes over two clusters, X and Y, as

    .. math::

        t_ij^ab = \sum_L (ia|L)(L|j'b') / (ei + ej' - ea - eb)

    where i,a are in cluster X and j,b are in cluster Y.

    Parameters
    ----------
    bno_threshold: float, optional
        Threshold for virtual BNO space. Default: 1e-8.
    direct: bool, optional
        Calculate energy contribution from the second-order direct MP2 term. Default: True.
    exchange: bool, optional
        Calculate energy contribution from the second-order exchange MP2 term. Default: True.

    Returns
    -------
    e_icmp2: float
        Intercluster MP2 energy contribution.
    """
    if project_dc != 'vir':
        raise NotImplementedError

    e_direct = 0.0
    e_exchange = 0.0
    with log_time(emb.log.timing, "Time for intercluster MP2 energy: %s"):
        ovlp = emb.get_ovlp()
        if emb.kdf is not None:
            # We need the supercell auxiliary cell here:
            cellmesh = emb.mf.subcellmesh
            auxmol = pyscf.pbc.tools.super_cell(emb.kdf.auxcell, cellmesh)
        else:
            try:
                auxmol = emb.df.auxmol
            except AttributeError:
                auxmol = emb.df.auxcell
        auxsym = type(emb.symmetry)(auxmol)

        with log_time(emb.log.timing, "Time for intercluster MP2 energy setup: %s"):
            coll = {}
            # Loop over symmetry unique fragments
            for x in _get_icmp2_fragments(emb, mpi_rank=mpi.rank, sym_parent=None):
                #c_occ = x.bath.dmet_bath.c_cluster_occ
                c_occ = x._dmet_bath.c_cluster_occ
                coll[x.id, 'p_frag_a'] = dot(x.c_proj[0].T, ovlp, c_occ[0])
                coll[x.id, 'p_frag_b'] = dot(x.c_proj[1].T, ovlp, c_occ[1])
                c_bath_vir = x._bath_factory_vir.get_bath(bno_threshold=bno_threshold, verbose=False)[0]
                c_vir = x.canonicalize_mo(x._dmet_bath.c_cluster_vir, c_bath_vir)[0]
                coll[x.id, 'c_vir_a'], coll[x.id, 'c_vir_b'] = c_vir
                coll[x.id, 'e_occ_a'], coll[x.id, 'e_occ_b'] = x.get_fragment_mo_energy(c_occ)
                coll[x.id, 'e_vir_a'], coll[x.id, 'e_vir_b'] = x.get_fragment_mo_energy(c_vir)
                cderi_a, cderi_a_neg = emb.get_cderi((c_occ[0], c_vir[0]))   # TODO: Reuse BNO
                cderi_b, cderi_b_neg = emb.get_cderi((c_occ[1], c_vir[1]))   # TODO: Reuse BNO
                coll[x.id, 'cderi_a'] = cderi_a
                coll[x.id, 'cderi_b'] = cderi_b
                if cderi_a_neg is not None:
                    coll[x.id, 'cderi_a_neg'] = cderi_a_neg
                    coll[x.id, 'cderi_b_neg'] = cderi_b_neg
                # Symmetry related fragments
                for y in _get_icmp2_fragments(emb, sym_parent=x):
                    sym_op = y.get_symmetry_operation()
                    coll[y.id, 'c_vir_a'] = sym_op(c_vir[0])
                    coll[y.id, 'c_vir_b'] = sym_op(c_vir[1])
                    # TODO: Why do we need to invert the atom reordering with argsort?
                    sym_op_aux = type(sym_op)(auxsym, vector=sym_op.vector, atom_reorder=np.argsort(sym_op.atom_reorder))
                    coll[y.id, 'cderi_a'] = sym_op_aux(cderi_a)
                    coll[y.id, 'cderi_b'] = sym_op_aux(cderi_b)
                    if cderi_a_neg is not None:
                        coll[y.id, 'cderi_a_neg'] = cderi_a_neg
                        coll[y.id, 'cderi_b_neg'] = cderi_b_neg
            # Convert into remote memory access (RMA) dictionary:
            if mpi:
                coll = mpi.create_rma_dict(coll)

        for ix, x in enumerate(_get_icmp2_fragments(emb, mpi_rank=mpi.rank, sym_parent=None)):
            cx = ClusterUHF(x, coll)

            eia_xa = cx.e_occ[0][:,None] - cx.e_vir[0][None,:]
            eia_xb = cx.e_occ[1][:,None] - cx.e_vir[1][None,:]

            # Already contract these parts of P_dc and S_vir, to avoid having the n(AO)^2 overlap matrix in the n(Frag)^2 loop:
            pdcv0a = np.dot(x.cluster.c_active_vir[0].T, ovlp)
            pdcv0b = np.dot(x.cluster.c_active_vir[1].T, ovlp)
            if exchange:
                svir0a = np.dot(cx.c_vir[0].T, ovlp)
                svir0b = np.dot(cx.c_vir[1].T, ovlp)

            # Loop over all other fragments
            for iy, y in enumerate(_get_icmp2_fragments(emb)):
                cy = ClusterUHF(y, coll)

                eia_ya = cy.e_occ[0][:,None] - cy.e_vir[0][None,:]
                eia_yb = cy.e_occ[1][:,None] - cy.e_vir[1][None,:]

                # Make T2
                # TODO: save memory by blocked loop
                # OR: write C function (also useful for BNO build)
                eris_aa = einsum('Lia,Ljb->ijab', cx.cderi[0], cy.cderi[0]) # O(n(frag)^2) * O(naux)
                eris_ab = einsum('Lia,Ljb->ijab', cx.cderi[0], cy.cderi[1]) # O(n(frag)^2) * O(naux)
                eris_ba = einsum('Lia,Ljb->ijab', cx.cderi[1], cy.cderi[0]) # O(n(frag)^2) * O(naux)
                eris_bb = einsum('Lia,Ljb->ijab', cx.cderi[1], cy.cderi[1]) # O(n(frag)^2) * O(naux)
                if cx.cderi_neg is not None:
                    eris_aa -= einsum('Lia,Ljb->ijab', cx.cderi_neg[0], cy.cderi_neg[0])
                    eris_ab -= einsum('Lia,Ljb->ijab', cx.cderi_neg[0], cy.cderi_neg[1])
                    eris_ab -= einsum('Lia,Ljb->ijab', cx.cderi_neg[1], cy.cderi_neg[0])
                    eris_bb -= einsum('Lia,Ljb->ijab', cx.cderi_neg[1], cy.cderi_neg[1])
                # Alpha-alpha
                eijab_aa = (eia_xa[:,None,:,None] + eia_ya[None,:,None,:])
                eijab_ab = (eia_xa[:,None,:,None] + eia_yb[None,:,None,:])
                eijab_ba = (eia_xb[:,None,:,None] + eia_ya[None,:,None,:])
                eijab_bb = (eia_xb[:,None,:,None] + eia_yb[None,:,None,:])
                t2aa = (eris_aa / eijab_aa)
                t2ab = (eris_ab / eijab_ab)
                t2ba = (eris_ba / eijab_ba)
                t2bb = (eris_bb / eijab_bb)

                # Project i onto F(x) and j onto F(y):
                t2aa = einsum('xi,yj,ijab->xyab', cx.p_frag[0], cy.p_frag[0], t2aa)
                t2ab = einsum('xi,yj,ijab->xyab', cx.p_frag[0], cy.p_frag[1], t2ab)
                t2ba = einsum('xi,yj,ijab->xyab', cx.p_frag[1], cy.p_frag[0], t2ba)
                t2bb = einsum('xi,yj,ijab->xyab', cx.p_frag[1], cy.p_frag[1], t2bb)
                eris_aa = einsum('xi,yj,ijab->xyab', cx.p_frag[0], cy.p_frag[0], eris_aa)
                eris_ab = einsum('xi,yj,ijab->xyab', cx.p_frag[0], cy.p_frag[1], eris_ab)
                eris_ba = einsum('xi,yj,ijab->xyab', cx.p_frag[1], cy.p_frag[0], eris_ba)
                eris_bb = einsum('xi,yj,ijab->xyab', cx.p_frag[1], cy.p_frag[1], eris_bb)

                # Overlap of virtual space between X and Y
                if exchange:
                    svira = np.dot(svir0a, cy.c_vir[0])
                    svirb = np.dot(svir0b, cy.c_vir[1])

                # Projector to remove double counting with intracluster energy
                ed = ex = 0
                pdcva = np.dot(pdcv0a, cy.c_vir[0])
                pdcvb = np.dot(pdcv0b, cy.c_vir[1])
                pdcva = (np.eye(pdcva.shape[-1]) - dot(pdcva.T, pdcva))
                pdcvb = (np.eye(pdcvb.shape[-1]) - dot(pdcvb.T, pdcvb))

                if direct:
                    ed += einsum('ijab,ijaB,bB->', t2aa, eris_aa, pdcva)/2
                    ed += einsum('ijab,ijaB,bB->', t2ab, eris_ab, pdcvb)/2
                    ed += einsum('ijab,ijaB,bB->', t2ba, eris_ba, pdcva)/2
                    ed += einsum('ijab,ijaB,bB->', t2bb, eris_bb, pdcvb)/2
                if exchange:
                    ex += -einsum('ijaB,ijbC,CA,aA,bB->', t2aa, eris_aa, pdcva, svira, svira)/2
                    ex += -einsum('ijaB,ijbC,CA,aA,bB->', t2bb, eris_bb, pdcvb, svirb, svirb)/2

                prefac = x.sym_factor * x.symmetry_factor * y.sym_factor
                e_direct += prefac * ed
                e_exchange += prefac * ex

                if ix+iy == 0:
                    emb.log.debugv("Intercluster MP2 energies:")
                xystr = '%s <- %s:' % (x.id_name, y.id_name)
                estr = energy_string
                emb.log.debugv("  %-12s  direct= %s  exchange= %s  total= %s", xystr, estr(ed), estr(ex), estr(ed+ex))

        if mpi:
            e_direct = mpi.world.allreduce(e_direct)
            e_exchange = mpi.world.allreduce(e_exchange)
        e_direct /= emb.ncells
        e_exchange /= emb.ncells
        e_icmp2 = e_direct + e_exchange
        if mpi.is_master:
            emb.log.info("  %-12s  direct= %s  exchange= %s  total= %s", "Total:", estr(e_direct), estr(e_exchange), estr(e_icmp2))
        coll.clear()    # Important in order to not run out of MPI communicators

    return e_icmp2
