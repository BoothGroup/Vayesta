import numpy as np

from vayesta.core import UEmbedding
from vayesta.core.util import *

from vayesta.ewf import REWF
from vayesta.ewf.ufragment import UEWFFragment as Fragment
from vayesta.core.mpi import mpi
from vayesta.core.mpi import RMA_Dict

# Amplitudes
from .amplitudes import get_global_t1_uhf
from .amplitudes import get_global_t2_uhf
# Density-matrices
from .urdm import make_rdm1_ccsd


class UEWF(REWF, UEmbedding):

    Fragment = Fragment

    def get_init_mo_coeff(self, mo_coeff=None):
        """Orthogonalize insufficiently orthogonal MOs.

        (For example as a result of k2gamma conversion with low cell.precision)
        """
        if mo_coeff is None: mo_coeff = self.mo_coeff
        c = mo_coeff.copy()
        ovlp = self.get_ovlp()
        assert np.all(c.imag == 0), "max|Im(C)|= %.2e" % abs(c.imag).max()

        for s, spin in enumerate(('alpha', 'beta')):
            err = abs(dot(c[s].T, ovlp, c[s]) - np.eye(c[s].shape[-1])).max()
            if err > 1e-5:
                self.log.error("Orthogonality error of %s-MOs= %.2e !!!", spin, err)
            else:
                self.log.debug("Orthogonality error of %s-MOs= %.2e", spin, err)
        if self.opts.orthogonal_mo_tol and err > self.opts.orthogonal_mo_tol:
            raise NotImplementedError()
            #t0 = timer()
            #self.log.info("Orthogonalizing orbitals...")
            #c_orth = helper.orthogonalize_mo(c, ovlp)
            #change = abs(einsum('ai,ab,bi->i', c_orth, ovlp, c)-1)
            #self.log.info("Max. orbital change= %.2e%s", change.max(), " (!!!)" if change.max() > 1e-4 else "")
            #self.log.timing("Time for orbital orthogonalization: %s", time_string(timer()-t0))
            #c = c_orth
        return c

    def check_fragment_nelectron(self):
        nelec_frags = (sum([f.sym_factor*f.nelectron[0] for f in self.loop()]),
                       sum([f.sym_factor*f.nelectron[1] for f in self.loop()]))
        self.log.info("Total number of mean-field electrons over all fragments= %.8f , %.8f", *nelec_frags)
        if abs(nelec_frags[0] - np.rint(nelec_frags[0])) > 1e-4 or abs(nelec_frags[1] - np.rint(nelec_frags[1])) > 1e-4:
            self.log.warning("Number of electrons not integer!")
        return nelec_frags

    # --- CC Amplitudes
    # -----------------

    # T-amplitudes
    get_global_t1 = get_global_t1_uhf
    get_global_t2 = get_global_t2_uhf

    def t1_diagnostic(self, warn_tol=0.02):
        # Per cluster
        for f in self.get_fragments(mpi_rank=mpi.rank):
            t1 = f.results.t1
            if t1 is None:
                self.log.error("No T1 amplitudes found for %s.", f)
                continue
            nelec = t1[0].shape[0] + t1[1].shape[0]
            t1diag = (np.linalg.norm(t1[0]) / np.sqrt(nelec),
                      np.linalg.norm(t1[1]) / np.sqrt(nelec))
            if max(t1diag) > warn_tol:
                self.log.warning("T1 diagnostic for %-20s alpha= %.5f beta= %.5f", str(f)+':', *t1diag)
            else:
                self.log.info("T1 diagnostic for %-20s alpha= %.5f beta= %.5f", str(f)+':', *t1diag)
        # Global
        t1 = self.get_global_t1(mpi_target=0)
        if mpi.is_master:
            nelec = t1[0].shape[0] + t1[1].shape[0]
            t1diag = (np.linalg.norm(t1[0]) / np.sqrt(nelec),
                      np.linalg.norm(t1[1]) / np.sqrt(nelec))
            if max(t1diag) > warn_tol:
                self.log.warning("Global T1 diagnostic: alpha= %.5f beta= %.5f", *t1diag)
            else:
                self.log.info("Global T1 diagnostic: alpha= %.5f beta= %.5f", *t1diag)


    # --- Density-matrices
    # --------------------

    make_rdm1_ccsd = make_rdm1_ccsd

    # TODO
    def make_rdm2_ccsd(self, *args, **kwargs):
        raise NotImplementedError()

    def get_intercluster_mp2_energy(self, bno_threshold=1e-9, direct=True, exchange=True):
        """Get long-range, inter-cluster energy contribution on the MP2 level.

        This constructs T2 amplitudes over two clusters, X and Y, as

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

        e_direct = 0.0
        e_exchange = 0.0
        with log_time(self.log.timing, "Time for intercluster MP2 energy: %s"):
            ovlp = self.get_ovlp()

            with log_time(self.log.timing, "Time for intercluster MP2 energy setup: %s"):
                coll = RMA_Dict(mpi)
                # TODO: Does this only allow 2024 / 6 / n(MPI) fragments?
                with coll.writable():
                    for x in self.get_fragments(mpi_rank=mpi.rank):
                        xid = x.id
                        x0 = x.get_symmetry_parent()
                        sym_op = x.get_symmetry_operation()
                        c_occ = x0.bath.dmet_bath.c_cluster_occ
                        coll[xid, 'p_frag_a'] = dot(x0.c_proj[0].T, ovlp, c_occ[0])
                        coll[xid, 'p_frag_b'] = dot(x0.c_proj[1].T, ovlp, c_occ[1])
                        c_bath_vir = x0.bath.get_virtual_bath(bno_threshold=bno_threshold, verbose=False)[0]
                        c_vir = x0.canonicalize_mo(x0.bath.c_cluster_vir, c_bath_vir)[0]
                        coll[xid, 'c_vir_a'] = sym_op(c_vir[0])
                        coll[xid, 'c_vir_b'] = sym_op(c_vir[1])
                        e_occ = x.get_fragment_mo_energy(c_occ)
                        coll[xid, 'e_occ_a'] = e_occ[0]
                        coll[xid, 'e_occ_b'] = e_occ[1]
                        e_vir = x.get_fragment_mo_energy(c_vir)
                        coll[xid, 'e_vir_a'] = e_vir[0]
                        coll[xid, 'e_vir_b'] = e_vir[1]
                        coll[xid, 'cderia'], cderia_neg = self.get_cderi((sym_op(c_occ[0]), sym_op(c_vir[0])))   # TODO: Reuse BNO
                        coll[xid, 'cderib'], cderib_neg = self.get_cderi((sym_op(c_occ[1]), sym_op(c_vir[1])))   # TODO: Reuse BNO
                        if cderia_neg is not None:
                            coll[xid, 'cderia_neg'] = cderia_neg
                            coll[xid, 'cderib_neg'] = cderib_neg

            class Cluster:

                def __init__(self, xid):
                    self.p_frag = (coll[xid, 'p_frag_a'], coll[xid, 'p_frag_b'])
                    self.c_vir = (coll[xid, 'c_vir_a'], coll[xid, 'c_vir_b'])
                    self.e_occ = (coll[xid, 'e_occ_a'], coll[xid, 'e_occ_b'])
                    self.e_vir = (coll[xid, 'e_vir_a'], coll[xid, 'e_vir_b'])
                    self.cderi = (coll[xid, 'cderia'], coll[xid, 'cderib'])
                    if (xid, 'cderi_neg') in coll:
                        self.cderi_neg = (coll[xid, 'cderia_neg'], coll[xid, 'cderib_neg'])
                    else:
                        self.cderi_neg = None

            for ix, x in enumerate(self.get_fragments(mpi_rank=mpi.rank, sym_parent=None)):
                cx = Cluster(x.id)

                eia_xa = cx.e_occ[0][:,None] - cx.e_vir[0][None,:]
                eia_xb = cx.e_occ[1][:,None] - cx.e_vir[1][None,:]

                # Already contract these parts of P_dc and S_vir, to avoid having the n(AO)^2 overlap matrix in the n(Frag)^2 loop:
                pdcv0a = np.dot(x.cluster.c_active_vir[0].T, ovlp)
                pdcv0b = np.dot(x.cluster.c_active_vir[1].T, ovlp)
                if exchange:
                    svir0a = np.dot(cx.c_vir[0].T, ovlp)
                    svir0b = np.dot(cx.c_vir[1].T, ovlp)

                # Loop over all other fragments
                for iy, y in enumerate(self.get_fragments()):
                    cy = Cluster(y.id)

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
                        self.log.debugv("Intercluster MP2 energies:")
                    xystr = '%s <- %s:' % (x.id_name, y.id_name)
                    estr = energy_string
                    self.log.debugv("  %-12s  direct= %s  exchange= %s  total= %s", xystr, estr(ed), estr(ex), estr(ed+ex))

            e_direct = mpi.world.allreduce(e_direct)
            e_exchange = mpi.world.allreduce(e_exchange)
            e_icmp2 = e_direct + e_exchange
            if mpi.is_master:
                self.log.info("  %-12s  direct= %s  exchange= %s  total= %s", "Total:", estr(e_direct), estr(e_exchange), estr(e_icmp2))
            coll.clear()    # Important in order to not run out of MPI communicators

        return e_icmp2
