import dataclasses

import numpy as np

import pyscf
import pyscf.ao2mo
import pyscf.ci
import pyscf.cc
import pyscf.mcscf
import pyscf.fci
import pyscf.fci.addons

from vayesta.core.util import *
from vayesta.core.types import Orbitals
from vayesta.core.types import FCI_WaveFunction
from vayesta.core.qemb.scrcoulomb import get_screened_eris_full
from .solver import ClusterSolver


class FCI_Solver(ClusterSolver):

    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
        threads: int = 1            # Number of threads for multi-threaded FCI
        max_cycle: int = 300
        lindep: float = None        # Linear dependency tolerance. If None, use PySCF default
        conv_tol: float = None      # Convergence tolerance. If None, use PySCF default
        solver_spin: bool = True    # Use direct_spin1 if True, or direct_spin0 otherwise
        fix_spin: float = 0.0       # If set to a number, the given S^2 value will be enforced
        #fix_spin_penalty: float = 1.0
        fix_spin_penalty: float = 1e3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        solver_cls = self.get_solver_class()
        solver = solver_cls(self.mol)
        self.log.debugv("type(solver)= %r", type(solver))
        # Set options
        if self.opts.threads is not None: solver.threads = self.opts.threads
        if self.opts.conv_tol is not None: solver.conv_tol = self.opts.conv_tol
        if self.opts.lindep is not None: solver.lindep = self.opts.lindep
        if self.opts.max_cycle is not None: solver.max_cycle = self.opts.max_cycle
        if self.opts.fix_spin not in (None, False):
            spin = self.opts.fix_spin
            self.log.debugv("Fixing spin of FCI solver to S^2= %f", spin)
            solver = pyscf.fci.addons.fix_spin_(solver, shift=self.opts.fix_spin_penalty, ss=spin)
        self.solver = solver

        # --- Results
        self.civec = None
        self.c0 = None
        self.c1 = None
        self.c2 = None

    def get_solver_class(self):
        if self.opts.solver_spin:
            return pyscf.fci.direct_spin1.FCISolver
        return pyscf.fci.direct_spin0.FCISolver

    def reset(self):
        super().reset()
        self.civec = None
        self.c0 = None
        self.c1 = None
        self.c2 = None

    @property
    def ncas(self):
        return self.cluster.norb_active

    @property
    def nelec(self):
        return 2*self.cluster.nocc_active

    def get_init_guess(self):
        return {'ci0' : self.civec}

    @deprecated()
    def get_t1(self):
        return self.get_c1(intermed_norm=True)

    @deprecated()
    def get_t2(self):
        return (self.c2 - einsum('ia,jb->ijab', self.c1, self.c1))/self.c0

    @deprecated()
    def get_c1(self, intermed_norm=False):
        norm = 1/self.c0 if intermed_norm else 1
        return norm*self.c1

    @deprecated()
    def get_c2(self, intermed_norm=False):
        norm = 1/self.c0 if intermed_norm else 1
        return norm*self.c2

    @deprecated()
    def get_l1(self, **kwargs):
        return None

    @deprecated()
    def get_l2(self, **kwargs):
        return None

    def kernel(self, ci0=None, eris=None, seris_ov=None):
        """Run FCI kernel."""

        if eris is None: eris = self.get_eris()
        heff = self.get_heff(eris)
        # Screening
        if seris_ov is not None:
            eris = get_screened_eris_full(eris, seris_ov, log=self.log)

        t0 = timer()
        #self.solver.verbose = 10
        e_fci, self.civec = self.solver.kernel(heff, eris, self.ncas, self.nelec, ci0=ci0)
        if not self.solver.converged:
            self.log.error("FCI not converged!")
        else:
            self.log.debugv("FCI converged.")
        self.log.timing("Time for FCI: %s", time_string(timer()-t0))
        self.log.debug("E(CAS)= %s", energy_string(e_fci))
        # TODO: This requires the E_core energy (and nuc-nuc repulsion)
        self.e_corr = np.nan
        self.converged = self.solver.converged
        s2, mult = self.solver.spin_square(self.civec, self.ncas, self.nelec)
        if not isinstance(self, UFCI_Solver) and (abs(s2) > 1e-8):
            if abs(s2) > 0.1:
                self.log.critical("FCI: S^2= %.10f  multiplicity= %.10f", s2, mult)
                raise RuntimeError("Spin restricted FCI encountered solution with S^2 >> 0")
            self.log.warning("FCI: S^2= %.10f  multiplicity= %.10f", s2, mult)
        else:
            self.log.info("FCI: S^2= %.10f  multiplicity= %.10f", s2, mult)
        self.c0, self.c1, self.c2 = self.get_cisd_amps(self.civec)
        self.log.info("FCI: weight of reference determinant= %.8g", abs(self.c0))
        mo = Orbitals(self.cluster.c_active, occ=self.cluster.nocc_active)
        self.wf = FCI_WaveFunction(mo, self.civec)

    #def get_cisd_amps(self, civec):
    #    cisdvec = pyscf.ci.cisd.from_fcivec(civec, self.ncas, self.nelec)
    #    c0, c1, c2 = pyscf.ci.cisd.cisdvec_to_amplitudes(cisdvec, self.ncas, self.cluster.nocc_active)
    #    c1 = c1/c0
    #    c2 = c2/c0
    #    return c0, c1, c2

    def get_cisd_amps(self, civec, intermed_norm=False):
        nocc, nvir = self.cluster.nocc_active, self.cluster.nvir_active
        t1addr, t1sign = pyscf.ci.cisd.t1strs(self.ncas, nocc)
        c0 = civec[0,0]
        c1 = civec[0,t1addr] * t1sign
        c2 = einsum('i,j,ij->ij', t1sign, t1sign, civec[t1addr[:,None],t1addr])
        c1 = c1.reshape(nocc,nvir)
        c2 = c2.reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3)
        if intermed_norm:
            c1 = c1/c0
            c2 = c2/c0
            c0 = 1.0
        return c0, c1, c2

    def _debug_exact_wf(self, wf):
        from pyscf.fci.addons import transform_ci
        mo = Orbitals(self.cluster.c_active, occ=self.cluster.nocc_active)
        nelec = wf.mo.nelec
        if mo.nelec != nelec:
            raise NotImplementedError
        u = self.fragment.get_overlap('mo|cluster')
        ci = transform_ci(wf.ci, nelec, u)
        wf = FCI_WaveFunction(mo, ci)
        self.wf = wf
        self.converged = True

class UFCI_Solver(FCI_Solver):
    """FCI with UHF orbitals."""

    @dataclasses.dataclass
    class Options(FCI_Solver.Options):
        fix_spin: float = None

    @property
    def ncas(self):
        ncas = self.cluster.norb_active
        if ncas[0] != ncas[1]:
            raise NotImplementedError()
        return ncas[0]

    @property
    def nelec(self):
        return self.cluster.nocc_active

    def get_solver_class(self):
        return pyscf.fci.direct_uhf.FCISolver

    @deprecated()
    def get_t2(self):
        ca, cb = self.get_c1(intermed_norm=True)
        caa, cab, cbb = self.get_c2(intermed_norm=True)
        taa = caa - einsum('ia,jb->ijab', ca, ca) + einsum('ib,ja->ijab', ca, ca)
        tbb = cbb - einsum('ia,jb->ijab', cb, cb) + einsum('ib,ja->ijab', cb, cb)
        tab = cab - einsum('ia,jb->ijab', ca, cb)
        return (taa, tab, tbb)

    @deprecated()
    def get_c1(self, intermed_norm=False):
        norm = 1/self.c0 if intermed_norm else 1
        return (norm*self.c1[0], norm*self.c1[1])

    @deprecated()
    def get_c2(self, intermed_norm=False):
        norm = 1/self.c0 if intermed_norm else 1
        return (norm*self.c2[0], norm*self.c2[1], norm*self.c2[2])

    #def get_cisd_amps(self, civec):
    #    cisdvec = pyscf.ci.ucisd.from_fcivec(civec, self.ncas, self.nelec)
    #    c0, (c1a, c1b), (c2aa, c2ab, c2bb) = pyscf.ci.ucisd.cisdvec_to_amplitudes(cisdvec, 2*[self.ncas], self.nelec)
    #    c1a = c1a/c0
    #    c1b = c1b/c0
    #    c2aa = c2aa/c0
    #    c2ab = c2ab/c0
    #    c2bb = c2bb/c0
    #    return c0, (c1a, c1b), (c2aa, c2ab, c2bb)

    def get_cisd_amps(self, civec, intermed_norm=False):
        norba, norbb = self.cluster.norb_active
        nocca, noccb = self.cluster.nocc_active
        nvira, nvirb = self.cluster.nvir_active

        t1addra, t1signa = pyscf.ci.cisd.tn_addrs_signs(norba, nocca, 1)
        t1addrb, t1signb = pyscf.ci.cisd.tn_addrs_signs(norbb, noccb, 1)
        t2addra, t2signa = pyscf.ci.cisd.tn_addrs_signs(norba, nocca, 2)
        t2addrb, t2signb = pyscf.ci.cisd.tn_addrs_signs(norbb, noccb, 2)
        na = pyscf.fci.cistring.num_strings(norba, nocca)
        nb = pyscf.fci.cistring.num_strings(norbb, noccb)

        civec = civec.reshape(na,nb)
        c0 = civec[0,0]
        c1a = (civec[t1addra,0] * t1signa).reshape(nocca,nvira)
        c1b = (civec[0,t1addrb] * t1signb).reshape(noccb,nvirb)

        nocca_comp = nocca*(nocca-1)//2
        noccb_comp = noccb*(noccb-1)//2
        nvira_comp = nvira*(nvira-1)//2
        nvirb_comp = nvirb*(nvirb-1)//2
        c2aa = (civec[t2addra,0] * t2signa).reshape(nocca_comp, nvira_comp)
        c2bb = (civec[0,t2addrb] * t2signb).reshape(noccb_comp, nvirb_comp)
        c2aa = pyscf.cc.ccsd._unpack_4fold(c2aa, nocca, nvira)
        c2bb = pyscf.cc.ccsd._unpack_4fold(c2bb, noccb, nvirb)
        c2ab = einsum('i,j,ij->ij', t1signa, t1signb, civec[t1addra[:,None],t1addrb])
        c2ab = c2ab.reshape(nocca,nvira,noccb,nvirb).transpose(0,2,1,3)

        # C1 and C2 in intermediate normalization:
        if intermed_norm:
            c1a = c1a/c0
            c1b = c1b/c0
            c2aa = c2aa/c0
            c2ab = c2ab/c0
            c2bb = c2bb/c0
            c0 = 1.0
        return c0, (c1a, c1b), (c2aa, c2ab, c2bb)

    def make_rdm1(self, civec=None):
        if civec is None: civec = self.civec
        self.dm1 = self.solver.make_rdm1s(civec, self.ncas, self.nelec)
        return self.dm1

    def make_rdm12(self, civec=None):
        if civec is None: civec = self.civec
        self.dm1, self.dm2 = self.solver.make_rdm12s(civec, self.ncas, self.nelec)
        return self.dm1, self.dm2
