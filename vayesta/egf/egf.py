# --- Standard
import dataclasses

# --- External
import numpy as np
from vayesta.core.util import (
    NotCalculatedError,
    break_into_lines,
    cache,
    deprecated,
    dot,
    einsum,
    energy_string,
    log_method,
    log_time,
    time_string,
    timer,
)
from vayesta.core.qemb import Embedding
from vayesta.core.fragmentation import SAO_Fragmentation
from vayesta.core.fragmentation import IAOPAO_Fragmentation
from vayesta.mpi import mpi
from vayesta.ewf.ewf import REWF
from vayesta.egf.fragment import Fragment
from vayesta.egf.self_energy import *

from dyson import MBLGF, MBLSE, FCI, CCSD, AufbauPrinciple, AuxiliaryShift, MixedMBLGF, NullLogger, Lehmann, build_spectral_function


@dataclasses.dataclass
class Options(REWF.Options):
    """Options for EGF calculations."""
    eta: float = 1e-1 # Broadening factor
    proj: int = 2     # Number of projectors used on self-energy (1, 2)
    use_sym: bool = True
    se_degen_tol: float = 1e-6 # Tolerance for degeneracy of self-energy poles
    se_eval_tol: float = 1e-6  # Tolerance for self-energy eignvalues
    drop_non_causal: bool = False # Drop non-causal poles
    aux_shift: bool = False # Use auxiliary shift to ensure correct electron number in the physical space
    aux_shift_frag: bool = False # Use auxiliary shift to ensure correct electron number in the fragment space
    se_mode: str = 'moments' # Mode for self-energy reconstruction (moments, lehmann)
    nmom_se: int = np.inf # Number of conserved moments for self-energy
    non_local_se: str = None # Non-local self-energy (GW, CCSD, FCI)
    sym_moms: bool = True # Use symmetrized moments
    hermitian_mblgf: bool = True # Use hermitian MBLGF
    hermitian_mblse: bool = True # Use hermitian MBLSE
    global_static_potential: bool = False # Use global static potential

    solver_options: dict = Embedding.Options.change_dict_defaults("solver_options", conv_tol=1e-15, conv_tol_normt=1e-15)
    bath_options: dict = Embedding.Options.change_dict_defaults("bath_options", bathtype='ewdmet', order=1, max_order=1, dmet_threshold=1e-12)

class REGF(REWF):
    Options = Options
    Fragment = Fragment

    def __init__(self, mf, solver="CCSD", log=None, **kwargs):
        super().__init__(mf, solver=solver, log=log, **kwargs)

        # Logging
        with self.log.indent():
            # Options
            self.log.info("Parameters of %s:", self.__class__.__name__)
            self.log.info(break_into_lines(str(self.opts), newline="\n    "))
            #self.log.info("Time for %s setup: %s", self.__class__.__name__, time_string(timer() - t0))

    def kernel(self):
        super().kernel()
        couplings = []
        energies = []
        self.static_self_energy = np.zeros_like(self.get_fock())

        #if self.opts.nmom_se == np.inf:
        if self.opts.se_mode == 'lehmann':
            if self.opts.proj == 1:
                self.self_energy, self.static_self_energy, self.static_potential = make_self_energy_1proj(self, hermitian=self.opts.hermitian_mblgf, sym_moms=self.opts.sym_moms, use_sym=self.opts.use_sym, eta=self.opts.eta,aux_shift_frag=self.opts.aux_shift_frag, se_degen_tol=self.opts.se_degen_tol, se_eval_tol=self.opts.se_eval_tol)
            elif self.opts.proj == 2:
                self.self_energy, self.static_self_energy, self.static_potential = make_self_energy_2proj(self, hermitian=self.opts.hermitian_mblgf, sym_moms=self.opts.sym_moms, use_sym=self.opts.use_sym, eta=self.opts.eta)
            else:
                return NotImplementedError()
        elif self.opts.se_mode == 'moments':
            self.self_energy_moments, self.static_self_energy, self.static_potential = make_self_energy_moments(self, proj=self.opts.proj, hermitian=self.opts.hermitian_mblgf, sym_moms=self.opts.sym_moms, use_sym=self.opts.use_sym, eta=self.opts.eta)
            if self.opts.non_local_se is not None:
                self.opts.non_local_se = self.opts.non_local_se.upper()
                if self.opts.non_local_se.upper() == 'GW-RPA' or self.opts.non_local_se.upper() == 'GW-dTDA':
                    try:
                        from momentGW import GW
                        gw = GW(self.mf)
                        if self.opts.non_local_se.upper() == 'GW-dRPA':
                            gw.polarizability = 'drpa'
                        elif self.opts.non_local_se.upper() == 'GW-dTDA':
                            gw.polarizability = 'dtda'
                        integrals = gw.ao2mo()
                        non_local_se_static = gw.build_se_static(integrals)
                        seh, sep = gw.build_se_moments(self.opts.nmom_se-1, integrals, mo_energy=dict(g=gw.mo_energy, w=gw.mo_energy))
                        non_local_se_moms = seh + sep
                    except ImportError:
                        raise ImportError("momentGW required for non-local GW self-energy contribution")
                elif self.opts.non_local_se == 'FCI' or self.opts.non_local_se == 'CCSD':
                    EXPR = FCI if self.opts.non_local_se == 'FCI' else CCSD
                    expr = FCI["1h"](self.mf)
                    th = expr.build_gf_moments(self.opts.nmom_se)
                    expr = FCI["1p"](self.mf)
                    tp = expr.build_gf_moments(self.opts.nmom_se)
                    
                    solverh = MBLGF(th, hermitian=self.opts.hermitian_mblgf, log=NullLogger())
                    solverp = MBLGF(tp, hermitian=self.opts.hermitian_mblgf, log=NullLogger())
                    solver = MixedMBLGF(solverh, solverp)
                    solver.kernel()
                    non_local_se_static = th[1] + tp[1]
                    non_local_se = solver.get_self_energy()
                    non_local_se_moms = np.array([non_local_se.moment(i) for i in range(self.opts.nmom_se)])
                else:
                    raise NotImplementedError()
                self.static_self_energy = remove_fragments_from_full_moments(self, non_local_se_static) + self.static_self_energy
                self.self_energy_moments = remove_fragments_from_full_moments(self, non_local_se_moms, proj=self.opts.proj) + self.self_energy_moments
            phys = self.mf.mo_coeff.T @ self.mf.get_fock() @ self.mf.mo_coeff + self.static_self_energy
            solver = MBLSE(phys, self.self_energy_moments, hermitian=self.opts.hermitian_mblse, log=self.log)
            solver.kernel()
            self.self_energy = solver.get_self_energy()

        phys = self.mo_coeff.T @ self.get_fock() @ self.mo_coeff + self.static_self_energy 
        gf = Lehmann(*self.self_energy.diagonalise_matrix_with_projection(phys), chempot=self.self_energy.chempot)
        dm = gf.occupied().moment(0) * 2.0
        nelec_gf = np.trace(dm)
        self.log.info('Number of electrons in GF: %f'%nelec_gf)
        Shift = AuxiliaryShift if self.opts.aux_shift else AufbauPrinciple
        shift = Shift(phys, self.self_energy, self.mf.mol.nelectron, occupancy=2, log=self.log)
        shift.kernel()
        self.self_energy = shift.get_self_energy()
        gf = shift.get_greens_function()
        dm = gf.occupied().moment(0) * 2.0
        nelec_gf = np.trace(dm)
        self.log.info('Number of electrons in (shifted) GF: %f'%nelec_gf)
        
        v_old = self.static_potential.copy()
        sc = self.mf.get_ovlp() @ self.mo_coeff
        if self.opts.global_static_potential:
            self.static_potential = self.mo_coeff @ self.self_energy.as_static_potential(self.mf.mo_energy, eta=self.opts.eta)  @ self.mo_coeff.T
        self.static_potential = self.mf.get_ovlp() @ self.static_potential @ self.mf.get_ovlp()

        # if diis is not None:
        #     self.static_potential = diis.update(self.static_potential)
        
        # new_fock = self.fock + sc @ self.static_self_energy @ sc.T + self.static_potential
        # self.sc_fock = self.damping * self.fock + (1-self.damping) * new_fock
        # #self.sc_fock = self.sc_fock + (1-self.damping) * self.static_potential
        self.gf = gf
        #self.gf, self.gf_qp = self.get_greens_function()

        # if self.sc:
        #     e, mo_coeff = self.fock_scf(self.static_potential)
        # else:
        #     e, mo_coeff = scipy.linalg.eigh(self.sc_fock, self.get_ovlp())

        gap = lambda gf: gf.physical().virtual().energies[0] - gf.physical().occupied().energies[-1]
        dynamic_gap = gap(self.gf)
        #static_gap = gap(self.gf_qp)
        self.log.info("Dynamic Gap = %f"%dynamic_gap)
        #self.log.info("Static Gap = %f"%static_gap)

