import dataclasses

import numpy as np
import pyscf.lo

from vayesta.core.types import FCI_WaveFunction
from vayesta.core.types.wf.fci import UFCI_WaveFunction_w_dummy
from vayesta.core.util import log_time
from vayesta.solver.solver import ClusterSolver, UClusterSolver
from vayesta.core.types.wf import RRDM_WaveFunction


class DMRG_Solver(ClusterSolver):
    @dataclasses.dataclass
    class Options(ClusterSolver.Options): 
        localiser: str = None           # Localisation method for cluster orbitals
        bond_dims: list = None          # Bond dimensions for DMRG
        noises: list = None             # Noises for DMRG
        thrds: list = None              # Thresholds for DMRG
        n_sweeps: int = 2               # Number of sweeps for DMRG

        n_moments: tuple = None         # Number of moments to calculate
        bond_dims_moments: list = None  # Bond dimensions for DMRG moments
        noises_moments: list = None     # Noises for DMRG moments
        thrds_moments: list = None      # Thresholds for DMRG moments
        n_sweeps_moments: int = None    # Number of sweeps for DMRG moments



    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.log.debugv("type(solver)= %r", type(self))

        if self.opts.bond_dims_moments is None:
            self.opts.bond_dims_moments = self.opts.bond_dims
        if self.opts.noises_moments is None:
            self.opts.noises_moments = self.opts.noises
        if self.opts.thrds_moments is None:
            self.opts.thrds_moments = self.opts.thrds
        if self.opts.n_sweeps_moments is None:
            self.opts.n_sweeps_moments = self.opts.n_sweeps 


    def kernel(self):
        try:
            from pyblock2._pyscf.ao2mo import integrals as itg
            from pyblock2.driver.core import DMRGDriver, SymmetryTypes
        except ImportError:
            self.log.error("Block2 not found - required for DMRG calculations")

        if self.opts.localiser is not None:
            if self.opts.localiser.lower() == "lowdin":
                mo_coeff_old = self.mf.mo_coeff.copy()
                self.mf.mo_coeff = lo.orth.lowdin(self.mf.get_ovlp())
            elif self.opts.localiser.lower() == "er":
                mo_coeff_old = self.mf.mo_coeff.copy()
                self.mf.mo_coeff = lo.orth.ER(self.mf.get_ovlp())

        mf_clus = self.hamil.to_pyscf_mf(allow_dummy_orbs=True, allow_df=True)[0]
        ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf_clus, ncore=0, ncas=None, g2e_symm=1)
        self.driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=4)
        self.driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)        

        self.mpo = self.driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, integral_cutoff=1E-8, iprint=1)
        self.ket = self.driver.get_random_mps(tag="KET", bond_dim=100, nroots=1)
        
        with log_time(self.log.timing, "Time for DMRG: %s"):
            self.energy = self.driver.dmrg(self.mpo, self.ket, n_sweeps=self.opts.n_sweeps, bond_dims=self.opts.bond_dims, noises=self.opts.noises, thrds=self.opts.thrds, iprint=1)

        #self.converged = self.driver.converged

        dm1 = self.driver.get_1pdm(self.ket)
        dm2 = self.driver.get_2pdm(self.ket).transpose(0, 3, 1, 2)
        self.wf = RRDM_WaveFunction(self.hamil.mo, dm1, dm2)

        # Cluster Moments
        if self.opts.n_moments is not None:
            try:
                from dyson.expressions import DMRG
            except ImportError:
                self.log.error("Dyson not found - required for moment calculations")
                self.log.info("Skipping cluster moment calculations")
            
            
            mf_clus, frozen = self.hamil.to_pyscf_mf(allow_dummy_orbs=True, allow_df=True)

            with log_time(self.log.timing, "Time for hole moments: %s"):
                expr = FCI["1h"](mf_clus,)
                self.hole_moments = expr.build_gf_moments(nmom[0])
            with log_time(self.log.timing, "Time for hole moments: %s"):    
                expr = FCI["1p"](mf_clus, )
                self.particle_moments = expr.build_gf_moments(nmom[1])            

