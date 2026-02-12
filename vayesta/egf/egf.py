# --- Standard
import dataclasses
from multiprocessing import Value

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
from vayesta.core.qemb.static_observable import make_global_one_body, make_local_one_body, symmetrize_observable
from vayesta.core.types.dynamical import SE_LehmannRep
from vayesta.mpi import mpi
from vayesta.ewf.ewf import REWF
from vayesta.egf.fragment import Fragment
from vayesta.egf.self_energy import *
from vayesta.egf.qsegf import QSEGF_RHF

from dyson import MBLGF, MBLSE, FCI, CCSD, AufbauPrinciple, AuxiliaryShift, Lehmann
from dyson.solvers.static.chempot import search_aufbau_global as find_chempot
from dyson.util.moments import se_moments_to_gf_moments, gf_moments_to_se_moments


@dataclasses.dataclass
class Options(REWF.Options):
    """Options for EGF calculations."""
    proj: int = 2     # Number of projectors used on self-energy (1, 2)
    proj_static_se: int = None # Number of projectors used on static self-energy (same as proj if None)
    use_sym: bool = True # Use symmetry for self-energy reconstruction
    se_degen_tol: float = 1e-6 # Tolerance for degeneracy of self-energy poles
    se_eval_tol: float = 1e-6  # Tolerance for self-energy eignvalues
    img_space : bool = True    # Use image space for self-energy reconstruction
    drop_non_causal: bool = False # Drop non-causal poles
    chempot_global: str = None # Use auxiliary shift to ensure correct electron number in the physical space (None, 'auf', 'aux')
    chempot_clus: str = 'auto' # Use auxiliary shift to ensure correct electron number in the fragment space (None, 'auf', 'aux')
    se_mode: str = 'moments_mblgf' # Mode for self-energy reconstruction (moments, lehmann)
    se_static: str = 'cluster_moments_corr' # Method for static self-energy (cluster_moments, fock, fock_corr, cluster_fock_corr)
    nmom_se: int = None      # Number of conserved moments for self-energy
    non_local_se: str = None # Non-local self-energy (GW, CCSD, FCI)
    sym_moms: bool = True # Use symmetrized moments
    hermitian_lanczos: bool = True # Use hermitian Lanczos algorithm 
    global_1dm: bool = False  # Use global 1DM for zeroth moments
    global_1dm_static: bool = False  # Use Fock matrix evaluated at global 1DM for static self-energy
    static_with_mf: bool = False # Include mean-field in cluster static self-energy
    combine_sectors_in_cluster: bool = True # Combine sectors when reconstructing self-energy in cluster


    solver_options: dict = Embedding.Options.change_dict_defaults("solver_options", n_moments=(6,6), conv_tol=1e-15, conv_tol_normt=1e-15)
    bath_options: dict = Embedding.Options.change_dict_defaults("bath_options", bathtype='ewdmet', order=1, max_order=1, dmet_threshold=1e-12)

class REGF(REWF):
    Options = Options
    Fragment = Fragment

    def __init__(self, mf, solver="CCSD", log=None, **kwargs):
        super().__init__(mf, solver=solver, log=log, **kwargs)

        if self.opts.proj_static_se is None:
            self.opts.proj_static_se = self.opts.proj

        if self.opts.chempot_clus == 'auto':
            if self.opts.se_mode == 'moments' or self.opts.se_mode == 'moments_mblgf':
                self.opts.chempot_clus = 'aux' if self.opts.chempot_global == 'aux' else 'auf'
            elif self.opts.se_mode == 'lehmann':
                self.opts.chempot_clus = None
            elif self.opts.se_mode == 'spectral':
                self.opts.chempot_clus = 'aux' if self.opts.chempot_global == 'aux' else None
            else:
                raise ValueError("Invalid self-energy mode")
            
        # Logging
        with self.log.indent():
            # Options
            self.log.info("Parameters of %s:", self.__class__.__name__)
            self.log.info(break_into_lines(str(self.opts), newline="\n    "))
            #self.log.info("Time for %s setup: %s", self.__class__.__name__, time_string(timer() - t0))
    
    def kernel(self):
        """Run the EGF calculation"""
        super().kernel()
       
            
        self.se = self.make_self_energy(se_mode=self.opts.se_mode, hermitian_lanczos=self.opts.hermitian_lanczos, combine_sectors=self.opts.combine_sectors_in_cluster, proj=self.opts.proj, global_1dm=self.opts.global_1dm)


        self.gf, self.se = self.make_greens_function(self.se, chempot_global=self.opts.chempot_global)
        
        #gm_energy = self.galitskii_migdal(self.gf, self.se)
        #self.log.info("Galitskii-Migdal energy: %s", energy_string(gm_energy))

        ea = self.gf.physical().virtual().energies[0] 
        ip = self.gf.physical().occupied().energies[-1]
        gap = ea - ip
        self.log.info("IP  = %8f"%ip)
        self.log.info("EA  = %8f"%ea)
        self.log.info("Gap = %8f"%gap)

    def make_greens_function(self, se, chempot_global=None):
        """
        Calculate Green's function from self-energy using Dyson equation.

        Parameters
        ----------
        se : SE_Lehmann
            Self-energy in Lehmann representation
        chempot_global : (None, 'auf', 'aux')
            Type of chemical potential optimisation for full system Green's function. AufbauPrinciple or AuxiliaryShift.
        
        Returns
        -------
        gf : Lehmann
            Green's function
        """

        if chempot_global is None:
           chempot_global = self.opts.chempot_global
        SC = self.get_ovlp() @ self.mo_coeff

        assert isinstance(se, SE_LehmannRep) # Change to accept all SE types?
        assert se.nsectors == 1 # Implement for separat particle hole GFs?

        if se.statics.ndim == 3:
            static_self_energy = np.sum(se.statics, axis=0)
        else:
            static_self_energy = se.statics
        
        if se.overlaps is not None:
            if se.overlaps.ndim == 3:
                overlap = np.sum(se.overlaps, axis=0)
            else:
                overlap = se.overlaps
        else:
            overlap = None
        self_energy = se.lehmanns[0]
    

        gf = Lehmann(*self_energy.diagonalise_matrix_with_projection(static_self_energy, overlap=overlap) )

        # Add fock self-consistency here?

        if chempot_global == 'auf':
            cpt, err = find_chempot(gf, self.mf.mol.nelectron, occupancy=2)
            gf = gf.copy(chempot=cpt)
            self.log.info("Applied global chemical potential shift: %f (error in N_elec: %e)"%(cpt, err))


        elif chempot_global == 'aux':
            mu_solver = dyson.solvers.static.chempot.AuxiliaryShift(static_self_energy, self_energy, self.mf.mol.nelectron, overlap=overlap)
            mu_solver.kernel()
            result = mu_solver.result
            gf = result.get_greens_function()
            self_energy = result.get_self_energy()
            self.log.info("Auxilliary shift applied for global chemical potential")

        dm = gf.occupied().moment(0) 
        nelec_gf = np.trace(dm) * 2.0
        self.log.info('Number of electrons in GF: %f'%nelec_gf)
        return gf, self_energy


    def make_self_energy(self, se_mode=None, se_static_mode=None, hermitian_lanczos=None, combine_sectors=False, proj=None, global_1dm=None):

        se_mode = self.opts.se_mode if se_mode is None else se_mode
        se_static_mode = self.opts.se_static if se_static_mode is None else se_static_mode
        hermitian_lanczos = self.opts.hermitian_lanczos if hermitian_lanczos is None else hermitian_lanczos
        combine_sectors = self.opts.combine_sectors_in_cluster if combine_sectors is None else combine_sectors
        proj = self.opts.proj if proj is None else proj
        global_1dm = self.opts.global_1dm if global_1dm is None else global_1dm

        # if self.opts.se_mode == 'lehmann' or self.opts.se_mode == 'spectral':
        #     overlap, static_self_energy, self_energy = self.make_self_energy_lehmann(self.opts.proj)
        #     self.se_overlap = None
        #     phys = np.sum(self.static_self_energy, axis=0)
        #     phys = self.static_self_energy + self.mo_coeff.T @ self.get_fock() @ self.mo_coeff
        #     static_self_energy = phys
            
        # elif self.opts.se_mode == 'moments':
        #     overlap, static_self_energy, self_energy= self.make_self_energy_moments(self.opts.proj, nmom_se=self.opts.nmom_se, hermitian_mblse=self.opts.hermitian_lanczos, non_local_se=self.opts.non_local_se)

        # else:
        #     pass
        #     #raise ValueError("Invalid self-energy mode: %s"%self.opts.se_mode)

        dm1 = None
        if se_static_mode == 'cluster_moments':
            se_static = make_static_self_energy(self, proj=1, sym_moms=self.opts.sym_moms, with_mf=True, use_sym=self.opts.use_sym, orth_basis=self.opts.global_1dm)
        elif se_static_mode == 'cluster_moments_corr':
            se_static = make_static_self_energy(self, proj=1, sym_moms=self.opts.sym_moms, with_mf=False, use_sym=self.opts.use_sym, orth_basis=self.opts.global_1dm)
        elif se_static_mode == 'fock':
            se_static = np.diag(self.mf.mo_energy)
        elif se_static_mode == 'fock_corr':
            dm1 = self._make_rdm1_ccsd_global_wf(self, slow=True, ao_basis=True) 
            fock_corr_ao = self.mf.get_fock(dm=dm1) 
            se_static = self.mo_coeff.T @ fock_corr_ao @ self.mo_coeff
        elif se_static_mode == 'cluster_fock_corr':
            raise NotImplementedError()
        else:
            raise ValueError("Invalid static self-energy mode: %s"%se_static_mode)
        
        se = make_self_energy(self, se_mode=self.opts.se_mode, combine_sectors=self.opts.combine_sectors_in_cluster, proj=self.opts.proj, orth_basis=self.opts.global_1dm)
        se._statics = se_static

        if se_static_mode == 'cluster_moments_corr':
            nocc = self.mf.mol.nelectron // 2
            fock_mo = np.diag(self.mf.mo_energy)

            if se.statics.ndim == 2:
                se._statics += fock_mo
            else:
                se._statics[0][:nocc, :nocc] += fock_mo[:nocc, :nocc]
                se._statics[1][nocc:, nocc:] += fock_mo[nocc:, nocc:]
            
        #se._overlaps = None
        self.se_orig = se
        assert se.hermitian == hermitian_lanczos, "Hermiticity of self-energy does not match specified value"
    
        if self.opts.se_mode == 'moments' or self.opts.se_mode == 'moments_mblgf':

            
            #TODO: Refactor this into egf.make_self_energy and egf.make_static_self_energy functions
            if self.opts.global_1dm:
                if dm1 is None:
                    dm1 = self._make_rdm1_ccsd_global_wf(self, slow=True) / 2
                else:
                    dm1 = dm1 / 2
                eye_m_dm1 = np.eye(dm1.shape[0]) - dm1
                self.ovlp_global_wf = np.array([dm1, eye_m_dm1])

                # statics = se.statics
                # from dyson.util.linalg import matrix_power
                # unorth_h, _ = matrix_power(self.ovlp_global_wf[0], 0.5, hermitian=True, return_error=False)
                # unorth_p, _ = matrix_power(self.ovlp_global_wf[1], 0.5, hermitian=True, return_error=False)

                # unorth = np.array([unorth_h, unorth_p])
                # statics = einsum('spP,sqQ,sPQ->spq', unorth,  unorth, statics)
                # se_moms = se.moments
                # se_moms = einsum('spP,sqQ,s...PQ->s...pq', unorth, unorth, se_moms)

                se = se.unorthogonalize(overlaps=self.ovlp_global_wf)

            else:
                #se = make_self_energy(self, se_mode='moments', combine_sectors=self.opts.combine_sectors_in_cluster, proj=self.opts.proj, orth_basis=False)
                pass
            if se.statics.ndim == 3:
                se._statics = np.sum(se.statics, axis=0)    
            spec = se.to_spectral().combine_sectors()
            se = spec.to_se_lehmann()
            se._overlaps = None

        elif self.opts.se_mode == 'lehmann':
            if not self.opts.combine_sectors_in_cluster:
                se = se.combine_sectors()
            

        elif self.opts.se_mode == 'spectral':
            se = se.to_se_lehmann()


        # if self._ex_stat is not None:
        #     se._statics[0] = self._ex_stat
        self.se_moms = se
        #assert se.hermitian == hermitian_lanczos, "Hermiticity of self-energy does not match specified value"

        return se
    

    def make_static_self_energy(self, proj, sym_moms=False, with_mf=False, use_sym=True):
        return make_static_self_energy(self, proj=proj, sym_moms=sym_moms, with_mf=with_mf, use_sym=use_sym)

    def make_gf_moments(self, nmom=2):

        fragments = self.get_fragments(sym_parent=None) if self.opts.use_sym else self.get_fragments()
        # overlap and static self-energy are first two GF moments
        gf_moms_clusters = [f.results.gf_moments[:,:nmom,:,:] for f in fragments]
        gf_moms = make_global_one_body(self, gf_moms_clusters, symmetrize=sym_moms, use_sym=self.opts.use_sym, proj=1, fragments=fragments)
        return gf_moms
    
    def make_self_energy_lehmann(self, proj, nmom_gf=None):
        """
        Reconstruct self-energy from fragment using block Lanczos to obtain a Lehmann representation
        in the cluster, projecting and summing over all fragments.
        
        Parameters
        ----------
        proj : int
            Number of projectors to use for fragment projection of self-energy
        nmom_gf : int
            Number of Green's function moments to use for block Lanczos
        
        Returns
        -------
        static_self_energy : ndarray
            Static part of the self-energy
        self_energy : Lehmann
            Self-energy in Lehmann representation
        """
        
        if proj == 1:
            static_self_energy, self_energy = make_self_energy_1proj(self, nmom_gf=nmom_gf, hermitian=self.opts.hermitian_mblgf, sym_moms=self.opts.sym_moms, use_sym=self.opts.use_sym, img_space=self.opts.img_space, chempot_clus=self.opts.chempot_clus, se_degen_tol=self.opts.se_degen_tol, se_eval_tol=self.opts.se_eval_tol)
        elif proj == 2:
            static_self_energy, self_energy = make_self_energy_2proj(self, nmom_gf=nmom_gf, hermitian=self.opts.hermitian_mblgf, sym_moms=self.opts.sym_moms, use_sym=self.opts.use_sym, chempot_clus=self.opts.chempot_clus, se_degen_tol=self.opts.se_degen_tol, se_eval_tol=self.opts.se_eval_tol)
        else:
            raise NotImplementedError()
        return None, static_self_energy, self_energy
        

    def make_self_energy_moments(self, proj, nmom_se=None, hermitian_mblse=False, non_local_se=True, **kwargs):
        """
        Reconstruct self-energy moments from fragment self-energy moments and perform block Lanczos
        on full system to obtain a Lehmann self-energy.
        
        Parameters
        ----------
        proj : int
            Number of projectors to use for fragment projection of self-energy moments
        nmom_se : int
            Number of self-energy moments to calculate
        hermitian_mblse : bool
            Use hermitian MBLSE
        non_local_se : str
            Combine fragment self-energy with non-local self-energy (GW, CCSD, FCI) calculated on full system

        Returns
        -------
        static_self_energy : ndarray
            Static part of the self-energy
        self_energy : MBLSE
            Self-energy in Lehmann representation
        """
        
        fragments = self.get_fragments(sym_parent=None) if self.opts.use_sym else self.get_fragments()
        # overlap and static self-energy are first two GF moments
        gf_moms_clusters = [f.results.gf_moments[:,:2,:,:] for f in fragments]
        gf_moms = make_global_one_body(self, gf_moms_clusters, symmetrize=False, use_sym=self.opts.use_sym, proj=1, fragments=fragments)


        if self.opts.global_1dm:
            # Replace zeroth moment with global 1DM
            global_gf_mom_0_h = self._make_rdm1_ccsd_global_wf() / 2
            global_gf_mom_0_p = np.eye(self.mo_coeff.shape[1]) - global_gf_mom_0_h
            global_gf_mom_0 = np.array([global_gf_mom_0_h, global_gf_mom_0_p])
            gf_moms[:,0,:,:] = global_gf_mom_0
            
            global_gf_mom_0_clus = make_local_one_body(self, global_gf_mom_0, fragments=fragments)

            for i, f in enumerate(fragments):
                for j, _ in enumerate(['h','p']):
                    f.results.gf_moments[i,j,:,:] = global_gf_mom_0_clus[i,j]
                    f.results.se_static[j], f.results.se_moments[j] = gf_moments_to_se_moments(f.results.gf_moments[j]) 
                    f.results.gf_moments[j,0,:,:] = global_gf_mom_0_clus[i,j]
                    f.results.gf_moments[j,1,:,:] = f.results.se_static[j]


        def cluster_shift_moms(f, shift='fragment', shift_type=True):
            gf_moms = f.results.gf_moments
            se_moms = f.results.se_moments
            if shift == 'fragment':
                cf = f.get_overlap('frag|cluster')
                cfc = cf.T @ cf
                if self.opts.proj == 1:
                    self.log.info("Projecting cluster Green's function moments with 1 projector")
                    gf_moms = 0.5 * (einsum('pP,...Pq->...pq', cfc, gf_moms) + einsum('qQ,...pQ->...pq', cfc, gf_moms))
                elif self.opts.proj == 2:
                    self.log.info("Projecting cluster Green's function moments with 2 projectors")
                    gf_moms = einsum('pP,qQ,...PQ->...pq', cfc, cfc, gf_moms)
            #se_static, se_moms = f.results.se_static, f.results.se_moments
            
            results = []
            for i, s in enumerate(['h', 'p']):
                solver = dyson.MBLGF(gf_moms[i], hermitian=self.opts.hermitian_mblgf)
                solver.kernel()
                results.append(solver.result)

            result = dyson.Spectral.combine(results[0], results[1])
            gf = result.get_greens_function()
            dm = gf.occupied().moment(0)


            if shift_type is not None:
                nelec_target = f.nelectron if shift == 'fragment' else 2 * f.cluster.nocc
                self.log.info("Fragment %d Tr[dm1] = : %.6f   target = %.6f"%(f.id, 2*np.trace(dm), nelec_target))

                

                mu_solver = dyson.solvers.static.chempot.AuxiliaryShift(result.get_static_self_energy(), result.get_self_energy(), nelec_target, overlap=result.get_overlap())
                mu_solver.kernel()
                result = mu_solver.result
                gf = result.get_greens_function()
                dm = gf.occupied().moment(0)    
                self.log.info("Fragment %d Aux shift Tr[dm1] = : %.6f   target = %.6f"%(f.id, 2*np.trace(dm), nelec_target))

            se = result.get_self_energy()
            se_moms_h = se.occupied().moment(range(se_moms.shape[1]))
            se_moms_p = se.virtual().moment(range(se_moms.shape[1]))

            gf = result.get_greens_function()
            gf_moms_h = gf.occupied().moments(range(2))
            gf_moms_p = gf.virtual().moments(range(2))

            return np.array([gf_moms_h, gf_moms_p]), np.array([se_moms_h, se_moms_p]) 
        

        

        # se_moms_clusters = []        
        # gf_moms_clusters = []
        # for f in fragments:
        #     gf, se = cluster_shift_moms(f, shift='cluster', shift_type=True)
        #     se_moms_clusters.append(se)
        #     gf_moms_clusters.append(gf)
        # se_moms_clusters = np.array(se_moms_clusters)
        # gf_moms_clusters = np.array(gf_moms_clusters)


        se_moms_clusters = [f.results.se_moments for f in fragments]
        gf_moms_clusters = [f.results.gf_moments for f in fragments]

        se_moms = make_global_one_body(self, se_moms_clusters, symmetrize=False, use_sym=self.opts.use_sym, proj=proj, fragments=fragments)
        gf_moms = make_global_one_body(self, gf_moms_clusters, symmetrize=False, use_sym=self.opts.use_sym, proj=proj, fragments=fragments)

        self.gf_moms = gf_moms
        nmom_se = se_moms.shape[1]

        if non_local_se is not None:
            # if non_local_se.upper() == 'GW-dRPA' or non_local_se.upper() == 'GW-dTDA':
            #     try:
            #         from momentGW import GW
            #         gw = GW(self.mf)
            #         if non_local_se.upper() == 'GW-dRPA':
            #             gw.polarizability = 'drpa'
            #         elif non_local_se.upper() == 'GW-dTDA':
            #             gw.polarizability = 'dtda'
            #         integrals = gw.ao2mo()
            #         non_local_se_static = gw.build_se_static(integrals)
            #         seh, sep = gw.build_se_moments(nmom_se-1, integrals, mo_energy=dict(g=gw.mo_energy, w=gw.mo_energy))
            #         non_local_se_moms = seh + sep
            #     except ImportError:
            #         raise ImportError("momentGW required for non-local GW self-energy contribution")
            # elif non_local_se == 'FCI' or non_local_se == 'CCSD':
            #     EXPR = FCI if non_local_se == 'FCI' else CCSD
            #     expr = FCI["1h"](self.mf)
            #     th = expr.build_gf_moments(nmom_se)
            #     expr = FCI["1p"](self.mf)
            #     tp = expr.build_gf_moments(nmom_se)
                
            #     solverh = MBLGF(th, hermitian=hermitian_mblgf, log=NullLogger())
            #     solverp = MBLGF(tp, hermitian=hermitian_mblgf, log=NullLogger())
            #     solver = MixedMBLGF(solverh, solverp)
            #     solver.kernel()
            #     non_local_se_static = th[1] + tp[1]
            #     non_local_se = solver.get_self_energy()
            #     non_local_se_moms = np.array([non_local_se.moment(i) for i in range(nmom_se)])
            # else:
            #     raise NotImplementedError()
            # static_self_energy = remove_fragments_from_full_moments(self, non_local_se_static) + static_self_energy
            # self_energy_moments = remove_fragments_from_full_moments(self, non_local_se_moms, proj=proj) + self_energy_moments
            try:
                import momentGW
            except ImportError:
                raise ImportError("momentGW required for non-local GW self-energy contribution")
            
            
            relaxed = True
            
            gw = momentGW.GW(self.mf)
            gw.polarizability = 'dTDA'
            integrals = gw.ao2mo()
            se_static = gw.build_se_static(integrals)
            gw_se_moms_h, gw_se_moms_p = gw.build_se_moments(nmom_se, integrals)
            gw_se_moms_full = np.array([gw_se_moms_h, gw_se_moms_p])
            

            gw_dc_se_moms_clus = []
            if relaxed:
                for f in fragments:
                    mc = f.get_overlap('mo|cluster')
                    gw_se_moms_clus = np.matmul(mc.T, np.matmul(gw_se_moms_full, mc))
                    gw_dc_se_moms_clus.append(gw_se_moms_clus)
            else:
                for f in fragments:
                    gw = momentGW.GW(f.hamil.to_pyscf_mf()[0].density_fit())
                    gw.polarizability = 'dTDA'
                    integrals = gw.ao2mo()
                    se_static = gw.build_se_static(integrals)
                    gw_se_moms_h_clus, gw_se_moms_p_clus = gw.build_se_moments(nmom_se, integrals)

                    gw_dc_se_moms_clus.append(np.array([gw_se_moms_h_clus, gw_se_moms_p_clus]))

            gw_dc_se_moms = make_global_one_body(self, gw_dc_se_moms_clus, symmetrize=False, use_sym=self.opts.use_sym, proj=proj, fragments=fragments)
            se_moms = se_moms + (gw_se_moms_full[:,:nmom_se] - gw_dc_se_moms[:,:nmom_se])
            self.log.info("Added non-local GW self-energy contribution")

        # for i, _ in enumerate(['h','p']):
        #     for j in range(gf_moms.shape[1]):
        #         gf_moms[i,j,:,:] = symmetrize_observable(self, gf_moms[i,j,:,:], ao=False)
        #     for j in range(se_moms.shape[1]):
        #         se_moms[i,j,:,:] = symmetrize_observable(self, se_moms[i,j,:,:], ao=False)
        

        self.gf_overlap = gf_moms[:, 0]
        self.static_self_energy = gf_moms[:, 1]
        self.self_energy_moments = se_moms

        # MBLSE for hole and particle moments
        solvers = []
        for i in range(2):
            solver = MBLSE(gf_moms[i,1], se_moms[i], hermitian=hermitian_mblse, overlap=gf_moms[i,0])
            solver.kernel()
            solvers.append(solver)

        self.solvers = solvers

        self.result = Spectral.combine_dyson(solvers[0].result, solvers[1].result)
        #self.se = result.get_self_energy()
        #self.gf = result.get_greens_function()



        return self.result.get_overlap(), self.result.get_static_self_energy(), self.result.get_self_energy()

        
    


    def qsEGF(self, *args, **kwargs):
        """Convert EGF to qsEGF"""
        self.with_scmf = QSEGF_RHF(self, *args, **kwargs)
        self.kernel = self.with_scmf.kernel



