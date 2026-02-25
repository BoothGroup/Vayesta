"""Routines to reconstruct the full system self-energy from cluster spectral moments"""

import numpy as np
import scipy

from functools import reduce

from vayesta.core.util import NotCalculatedError, Object, dot, einsum
from vayesta.core.types import SE_MomentRep
try:
    import dyson
    from dyson import MBLGF, MBLSE, AuxiliaryShift, AufbauPrinciple
    from dyson.representations import Lehmann, Spectral
    from dyson.solvers.static.chempot import search_aufbau_global
    from dyson.util.linalg import matrix_power
    dyson.quiet()
except ImportError as e:
    print(e)
    print("Dyson required for self-energy calculations")


def make_static_self_energy(
        emb,
        proj=1, 
        sym_moms=False, 
        with_mf=False, 
        use_sym=True, 
        orth_basis=False
        ):
    """
    Construct global static self-energy equal to the first Green's function moment.

    Parameters
    ----------
    emb : EWF object
        Embedding object
    proj : int
        Number of projectors to use (1 or 2)
    sym_moms : bool
        Hermitise the static self energy
    with_mf : bool
        Include the fock matrix in the static self energy
    use_sym : bool
        Use symmetry
    
    Returns
    -------
    static_self_energy : ndarray (nmo,nmo)
        Static part of self-energy (MO basis)   
    """

    fock = emb.get_fock()
    static_self_energy = np.zeros([2] + list(fock.shape), dtype=fock.dtype)

    nao, nmo = emb.mo_coeff.shape

    fragments = emb.get_fragments(sym_parent=None) if use_sym else emb.get_fragments()
    for i, f in enumerate(fragments):

        mc = f.get_overlap('mo|cluster')
        mf = f.get_overlap('mo|frag')
        fc = f.get_overlap('frag|cluster')
        cfc = fc.T @ fc

        if f.results.gf is not None:
            gf = f.results.gf

        elif f.results.se is not None:
            gf = f.results.se.to_gf_moments()
        else:
            raise Exception("Cluster has no Green's function or self-energy")
        static = gf.moments[:,1].copy()
        overlap = gf.moments[:,0].copy()
        if not with_mf:
            fock_cls = f.cluster.c_active.T @ fock @ f.cluster.c_active
            nocc = f.cluster.nocc

            
            fock_oo = np.zeros_like(fock_cls)
            fock_oo[:nocc, :nocc] = fock_cls[:nocc, :nocc]
            static[0] -= fock_oo
            
            fock_vv = np.zeros_like(fock_cls)
            fock_vv[nocc:, nocc:] = fock_cls[nocc:, nocc:]
            static[1] -= fock_vv


        if orth_basis:
            for s in range(static.shape[0]):
                orth, _ = matrix_power(overlap[s], -0.5, hermitian=gf.hermitian, return_error=False)
                static[s] = orth @ static[s] @ orth

        if proj == 1:
            static_frag = cfc @ static
            static_frag = 0.5 * (cfc @ static + static @ cfc)
            static_self_energy += mc @ static_frag @ mc.T

            if use_sym:
                for child in f.get_symmetry_children():
                    mc_child = child.get_overlap('mo|cluster')
                    static_self_energy += mc_child @ static_frag @ mc_child.T 

        elif proj == 2:
            static_frag = fc @ static @ fc.T
            static_self_energy += mf @ static_frag @ mf.T

            if use_sym:
                for child in f.get_symmetry_children():
                    mf_child = child.get_overlap('mo|frag')
                    static_self_energy += mf_child @ static_frag @ mf_child.T

    if sym_moms:
        static_self_energy = 0.5 * (static_self_energy + static_self_energy.transpose(0,2,1).conj())
    
    return static_self_energy


def make_self_energy(
        emb,
        proj=1, 
        se_mode='moments', 
        combine_sectors=None, 
        use_sym=True, 
        hermitian=None, 
        chempot_clus=None, 
        orth_basis=False, 
        project_before=False,
        non_local_se = None, 
        se_dc_mode = 'subtract', 
        ):

    """
    Construct full system self-energy moments from Green's function moments

    Parameters
    ----------
    emb : EWF object
        Embedding object
    use_sym : bool
        Use symmetry to reconstruct self-energy
    proj : int
        Number of projectors to use (1 or 2)
    se_mode : string ('lehmann', 'moments', 'moments_mblgf' or 'spectral')
        Method used to combine cluster self-energies.
    chempot_clus : string or None ('aux' or 'auf')
        Method to optimize chemical potential in cluster spectral function.
    project_before : bool 
        Whether to project self-energy before or after sector combination and/or Lanczos.
    non_local_se : string or None ('gw_rpa' or 'gw_tda')
        Whether to include non-local self-energy from full system GW calculation, and if so, which polarizability to use in the GW calculation.
    se_dc_mode : string ('subtract' or 'project')
        Method to apply double counting correction when combining embedding self-energy with full system GW self-energy. 
        'subtract' subtracts the local GW self-energy moments from the cluster self-energy moments before projection.
        'project' projects out the cluster contribution to the full GW self-energy.

    Returns
    -------
    self_energy_moms : vayesta.core.types.dynamical.SelfEnergy
        Full system self-energy (MO basis)
    """

    if se_mode not in ['lehmann', 'moments', 'moments_mblgf', 'spectral']:
        raise ValueError("Invalid self-energy construction method")

    if non_local_se is not None and non_local_se.lower() not in ['gw_rpa', 'gw_tda']:
        raise ValueError("Invalid non-local self-energy method")
    
    if se_dc_mode not in ['subtract', 'project']:
        raise ValueError("Invalid double counting correction method")
    
    if se_mode not in ['moments', 'moments_mblgf'] and non_local_se is not None:
        raise NotImplementedError("Double counting correction only implemented for moment-based self-energy construction")
    
    hermitian = emb.opts.hermitian_lanczos if hermitian is None else hermitian
    combine_sectors = emb.opts.combine_sectors_in_cluster if combine_sectors is None else combine_sectors

    if non_local_se is not None:
        if non_local_se.lower() == 'gw_rpa':
            polarizability = 'drpa'
        elif non_local_se.lower() == 'gw_tda':
            polarizability = 'dtda'
        nmom_se = emb.fragments[0].results.gf.nmom - 1
        se_nl = calc_gw_self_energy_moments(emb.mf, nmom_se=nmom_se, polarizability=polarizability)

    fragments = emb.get_fragments(sym_parent=None) if use_sym else emb.get_fragments()
    ses_mo = []
    for i, f in enumerate(fragments):
        
        mc = f.get_overlap('mo|cluster')
        mf = f.get_overlap('mo|frag')
        fc = f.get_overlap('frag|cluster')
        cfc = fc.T @ fc
        
        # Get cluster self-energy and Green's function
        if f.results.gf is not None:
            using_gf = True
            gf = f.results.gf
            se = f.results.gf.to_se_moments(orth_basis=orth_basis)
            nmom_gf = gf.nmom
            nmom_se = se.nmom
        elif f.results.se is not None:
            using_gf = False
            se = f.results.se
            gf = f.results.se.to_gf_moments()
            nmom_se = se.nmom
            nmom_gf = gf.nmom
        else:
            raise Exception("No fragment Green's function or self-energy found")

        # Build double counting correction for non-local self-energy if needed
        if non_local_se is not None and se_dc_mode == 'subtract':
            se_dc = make_fragment_double_counting_correction(f, se_mode=se_mode, double_counting=non_local_se, nmom_se=nmom_se)
        elif non_local_se is not None and se_dc_mode == 'project':
            # rotate into cluster basis
            se_dc = se_nl.rotate(mc.T)
        else:
            se_dc = None

        if hermitian:
            se = se.hermitize()
            gf = gf.hermitize()
            # hermitize se_dc for CCSD?

        if project_before:
            se = se.project(cfc, proj)
            gf = gf.project(cfc, proj)
            if non_local_se is not None:
                se_dc = se_dc.project(cfc, proj)
            
        #gf = gf.project(cfc, proj)
        #se = se.orthogonalize()
        
        # projected Lehmann representation for cluster self-energy
        if se_mode == 'lehmann':
            if combine_sectors:
                se = gf.to_spectral(hermitian=hermitian).combine_sectors(greens_function=using_gf).to_se_lehmann()
            else:
                se = gf.to_spectral(hermitian=hermitian).to_se_lehmann()

        # independent SE moments corresponding to particle and hole Green's functions
        elif se_mode == 'moments':
            if combine_sectors:
                se = se.combine_sectors(hermitian=hermitian, greens_function=using_gf)
            else:
                se = se

        # effective SE moments after performing Block Lanczos
        elif se_mode == 'moments_mblgf':

            if combine_sectors:
                spec = gf.to_spectral(hermitian=hermitian)
                spec = spec.combine_sectors(greens_function=using_gf)
                if chempot_clus is not None:
                    spec = spec.optimize_chempot(f.cluster.nocc*2, method=chempot_clus, occupancy=2)
                se = spec.to_se_moments(split=True, nmom=nmom_se)
            else:
                se = gf.to_spectral(hermitian=hermitian).to_se_moments()
                
                
        # use full spectral representation in cluster 
        elif se_mode == 'spectral':
            se = se.to_spectral(hermitian=hermitian)
            if combine_sectors:
                se.combine_sectors()

        assert se.hermitian == hermitian, "Hermiticity of self-energy does not match specified value"

        # Subtract double counting correction from cluster self-energy moments 
        if non_local_se is not None:
            se._moments -= se_dc.moments[:,:se.nmom]

        if not project_before:
            pse = se.project(cfc, proj)
        else:
            pse = se

        #assert pse.hermitian == hermitian, "Hermiticity of projected self-energy does not match specified value"

        ses_mo.append(pse.rotate(mc))
        if use_sym:
            for child in f.get_symmetry_children():
                mc_child = child.get_overlap('mo|cluster')
                ses_mo.append(pse.rotate(mc_child))


    #se = reduce(type(se).combine, ses_mo)
    se = type(se).combine(*ses_mo)

    if non_local_se is not None:
        se._moments += se_nl._moments[:,:se.nmom]

    if se_mode == 'lehmann':
        se = se.remove_degeneracies(etol=emb.opts.se_eval_tol, dtol=emb.opts.se_degen_tol)
    #assert se.hermitian == hermitian, "Hermiticity of combined self-energy does not match specified value"
    #se._static = np.diag(emb.mf.mo_energy)
    return se



def make_fragment_double_counting_correction(
        f, 
        se_mode='moments', 
        double_counting='gw_rpa',
        nmom_se = None,
    ):

    """Construct the double counting correction for a fragment when combining embedding self-energy with full system GW or CCSD self-energy.
    
    
    Parameters
    ----------
    f : Fragment object
        Fragment for which to construct double counting correction
    se_mode : string ('lehmann', 'moments', 'moments_mblgf' or 'spectral')
        Method used to construct fragment self-energy.
    double_counting : string ('gw_rpa' or 'gw_tda')
        Method to determine double counting correction when combining embedding self-energy with full system GW.

    Returns
    -------
    se_dc : 
        Double counting correction self-energy for the fragment
    """

    try:
        import momentGW
        momentGW.logging.silent = 1
    except ImportError:
        raise ImportError("momentGW is required for GW double-counting correction.")
    
    clus_mf = f.hamil.to_pyscf_mf(allow_df=True)[0]
    if double_counting.lower() == 'gw_rpa':
        polarizability = 'drpa'
    elif double_counting.lower() == 'gw_tda':
        polarizability = 'dtda'
    else:
        raise ValueError("Invalid double counting correction method")
    se_dc = calc_gw_self_energy_moments(clus_mf, nmom_se=nmom_se, polarizability=polarizability)
    
    return se_dc


def calc_gw_self_energy_moments(mf, nmom_se, polarizability='dTDA'):

    try:
        import momentGW
        momentGW.logging.silent = 1
    except ImportError:
        raise ImportError("momentGW is required for GW self-energy calculations.")
    
    gw = momentGW.GW(mf)
    gw.polarizability = polarizability
    integrals = gw.ao2mo()
    se_static = gw.build_se_static(integrals)
    seh, sep = gw.build_se_moments(nmom_se, integrals)
    se_moments = np.array([seh, sep])

    se = SE_MomentRep(se_static, se_moments, hermitian=True)
    
    return se




# def make_static_self_energy(emb, proj=1, sym_moms=False, with_mf=False, use_sym=True):
#     """
#     Construct global static self-energy equal to the first Green's function moment.

#     Parameters
#     ----------
#     emb : EWF object
#         Embedding object
#     proj : int
#         Number of projectors to use (1 or 2)
#     sym_moms : bool
#         Hermitise the static self energy
#     with_mf : bool
#         Include the fock matrix in the static self energy
#     use_sym : bool
#         Use symmetry
    
#     Returns
#     -------
#     static_self_energy : ndarray (nmo,nmo)
#         Static part of self-energy (MO basis)   
#     """

#     fock = emb.get_fock()
#     static_self_energy = np.zeros_like(fock)

#     nao, nmo = emb.mo_coeff.shape

#     fragments = emb.get_fragments(sym_parent=None) if use_sym else emb.get_fragments()
#     for i, f in enumerate(fragments):

#         mc = f.get_overlap('mo|cluster')
#         mf = f.get_overlap('mo|frag')
#         fc = f.get_overlap('frag|cluster')
#         cfc = fc.T @ fc

#         static_self_energy_clus = f.results.gf_moments[0][1] + f.results.gf_moments[1][1]

#         if not with_mf:
#             fock_cls = f.cluster.c_active.T @ fock @ f.cluster.c_active
#             static_self_energy_clus = static_self_energy_clus - fock_cls

#         if proj == 1:
#             static_self_energy_frag = cfc @ static_self_energy_clus
#             static_self_energy_frag = 0.5 * (cfc @ static_self_energy_clus + static_self_energy_clus @ cfc)
#             static_self_energy += mc @ static_self_energy_frag @ mc.T

#             if use_sym:
#                 for child in f.get_symmetry_children():
#                     mc_child = child.get_overlap('mo|cluster')
#                     static_self_energy += mc_child @ static_self_energy_frag @ mc_child.T 

#         elif proj == 2:
#             static_self_energy_frag = fc @ static_self_energy_clus @ fc.T
#             static_self_energy += mf @ static_self_energy_frag @ mf.T

#             if use_sym:
#                 for child in f.get_symmetry_children():
#                     mf_child = child.get_overlap('mo|frag')
#                     static_self_energy += mf_child @ static_self_energy_frag @ mf_child.T

#     if sym_moms:
#         static_self_energy = 0.5 * (static_self_energy + static_self_energy.T.conj())
    
#     return static_self_energy



def gf_moments_block_lanczos(gf_moments, hermitize=True, hermitian=True, sym_moms=True, shift=None, nelec=None, log=None, **kwargs):
    """
    Compute the Green's function moments from the spectral moments using the block Lanczos algorithm.

    Parameters
    ----------
    gf_moments : tuple (hole, particle) of ndarray (nmom, nmo, nmo)
        Spectral moments
    hermitian : bool
        Hermitize the Lehmann representations while preserving the GF pole energies
    hermitian : bool
        Use Hermitian block Lanczos solver
    sym_moms : bool
        Symmetrise moments
    shift : string ('None', 'aux' or 'auf')
        Method to determine filling. Auxilliary shift or Aufbau principle.
    nelec : float
        Number of electrons 
    log : Logger
        Logger object
    kwargs : dict
        Additional arguments to the block Lanczos solver

    Returns
    -------
    se, gf : tuple (Lehmann, Lehmann)
        Self-energy and Green's function in Lehmann representation
    """
    if sym_moms:
        th = gf_moments[0].copy()
        tp = gf_moments[1].copy()
        th = 0.5 * (th + th.transpose(0,2,1))
        tp = 0.5 * (tp + tp.transpose(0,2,1))
    else:
        th, tp = gf_moments[0].copy(), gf_moments[1].copy()

    solverh = MBLGF(th, hermitian=hermitian)
    solverh.kernel()
    solverp = MBLGF(tp, hermitian=hermitian)
    solverp.kernel()
    #result = Spectral.combine(solverh.result, solverp.result)
    result = Spectral.combine_from_poles(solverh.result, solverp.result, hermitize=hermitize)
    #result = Spectral.combine_dyson(solverh.result.hermitize(), solverp.result.hermitize(), ns_method='svd')

    se_static = result.get_static_self_energy()
    gf = result.get_greens_function()
    se = result.get_self_energy()

    if shift is not None:
        if nelec is None:
            raise ValueError("Number of electrons must be provided for shift")
        if shift == 'aux':
            chempot_solver = AuxiliaryShift(se_static, se, nelec, occupancy=2)
            chempot_solver.kernel()
            result = chempot_solver.result
        elif shift == 'auf':
            chempot, chempot_err  = search_aufbau_global(gf, nelec)
            result.chempot = chempot
        else:
            raise ValueError("Invalid cluster chempot optimisation method")

        


        # se.energies = se.energies - shift.chempot
        # gf.energies = gf.energies - shift.chempot
        # se.chempot = 0
        # gf.chempot = 0

        if log is not None:
            log.info("Shifted self-energy with %s"%shift.__class__.__name__)
            log.info("Chempot: %f"%se.chempot)
    
    return result

def se_moments_block_lanczos(se_static, se_moments, hermitian=True, sym_moms=True, shift=None, nelec=None, log=None, **kwargs):
    """
    Compute the Green's function moments from the spectral moments using the block Lanczos algorithm.

    Parameters
    ----------
    se_static : ndarray (nmo, nmo)
        Static part of the self-energy
    se_moments : ndarray (nmom, nmo, nmo)
        Self-energy moments
    hermitian : bool
        Use Hermitian block Lanczos solver
    sym_moms : bool
        Symmetrise moments
    shift : string ('None', 'aux' or 'auf')
        Method to determine filling. Auxilliary shift or Aufbau principle.
    nelec : float
        Number of electrons for shift
    log : Logger
        Logger object
    kwargs : dict
        Additional arguments to the block Lanczos solver

    Returns
    -------
    se, gf : tuple (Lehmann, Lehmann)
        Self-energy and Green's function in Lehmann representation
    """
    se_moments = np.array(se_moments)
    if len(se_moments.shape) == 3:
        ph_separation = False
    elif len(se_moments.shape) == 4:
        ph_separation = True
    else:
        raise ValueError("Invalid shape for self-energy moments")
    if sym_moms:
        # Use moveaxis to transpose last two axes
        se_moments = se_moments.copy()
        se_moments = 0.5 * (se_moments + np.moveaxis(se_moments, -1, -2))
    else:
        se_moments = se_moments.copy()
    
    if ph_separation:
        tp, th = se_moments[0], se_moments[1]
        solverh = MBLSE(se_static, th, hermitian=hermitian)
        solverh.kernel()
        solverp = MBLSE(se_static, tp, hermitian=hermitian)
        solverp.kernel()
        result = Spectral.combine(solverh.result, solverp.result)
    else:
        solver = MBLSE(se_static, se_moments, hermitian=hermitian)
        solver.kernel()
        result = solver.result

    gf = result.get_greens_function()
    se = result.get_self_energy()

    if shift is not None:
        if nelec is None:
            raise ValueError("Number of electrons must be provided for shift")
        if shift == 'aux':
            Shift = AuxiliaryShift
        elif shift == 'auf':
            Shift = AufbauPrinciple
        else:
            raise ValueError("Invalid cluster chempot optimisation method")
        shift = Shift(se_static, se, nelec, occupancy=2 )
        shift.kernel()
        se = shift.result.get_self_energy()
        gf = shift.result.get_greens_function()
    
    return se, gf



def make_self_energy_moments(emb, se_by_recursion=False, ph_separation=True, nmom_se=None, nmom_gf=None, hermitize=True, use_sym=True, proj=1, hermitian=True, chempot_clus=None, sym_moms=False, debug_gf_moms=None, debug_se_moms=None, subtract_local_gw=None):
    """
    Construct full system self-energy moments from Green's function moments

    Parameters
    ----------
    emb : EWF object
        Embedding object
    ph_separation : bool
        Return separate particle and hole self-energy moments
    nmom_se : int
        Number of self-energy moments to compute
    nmom_gf : int
        Number of Green's function moments to use for block Lanczos
    use_sym : bool
        Use symmetry to reconstruct self-energy
    proj : int
        Number of projectors to use (1 or 2)

    Returns
    -------
    self_energy_moms : ndarry (n_se_mom, nmo, nmo)
        Full system self-energy moments (MO basis)
    """

    fock = emb.get_fock()
    static_self_energy = np.zeros_like(fock)
    energies = []

    nao, nmo = emb.mo_coeff.shape
    dtype = emb.fragments[0].results.gf_moments[0].dtype

    if nmom_gf is None:
        nmom_gf = len(emb.fragments[0].results.gf_moments[0])
    if nmom_se is None:
        nmom_se = nmom_gf - 2

    if ph_separation:
        self_energy_moms_holes = np.zeros((nmom_se, nmo, nmo), dtype=dtype)
        self_energy_moms_parts = np.zeros((nmom_se, nmo, nmo), dtype=dtype)
    else:
        self_energy_moms = np.zeros((nmom_se, nmo, nmo), dtype=dtype)

    fragments = emb.get_fragments(sym_parent=None) if use_sym else emb.get_fragments()
    for i, f in enumerate(fragments):
        nmom_gf_max = len(f.results.gf_moments[0]), len(f.results.gf_moments[1])
        if nmom_gf is None:
            nmom_gf = nmom_gf_max
        elif type(nmom_gf) is int:
            assert nmom_gf <= nmom_gf_max[0] and nmom_gf <= nmom_gf_max[1]
            nmom_gf = (nmom_gf, nmom_gf)
        elif type(nmom_gf) is tuple:
            assert nmom_gf[0] <= nmom_gf_max[0] and nmom_gf[1] <= nmom_gf_max[1]
        
        mc = f.get_overlap('mo|cluster')
        mf = f.get_overlap('mo|frag')
        fc = f.get_overlap('frag|cluster')
        cfc = fc.T @ fc
        
        th, tp = f.results.gf_moments[0][:nmom_gf[0]], f.results.gf_moments[1][:nmom_gf[1]]

        print("nmom : ", nmom_se, nmom_gf)
        if se_by_recursion:

            nmom_seh, nmom_sep = nmom_gf[0]-2, nmom_gf[1]-2
            print("nmom_seh, nmom_sep: ", nmom_seh, nmom_sep)
            print(f.results.se_moments[0].shape)
            se_moms_clus_hole = f.results.se_moments[0][:nmom_seh]
            se_moms_clus_part = f.results.se_moments[1][:nmom_se]
            if not ph_separation:
                se_moms_clus = se_moms_clus_hole + se_moms_clus_part
            
            static_self_energy_clus = th[1] + tp[1]
        else:

            
            
            nelec = 2 * f.cluster.nocc
            mblgf_result = gf_moments_block_lanczos((th,tp), hermitize=hermitize, hermitian=hermitian, sym_moms=sym_moms, shift=chempot_clus, nelec=nelec, log=emb.log)
            gf = mblgf_result.get_greens_function()
            se = mblgf_result.get_self_energy()
            static_self_energy_clus = mblgf_result.get_static_self_energy()
            
        
            f.results.se = se
            f.results.gf = gf

            if ph_separation:
                se_moms_clus_holes = se.occupied().moment(range(nmom_se))
                se_moms_clus_parts = se.virtual().moment(range(nmom_se))
                # assert np.linalg.norm(se_moms_clus_holes.imag) < imag_tol
                # assert np.linalg.norm(se_moms_clus_parts.imag) < imag_tol
                se_moms_clus_holes = se_moms_clus_holes.real
                se_moms_clus_parts = se_moms_clus_parts.real  
            else:
                se_moms_clus = se.moment(range(nmom_se))
                #assert np.linalg.norm(se_moms_clus.imag) < imag_tol
                se_moms_clus = se_moms_clus.real

        
        imag_tol = 1e-10


        if subtract_local_gw is not None:

            try:
                import momentGW
            except ImportError:
                raise ImportError("momentGW is required for GW double-counting correction.")

            gw = momentGW.GW(f.hamil.to_pyscf_mf())
            gw.polarizability = 'dTDA'
            integrals = gw.ao2mo()
            se_static = gw.build_se_static(integrals)
            gw_se_clus_moms_holes, gw_se_clus_moms_parts = gw.build_se_moments(nmom_se, integrals)

            if ph_separation:
                se_moms_clus_holes -= gw_se_clus_moms_holes
                se_moms_clus_parts -= gw_se_clus_moms_parts
            else:
                se_moms_clus -= (gw_se_clus_moms_holes + gw_se_clus_moms_parts)


        if proj == 1:
            if ph_separation:
                se_moms_frag_holes = np.array([0.5*(cfc @ mom + mom @ cfc) for mom in se_moms_clus_holes])
                se_moms_frag_parts = np.array([0.5*(cfc @ mom + mom @ cfc) for mom in se_moms_clus_parts])
                self_energy_moms_holes += np.array([mc @ mom @ mc.T for mom in se_moms_frag_holes])
                self_energy_moms_parts += np.array([mc @ mom @ mc.T for mom in se_moms_frag_parts])
            else:
                
                se_moms_frag = np.array([0.5*(cfc @ mom + mom @ cfc) for mom in se_moms_clus])
                # assert np.linalg.norm(se_moms_frag.imag) < imag_tol
                se_moms_frag = se_moms_frag.real
                self_energy_moms += np.array([mc @ mom @ mc.T for mom in se_moms_frag])

            if use_sym:
                for child in f.get_symmetry_children():
                    mc_child = child.get_overlap('mo|cluster')
                    if ph_separation:
                        self_energy_moms_holes += np.array([mc_child @ mom @ mc_child.T for mom in se_moms_frag_holes])
                        self_energy_moms_parts += np.array([mc_child @ mom @ mc_child.T for mom in se_moms_frag_parts])
                    else:
                        self_energy_moms += np.array([mc_child @ mom @ mc_child.T for mom in se_moms_frag])

        elif proj == 2:
            if ph_separation:
                se_moms_frag_holes = np.array([(fc @ mom @ fc.T) for mom in se_moms_clus_holes])
                se_moms_frag_parts = np.array([(fc @ mom @ fc.T) for mom in se_moms_clus_parts])
                self_energy_moms_holes += np.array([mf @ mom @ mf.T for mom in se_moms_frag_holes])
                self_energy_moms_parts += np.array([mf @ mom @ mf.T for mom in se_moms_frag_parts])
            else:
                se_moms_frag = [(fc @ mom @ fc.T) for mom in se_moms_clus]
                self_energy_moms += np.array([mf @ mom @ mf.T for mom in se_moms_frag])

            if use_sym:
                for child in f.get_symmetry_children():
                    mf_child = child.get_overlap('mo|frag')
                    if ph_separation:
                        self_energy_moms_holes += np.array([mf_child @ mom @ mf_child.T for mom in se_moms_frag_holes])
                        self_energy_moms_parts += np.array([mf_child @ mom @ mf_child.T for mom in se_moms_frag_parts])
                    else:
                        self_energy_moms += np.array([mf_child @ mom @ mf_child.T for mom in se_moms_frag])

        # Static self-energy (correlated part)
        # if True:
        #     fock_cls = f.cluster.c_active.T @ fock @ f.cluster.c_active
        #     static_self_energy_clus = static_self_energy_clus - fock_cls

        # if proj == 1:
        #     static_self_energy_frag = cfc @ static_self_energy_clus
        #     static_self_energy_frag = 0.5 * (cfc @ static_self_energy_clus + static_self_energy_clus @ cfc)
        #     static_self_energy += mc @ static_self_energy_frag @ mc.T

        #     if use_sym:
        #         for child in f.get_symmetry_children():
        #             mc_child = child.get_overlap('mo|cluster')
        #             static_self_energy += mc_child @ static_self_energy_frag @ mc_child.T 

        # elif proj == 2:
        #     static_self_energy_frag = fc @ static_self_energy_clus @ fc.T
        #     static_self_energy += mf @ static_self_energy_frag @ mf.T

        #     if use_sym:
        #         for child in f.get_symmetry_children():
        #             mf_child = child.get_overlap('mo|frag')
        #             static_self_energy += mf_child @ static_self_energy_frag @ mf_child.T

    # if sym_moms:
    #     static_self_energy = 0.5 * (static_self_energy + static_self_energy.T.conj())

    if sym_moms:
        if ph_separation:
            self_energy_moms_holes = 0.5 * (self_energy_moms_holes + self_energy_moms_holes.transpose(0,2,1).conj())
            self_energy_moms_parts = 0.5 * (self_energy_moms_parts + self_energy_moms_parts.transpose(0,2,1).conj())
        else:
            self_energy_moms = 0.5 * (self_energy_moms + self_energy_moms.transpose(0,2,1).conj()) 

    if ph_separation:
        return (self_energy_moms_holes, self_energy_moms_parts)
    else:
        return  self_energy_moms

            
def remove_fragments_from_full_moments(emb, se_moms, proj=2, use_sym=False):
    """
    Remove the embedding contribution from a set of full system self-energy moments.
    Useful to combine embedding with full system GW or CCSD calculations and avoid double counting.

    Parameters
    ----------
    emb : EWF object
        Embedding object
    se_moms : ndarray (n_se_mom, nmo, nmo)
        Full system self-energy moments (MO basis)
    proj : int
        Number of projectors used to construct the self-energy moments.
        Should be consistent with the number of projectors used in the embedding.
    
    Returns
    -------
    corrected_moms : ndarray (n_se_mom, nmo, nmo)
        Self-energy moments with the embedding contributions removed. (MO basis)
    """
    corrected_moms = se_moms.copy() if proj == 2 else np.zeros_like(se_moms)
    fragments = emb.get_fragments(sym_parent=None) if use_sym else emb.get_fragments()
    for i, f in enumerate(fragments):
        mc = f.get_overlap('mo|cluster')
        fc = f.get_overlap('frag|cluster')
        fm = f.get_overlap('frag|mo')
        mfm = fm.T @ fm
        mcm = mc @ mc.T
        if proj == 1:
            for i2, f2 in enumerate(fragments):
                if i2 == i:
                    continue
                mc2 = f2.get_overlap('mo|cluster')
                fm2 = f2.get_overlap('frag|mo')
                mcm2 = mc2 @ (mc2.T) - mcm   
                corrected_moms += np.array([mfm @ mom @ mcm2 for mom in se_moms])
        elif proj == 2:
            corrected_moms -= np.array([mfm @ mom @ mfm for mom in se_moms])
    return corrected_moms


def make_self_energy_1proj_(emb, 
                            hermitian=False,
                            use_sym=True, 
                            hermitize=True, 
                            sym_moms=False, 
                            use_svd=True,
                            nmom_gf=None,
                            chempot_clus=False,
                            remove_degeneracy=True,
                            img_space=False,
                            se_degen_tol=1e-4,
                            se_eval_tol=1e-6,
                            drop_non_causal=False):
    fock = emb.get_fock()
    static_self_energy = np.zeros_like(fock, dtype=np.complex128)
    energies = []
    if use_svd:
        couplings_l, couplings_r = [], []
    else:
        couplings = []
    fragments = emb.get_fragments(sym_parent=None) if use_sym else emb.get_fragments()
    for i, f in enumerate(fragments):
        nmom_gf_max = len(f.results.gf_moments[0]), len(f.results.gf_moments[1])
        if nmom_gf is None:
            nmom_gf = nmom_gf_max
        elif type(nmom_gf) is int:
            assert nmom_gf <= nmom_gf_max[0] and nmom_gf <= nmom_gf_max[1]
            nmom_gf = (nmom_gf, nmom_gf)
        elif type(nmom_gf) is tuple:
            assert nmom_gf[0] <= nmom_gf_max[0] and nmom_gf[1] <= nmom_gf_max[1]
            
        th, tp = f.results.gf_moments[0][:nmom_gf[0]], f.results.gf_moments[1][:nmom_gf[1]]

        nelec = 2 * f.cluster.nocc
        mblgf_result = gf_moments_block_lanczos((th,tp), hermitize=hermitize, hermitian=hermitian, sym_moms=sym_moms, shift=chempot_clus, nelec=nelec, log=emb.log)

        gf = mblgf_result.get_greens_function()
        se = mblgf_result.get_self_energy()
        static_self_energy_clus = mblgf_result.get_static_self_energy()

        dm = gf.occupied().moment(0) * 2
        nelec = np.trace(dm)
        emb.log.info("Fragment %s: nelec: %f target: %f"%(f.id, nelec, f.nelectron))
        emb.log.info("Cluster couplings shape: %s %s"%se.unpack_couplings()[0].shape)

        mc = f.get_overlap('mo|cluster')
        fc = f.get_overlap('frag|cluster')
        cfc = fc.T @ fc


        static_self_energy_frag = cfc @ static_self_energy_clus @ cfc.T.conj()
        static_self_energy += mc @ static_self_energy_frag @ mc.T.conj()
    
    return static_self_energy

            


        
    

def make_self_energy_1proj(emb, hermitian=True, use_sym=True, hermitize=True, sym_moms=False, use_svd=True, nmom_gf=None, chempot_clus=False, remove_degeneracy=True, img_space=True, se_degen_tol=1e-4, se_eval_tol=1e-6, drop_non_causal=False):
    """
    Construct full system self-energy in Lehmann representation from cluster spectral moments using 1 projector

    TODO: MPI, SVD

    Parameters
    ----------
    emb : EWF object
        Embedding object
    hermitian : bool
        Use Hermitian block Lanczos solver
    remove_degeneracy : bool
        Combine degenerate poles in full system self-energy
    nmom_gf : int
        Number of Green's function moments to use for block Lanczos
    use_svd : bool
        Use SVD to decompose the self-energy as outer product
    img_space : bool
        Use image space for SVD or diagonalisation
    use_sym : bool
        Use symmetry to reconstruct self-energy
    sym_moms : bool
        Symmetrise moments
    se_degen_tol : float
        Tolerance for degeneracy in Lehmann representation
    se_eval_tol : float
        Tolerance for self-energy eigenvalues assumed to be kept
    drop_non_causal : bool
        Drop non-causal poles (negative eigenvalues) of self-energy

    Returns
    -------
    self_energy : Lehmann object
        Reconstructed self-energy in Lehmann representation (MO basis)
    """
    fock = emb.get_fock()
    static_self_energy = np.zeros_like(fock, dtype=np.complex128)
    energies = []
    if use_svd:
        couplings_l, couplings_r = [], []
    else:
        couplings = []
    fragments = emb.get_fragments(sym_parent=None) if use_sym else emb.get_fragments()
    for i, f in enumerate(fragments):
        nmom_gf_max = len(f.results.gf_moments[0]), len(f.results.gf_moments[1])
        if nmom_gf is None:
            nmom_gf = nmom_gf_max
        elif type(nmom_gf) is int:
            assert nmom_gf <= nmom_gf_max[0] and nmom_gf <= nmom_gf_max[1]
            nmom_gf = (nmom_gf, nmom_gf)
        elif type(nmom_gf) is tuple:
            assert nmom_gf[0] <= nmom_gf_max[0] and nmom_gf[1] <= nmom_gf_max[1]
            
        th, tp = f.results.gf_moments[0][:nmom_gf[0]], f.results.gf_moments[1][:nmom_gf[1]]

        nelec = 2 * f.cluster.nocc
        mblgf_result = gf_moments_block_lanczos((th,tp), hermitize=hermitize, hermitian=hermitian, sym_moms=sym_moms, shift=chempot_clus, nelec=nelec, log=emb.log)
        gf = mblgf_result.get_greens_function()
        se = mblgf_result.get_self_energy()
        static_self_energy_clus = mblgf_result.get_static_self_energy()
        ovlp = mblgf_result.get_overlap()
        norm = np.linalg.norm(ovlp - np.eye(ovlp.shape[0]))
        print("Norm ovlp: %s " %norm )
        
        
        # f.results.se = se
        # f.results.gf = gf

        dm = gf.occupied().moment(0) * 2
        nelec = np.trace(dm)
        emb.log.info("Fragment %s: nelec: %f target: %f"%(f.id, nelec, f.nelectron))
        emb.log.info("Cluster couplings shape: %s %s"%se.unpack_couplings()[0].shape)

        mc = f.get_overlap('mo|cluster')
        fc = f.get_overlap('frag|cluster')
        cfc = fc.T @ fc
        
        # Dynamic self-energy
        coup_l, coup_r = se.unpack_couplings()
        p_coup_l, p_coup_r = cfc @ coup_l, cfc @ coup_r
        sym_coup = 0.5*(einsum('pa,qa->apq', cfc @ coup_l , coup_r.conj()) + einsum('pa,qa->apq', coup_l , cfc @ coup_r.conj()))
        #sym_coup = 0.5*(einsum('pa,qa->apq', cfc @ coup_l.conj() , coup_r) + einsum('pa,qa->apq', coup_l.conj() , cfc @ coup_r))
        mat = sym_coup.sum(axis=0)
        if use_svd:
            energies_frag, couplings_l_frag, couplings_r_frag = project_1_to_fragment_svd(cfc, se, img_space=img_space, tol=1e-12)
            couplings_l.append(mc @ couplings_l_frag)
            couplings_r.append(mc @ couplings_r_frag)
            energies.append(energies_frag)
            
            mat2 = np.einsum('pa,qa->pq', couplings_l_frag, couplings_r_frag.conj())
            
        else:
            energies_frag, couplings_frag = project_1_to_fragment_eig(cfc, se, img_space=img_space, tol=1e-6)
            couplings.append(mc @ couplings_frag)
            energies.append(energies_frag)
            mat2 = np.einsum('pa,qa->pq', couplings_frag, couplings_frag.conj())

        emb.log.info("Norm diff of SE numerator %s"%np.linalg.norm(mat - mat2))
        if use_sym:
            for child in f.get_symmetry_children():
                mc_child = child.get_overlap('mo|cluster')
                energies.append(energies_frag)
                if use_svd:
                    couplings_l.append(mc_child @ couplings_l_frag)
                    couplings_r.append(mc_child @ couplings_r_frag)
                else:
                    couplings.append(mc_child @ couplings_frag)


        # Static self-energy (correlated part)
        proj = 1
        if True:
            fock_cls = f.cluster.c_active.T @ fock @ f.cluster.c_active
            static_self_energy_clus = static_self_energy_clus - fock_cls

        if proj == 1:
            static_self_energy_frag = cfc @ static_self_energy_clus
            static_self_energy_frag = 0.5 * (cfc @ static_self_energy_clus + static_self_energy_clus @ cfc)
            static_self_energy += mc @ static_self_energy_frag @ mc.T

            if use_sym:
                for child in f.get_symmetry_children():
                    mc_child = child.get_overlap('mo|cluster')
                    static_self_energy += mc_child @ static_self_energy_frag @ mc_child.T 

        elif proj == 2:
            static_self_energy_frag = fc @ static_self_energy_clus @ fc.T
            static_self_energy += mf @ static_self_energy_frag @ mf.T

            if use_sym:
                for child in f.get_symmetry_children():
                    mf_child = child.get_overlap('mo|frag')
                    static_self_energy += mf_child @ static_self_energy_frag @ mf_child.T

    if sym_moms:
        static_self_energy = 0.5 * (static_self_energy + static_self_energy.T.conj())

    energies = np.concatenate(energies)
    if use_svd:
        couplings = np.array([np.hstack(couplings_l), np.hstack(couplings_r)])
    else:
        couplings = np.hstack(couplings)

    

    self_energy = Lehmann(energies, couplings)
    emb.log.info("Removing SE degeneracy. Naux:    %s"%len(energies))

    if remove_degeneracy:
        if hermitian:
            self_energy = remove_se_degeneracy_sym(self_energy, dtol=se_degen_tol, etol=se_eval_tol, drop_non_causal=drop_non_causal, log=emb.log)
        else:
            self_energy = remove_se_degeneracy_nsym(self_energy, dtol=se_degen_tol, etol=se_eval_tol,  log=emb.log)

    emb.log.info("Removed SE degeneracy, new Naux: %s"%len(self_energy.energies)) 
    return static_self_energy, self_energy
        
def project_1_to_fragment_eig(cfc, se, hermitize=False, img_space=True, tol=1e-6):
    """
    DEPRECATED: Use SVD instead of eigenvalue decomposition

    Symmetrically project self-energy couplings to fragment as 0.5 * (PV V^T _ V PV^T) and rewrite as an outer product via diagonalization

    Parameters
    ---------- 
    cfc : ndarray (nclus, nclus)
        Fragment projector in cluster basis
    se : Lehmann object 
        Cluster self-energy
    img_space : bool
        Perform SVD in the image space
    tol : float
        Tolerance for self-energy singular values assumed to be kept
    
    Returns
    -------
    energies_frag : ndarray (naux_new)
        Energies of the fragment couplings
    couplings_l_frag : ndarray (nmo, naux_new)
        Left couplings of the fragment projected self-energy
    """
    coup_l, coup_r = se.unpack_couplings()
    if hermitize:
        coup = 0.5*(coup_l + coup_r)
    else:
        coup = coup_l
    p_coup = cfc @ coup
    sym_coup = 0.5*(einsum('pa,qa->apq', p_coup , coup.conj()) + einsum('pa,qa->apq', coup , p_coup.conj()))
    nmo, naux = coup.shape
    couplings_frag, energies_frag = [], []
    for a in range(naux):

        if img_space:
            vs = np.vstack([p_coup[:,a], coup[:,a]])
            ws = np.vstack([coup[:,a], p_coup[:,a]])
            val, w = eig_outer_sum(vs, ws, tol=tol, fac=0.5)
        else:
            vs = np.vstack([p_coup[:,a], coup[:,a]])
            ws = np.vstack([coup[:,a], p_coup[:,a]])
            val, w = eig_outer_sum_slow(vs, ws, tol=tol, fac=0.5)
        w = w[:, val > tol]

        if w.shape[0] != 0:
            couplings_frag.append(w)
            energies_frag += [se.energies[a] for e in range(w.shape[1])]

        mat = np.einsum('pa,qa->pq', w, w)
        norm = np.linalg.norm(mat - sym_coup[a])
    return np.array(energies_frag), np.hstack(couplings_frag)

def project_1_to_fragment_svd(cfc, se, img_space=True, tol=1e-6):
    """
    Symmetrically project self-energy couplings to fragment as 0.5 * (PV V^T _ V PV^T) and rewrite as an outer product via SVD

    Parameters
    ---------- 
    cfc : ndarray (nclus, nclus)
        Fragment projector in cluster basis
    se : Lehmann object 
        Cluster self-energy
    img_space : bool
        Perform SVD in the image space
    tol : float
        Tolerance for self-energy singular values assumed to be kept
    
    Returns
    -------
    energies_frag : ndarray (naux_new)
        Energies of the fragment couplings
    couplings_l_frag : ndarray (nmo, naux_new)
        Left couplings of the fragment projected self-energy
    couplings_r_frag : ndarray (nmo, naux_new)
        Right couplings of the fragment projected self-energy
    """

    coup_l, coup_r = se.unpack_couplings()
    p_coup_l, p_coup_r = cfc @ coup_l, cfc @ coup_r

    nmo, naux = coup_l.shape
    couplings_l_frag, couplings_r_frag, energies_frag = [], [], []

    
    for a in range(naux):
        m = 0.5 * (np.outer(p_coup_l[:,a], coup_r[:,a].conj()) + np.outer(coup_l[:,a], p_coup_r[:,a].conj()))
        
        if img_space:
            vs = np.vstack([p_coup_l[:,a], coup_l[:,a]])
            ws = np.vstack([coup_r[:,a], p_coup_r[:,a]])
            u, s, v = svd_outer_sum(vs, ws, tol=tol, fac=0.5)
        else:
            vs = np.vstack([p_coup_l[:,a], coup_l[:,a]])
            ws = np.vstack([coup_r[:,a], p_coup_r[:,a]])
            u, s, v = svd_outer_sum_slow(vs, ws, tol=tol, fac=0.5)

        if u.shape[0] != 0:   
            couplings_l_frag.append(u)
            couplings_r_frag.append(v)
        energies_frag += [se.energies[a] for e in range(u.shape[1])]

        # TODO DEBUG CASE WHERE COUPLINGS ARE EMPTY!
        # print("--------------------------------------------------------------------------------\n\n")
        # print([x.shape for x in couplings_l_frag])
        # print([x.shape for x in couplings_r_frag])
    return np.array(energies_frag), np.hstack(couplings_l_frag), np.hstack(couplings_r_frag)


    
def make_self_energy_2proj(emb, nmom_gf=None, hermitize=True, hermitian=True, sym_moms=False, remove_degeneracy=True, use_sym=True, chempot_clus=None, se_degen_tol=1e-4, se_eval_tol=1e-6):
    """
    Construct full system self-energy in Lehmann representation from cluster spectral moments using 2 projectors

    TODO: MPI, SVD

    Parameters
    ----------
    emb : EWF object
        Embedding object
    nmom_gf : int
        Number of Green's function moments to use for block Lanczos
    sym_moms : bool
        Symmetrise moments
    hermitian : bool
        Use Hermitian block Lanczos solver
    remove_degeneracy : bool
        Combine degenerate poles in full system self-energy
    use_sym : bool
        Use symmetry to reconstruct self-energy

    Returns
    -------
    self_energy : Lehmann object
        Reconstructed self-energy in Lehmann representation (MO basis)
    """
    fock = emb.get_fock()
    static_self_energy = np.zeros_like(fock, dtype=np.complex128)

    couplings, energies = [], []

    fragments = emb.get_fragments(sym_parent=None) if use_sym else emb.get_fragments()
    for i, f in enumerate(fragments):
        nmom_gf_max = len(f.results.gf_moments[0]), len(f.results.gf_moments[1])
        if nmom_gf is None:
            nmom_gf = nmom_gf_max
        elif type(nmom_gf) is int:
            assert nmom_gf <= nmom_gf_max[0] and nmom_gf <= nmom_gf_max[1]
            nmom_gf = (nmom_gf, nmom_gf)
        elif type(nmom_gf) is tuple:
            assert nmom_gf[0] <= nmom_gf_max[0] and nmom_gf[1] <= nmom_gf_max[1]
            
        th, tp = f.results.gf_moments[0][:nmom_gf[0]], f.results.gf_moments[1][:nmom_gf[1]]
        se_static = th[1] + tp[1]
        
        nelec = 2 * f.cluster.nocc
        mblgf_result = gf_moments_block_lanczos((th,tp), hermitize=True, hermitian=hermitian, sym_moms=sym_moms, shift=chempot_clus, nelec=nelec, log=emb.log)
        se = mblgf_result.get_self_energy()
        gf = mblgf_result.get_greens_function()
        static_self_energy_clus = mblgf_result.get_static_self_energy()
        
        # f.results.se = se
        # f.results.gf = gf

        dm = gf.occupied().moment(0) * 2
        nelec = np.trace(dm)
        emb.log.info("Fragment %s: nelec: %f target: %f"%(f.id, nelec, f.nelectron))

        mf = f.get_overlap('mo|frag')
        fc = f.get_overlap('frag|cluster')
        cfc = fc.T @ fc
        mc = f.get_overlap('mo|cluster')

        
        # Dynamic self-energy
        if type(se.couplings) is tuple:
            couplings_l, couplings_r = se.couplings
            couplings_l = mf @ fc @ couplings_l
            couplings_r = mf @ fc @ couplings_r
            couplings.append((couplings_l, couplings_r))
        else:
            couplings.append(mf @ fc @ se.couplings)
        energies.append(se.energies)

        if use_sym:
            for child in f.get_symmetry_children():
                mf_child = child.get_overlap('mo|frag')
                fc_child = child.get_overlap('frag|cluster')
                if type(se.couplings) is tuple:
                    couplings_l, couplings_r = se.couplings
                    couplings_l = mf_child @ fc_child @ couplings_l
                    couplings_r = mf_child @ fc_child @ couplings_r
                    couplings.append((couplings_l, couplings_r))
                else:
                    couplings.append(mf_child @ fc_child @ se.couplings)

                energies.append(se.energies)
        
        # Static self-energy (correlated part)
        proj = 1
        if True:
            fock_cls = f.cluster.c_active.T @ fock @ f.cluster.c_active
            static_self_energy_clus = static_self_energy_clus - fock_cls

        if proj == 1:
            static_self_energy_frag = cfc @ static_self_energy_clus
            static_self_energy_frag = 0.5 * (cfc @ static_self_energy_clus + static_self_energy_clus @ cfc)
            static_self_energy += mc @ static_self_energy_frag @ mc.T

            if use_sym:
                for child in f.get_symmetry_children():
                    mc_child = child.get_overlap('mo|cluster')
                    static_self_energy += mc_child @ static_self_energy_frag @ mc_child.T 

        elif proj == 2:
            static_self_energy_frag = fc @ static_self_energy_clus @ fc.T
            static_self_energy += mf @ static_self_energy_frag @ mf.T

            if use_sym:
                for child in f.get_symmetry_children():
                    mf_child = child.get_overlap('mo|frag')
                    static_self_energy += mf_child @ static_self_energy_frag @ mf_child.T

    if sym_moms:
        static_self_energy = 0.5 * (static_self_energy + static_self_energy.T.conj())

    if type(couplings[0]) is tuple:
        couplings_l, couplings_r = zip(*couplings)
        couplings = np.array([np.hstack(couplings_l), np.hstack(couplings_r)])
    else:
        couplings = np.hstack(couplings)
    energies = np.concatenate(energies)
    self_energy = Lehmann(energies, couplings)
    if remove_degeneracy:
        if hermitian:
            self_energy = remove_se_degeneracy_sym(self_energy, dtol=se_degen_tol, etol=se_eval_tol, drop_non_causal=False, log=emb.log)
        else:
            self_energy = remove_se_degeneracy_nsym(self_energy, dtol=se_degen_tol, etol=se_eval_tol,  log=emb.log)
    return static_self_energy, self_energy

def drop_and_reweight(se, tol=1e-12):
    couplings_l, couplings_r = se.unpack_couplings()
    nmo, naux = couplings_l.shape
    weights = np.einsum('pa,pa->pa', couplings_l, couplings_r.conj())
    energies = se.energies


    idx_p, idx_a = np.nonzero(weights < 0) # indices p, a of non-causal poles
    new_energies, new_couplings = [], []
    
    for a in range(naux):
        mat = np.einsum('p,q->pq', couplings_l[:,a], couplings_r[:,a].conj())
        mat = 0.5 * (mat + mat.T)
        val, vec = np.linalg.eigh(mat)
        idx = val>tol
        w = vec[:,idx] @ np.diag(np.sqrt(val[idx]))
        new_couplings.append(w)
        new_energies += [energies[a] for _ in range(idx.sum())]

    new_energies, new_couplings = np.array(new_energies), np.hstack(new_couplings)

    new_weights = np.einsum('pa,pa->pa', new_couplings, new_couplings.conj())

    scale_factor2 = weights.sum(axis=1) / new_weights.sum(axis=1)
    scale_factor = np.sqrt(scale_factor2)
    new_couplings = new_couplings * scale_factor[:,None]

    return Lehmann(new_energies, new_couplings)

def merge_non_causal_poles(se, weight_tol=1e-12):
    # TODO check, fix for dense non-causal poles
    U, V = se.unpack_couplings()
    nmo, naux = U.shape
    weights = np.einsum('pa,pa->pa', U, V)
    es = se.energies


    idx_p, idx_a = np.nonzero(weights < 0) # indices p, a of non-causal poles
    energies, couplings = [], []
    a = 0
    while a < naux:
        if a not in idx_a and a+1 not in idx_a:
            mat = np.einsum('p,q->pq', U[:,a], V[:,a])
            mat = 0.5 * (mat + mat.T) # Possibly uncecassary, symmetric by construction?
            val, vec = np.linalg.eigh(mat)
            idx = val>weight_tol
            w = vec[:,idx] @ np.diag(np.sqrt(val[idx]))
            couplings.append(w)
            energies += [es[a] for _ in range(idx.sum())]
            a += 1
        elif a+1 in idx_a:
            # sum over poles until next causal pole reached
            b = a+1
            while b in idx_a:
                b += 1
            mat = np.einsum('pa,qa->pq', U[:,a:b], V[:,a:b])
            mat = 0.5 * (mat + mat.T)
            val, vec = np.linalg.eigh(mat)
            idx = val>weight_tol
            w = vec[:,idx] @ np.diag(np.sqrt(val[idx]))
            couplings.append(w)
            pole_weighting = weights[:,a:b].sum(axis=0)
            new_energy = (es[a:b] * pole_weighting).sum() / pole_weighting.sum()
            energies += [new_energy for _ in range(idx.sum())]
            a = b+1
        else:
            raise Exception()
            a+=1

    return Lehmann(np.array(energies), np.hstack(couplings))

def eig_outer_sum(vs, ws, tol=1e-12, fac=1):
    """
    Calculate the eigenvalues and eigenvectors of the sum of symmetrised outer products.
    Given lists of vectors vs and ws, the function calcualtes the eigendecomposition
    of the matrix fac*(sum_i outer(vs[i], ws[i]) + outer(ws[i], vs[i])) working in
    the image space spanned by the input vectors.

    Parameters
    ----------
    vs : np.ndarray (n,N)
        List of n vectors of length N
    ws : np.ndarray (n,N)
        List of n vectors of length N
    tol : float
        Tolerance for eigenvalues to be kept
    fac : float
        Scaling factor for the outer products

    Returns
    -------
    val : np.ndarray (n,)
        Eigenvalues 
    vec : np.ndarray
        Eigenvectors (N,n)
    """
    mat = np.einsum('pa,qa->pq', vs,ws.conj())
    vs, ws = np.array(vs), np.array(ws)
    assert vs.shape == ws.shape
    rank = 2 * vs.shape[0]
    N = vs.shape[1]
    left, right = np.zeros((rank, N), dtype=mat.dtype), np.zeros((N, rank), dtype=mat.dtype)
    for i in range(len(vs)):
        left[2*i] = ws[i]
        left[2*i+1] = vs[i]
        right[:,2*i] = vs[i]
        right[:,2*i+1] = ws[i]
    mat = fac * (left @ right)
    val, vec = np.linalg.eig(mat)
    #assert np.allclose(val.imag, 0)
    #assert np.allclose(vec.imag, 0)
    val = val.real
    idx = np.abs(val) > tol
    val, vec = val[idx], vec[:,idx]
    idx = val.argsort()
    val, vec = val[idx], vec[:,idx]
    vec = right @ vec.real
    vec = vec / np.linalg.norm(vec, axis=0)
    v = vec @ np.diag(np.sqrt(val, dtype=np.complex128))
    return val, v

def svd_outer_sum(vs, ws, tol=1e-12, fac=1):
    """
    Calculate the singular value decomposition of the sum of outer products.
    Given lists of vectors vs and ws, the function calcualtes the SVD of the
    matrix fac * sum_i outer(vs[i], ws[i].conj() working in the image space 
    spanned by the input vectors.
        
    Parameters
    ----------
    vs : np.ndarray (n,N)
        List of n vectors of length N
    ws : np.ndarray (n,N)
        List of n vectors of length N
    tol : float
        Tolerance for singular values to be kept
    fac : float
        Scaling factor for the outer products
    
    Returns
    -------
    u : np.ndarray (N,n)
        Left singular vectors
    s : np.ndarray (n,)
        Singular values
    v : np.ndarray (N,n)
        Right singular vectors
    """
    rank = 2
    #basis_l = np.vstack([p_coup_l[:,a], coup_l[:,a]]).T
    basis_l = np.vstack(vs).T
    dbasis_l = np.linalg.pinv(basis_l)
    #basis_r = np.vstack([coup_r[:,a], p_coup_r[:,a]]).T
    basis_r = np.vstack(ws).T
    dbasis_r = np.linalg.pinv(basis_r)
    
    mat = fac * (basis_r.conj().T @ basis_r) # FIX ME
    
    U, s, Vh = np.linalg.svd(mat)
    idx = s > tol
    #assert idx.sum() <= 2 # Rank at most 2
    s = s[idx]
    u = basis_l @ U[:,idx] @ np.diag(np.sqrt(s))
    v = (np.diag(np.sqrt(s)) @ Vh[idx,:] @ dbasis_r).conj().T

    return u, s, v

    
def eig_outer_sum_slow(vs, ws, tol=1.e-10, fac=1):
    """
    Calculate the eigenvalues and eigenvectors of the sum of symmetrised outer products.
    Given lists of vectors vs and ws, the function calcualtes the eigendecomposition
    of the matrix fac*(sum_i outer(vs[i], ws[i]) + outer(ws[i], vs[i])) working in
    the full space.

    Parameters
    ----------
    vs : np.ndarray (n,N)
        List of n vectors of length N
    ws : np.ndarray (n,N)
        List of n vectors of length N
    tol : float
        Tolerance for eigenvalues to be kept
    fac : float
        Scaling factor for the outer products

    Returns
    -------
    val : np.ndarray (n,)
        Eigenvalues 
    vec : np.ndarray
        Eigenvectors (N,n)
    """
    #outer = 0.5*(np.einsum('ai,aj->ij', vs, ws) + np.einsum('ai,aj->ij', ws, vs))
    outer = fac * (np.tensordot(vs, ws, axes=([0],[0])) + np.tensordot(ws, vs, axes=([0],[0])))
    val, vec = np.linalg.eigh(outer)
    assert np.allclose(val.imag, 0)
    val = val.real
    idx = np.abs(val) > tol
    val, vec = val[idx], vec[:,idx]
    idx = val.argsort()
    val, vec = val[idx], vec[:,idx]
    v = vec @ np.diag(np.sqrt(val))
    return val, v

def svd_outer_sum_slow(vs, ws, tol=1e-12, fac=1):
    """
    Calculate the singular value decomposition of the sum of outer products.
    Given lists of vectors vs and ws, the function calcualtes the SVD of the
    matrix fac * sum_i outer(vs[i], ws[i].conj() working in the full space.
        
    Parameters
    ----------
    vs : np.ndarray (n,N)
        List of n vectors of length N
    ws : np.ndarray (n,N)
        List of n vectors of length N
    tol : float
        Tolerance for singular values to be kept
    fac : float
        Scaling factor for the outer products
    
    Returns
    -------
    u : np.ndarray (N,n)
        Left singular vectors
    s : np.ndarray (n,)
        Singular values
    v : np.ndarray (N,n)
        Right singular vectors
    """
    assert len(ws) == len(vs)
    outer = fac*np.einsum('ap,aq->pq', vs, ws.conj())
    U, s, Vh = np.linalg.svd(outer)
    idx = np.abs(s) > tol
    s = s[idx]
    u = U[:,idx] @ np.diag(np.sqrt(s))
    v = Vh.conj().T[:,idx] @ np.diag(np.sqrt(s))
    return u, s, v


def remove_se_degeneracy_sym(se, dtol=1e-8, etol=1e-6, drop_non_causal=False, img_space=False, log=None):
    if log is not None:
        log.info("Removing degeneracy in self-energy - degenerate energy tol=%e   evec tol=%e"%(dtol, etol))
    e = se.energies
    couplings_l, couplings_r = se.unpack_couplings()
    e_new, slices = get_unique(e, atol=dtol)#
    if log is not None:
        log.info("Number of energies = %d,  unique = %d"%(len(e),len(e_new)))
    energies, couplings = [], []
    warn_non_causal = False
    for i, s in enumerate(slices):
        mat = np.einsum('pa,qa->pq', couplings_l[:,s], couplings_r[:,s]).real
        #print("Hermitian: %s"%np.linalg.norm(mat - mat.T.conj()))
        if img_space:
            # TODO FIX ME - need to track non causal poles correctly and ensure maximal cancellation
            val, w = eig_outer_sum(couplings_l[:,s].T, couplings_r[:,s].T, tol=etol, fac=0.5)
            idx = val > etol if  drop_non_causal else np.abs(val) > etol
            if np.sum(val[idx] < -etol) > 0:
                warn_non_causal = True
            w = w[:,idx]
        else:
            mat = 0.5 * (mat + mat.T.conj())
            assert np.allclose(mat, mat.T.conj())
            val, vec = np.linalg.eigh(mat)
            idx = val > etol if  drop_non_causal else np.abs(val) > etol
            if np.sum(val[idx] < -etol) > 0:
                warn_non_causal = True
            w = vec[:,idx] @ np.diag(np.sqrt(val[idx], dtype=np.complex128))
        couplings.append(w)
        energies += [e_new[i] for _ in range(idx.sum())]

        if log is not None:
            log.debug("    | E = %e << %s"%(e_new[i],e[s]))
            log.debug("       evals: %s"%val)
            log.debug("       kept:  %s"%(val[idx]))
    if warn_non_causal:
        if log is not None:
            log.warning("Non-causal poles found in self-energy")
    couplings = np.hstack(couplings).real
    return Lehmann(np.array(energies), np.array(couplings))


def remove_se_degeneracy_nsym(se, dtol=1e-8, etol=1e-6, img_space=True, log=None):

    if log is None:
        log.debug("Removing degeneracy in self-energy - degenerate energy tol=%e   evec tol=%e"%(dtol, etol))
    e = se.energies
    couplings_l, couplings_r = se.unpack_couplings()
    e_new, slices = get_unique(e, atol=dtol)#
    if log is not None:
        log.debug("Number of energies = %d,  unique = %d"%(len(e),len(e_new)))
    energies, new_couplings_l, new_couplings_r = [], [], []
    for i, s in enumerate(slices):
        mat = np.einsum('pa,qa->pq', couplings_l[:,s], couplings_r[:,s].conj())
        if img_space:
            # Fast image space algorithm
            u, sing, v = svd_outer_sum(couplings_l[:,s].T, couplings_r[:,s].T, tol=etol)
            idx = sing > etol
        else:
            # Slow SVD in full space
            U, sing, Vh = np.linalg.svd(mat)
            idx = sing > etol
            u = U[:,idx] @ np.diag(np.sqrt(sing[idx]))
            v = Vh.T.conj()[:,idx] @ np.diag(np.sqrt(sing[idx]))
        new_couplings_l.append(u)
        new_couplings_r.append(v)
        energies += [e_new[i] for _ in range(u.shape[1])]

        if log is not None:
            log.debug("    | E = %e << %s"%(e_new[i],e[s]))
            log.debug("       sing vals: %s"%sing)
            log.debug("            kept:  %s"%(sing[idx]))
    
    couplings = np.array([np.hstack(new_couplings_l), np.hstack(new_couplings_r)])
    return Lehmann(np.array(energies), couplings)


def get_unique(array, atol=1e-15): 
    
    # Find elements of a sorted float array which are unique up to a tolerance
    
    assert len(array.shape) == 1
    
    i = 0
    slices = []
    while i < len(array):
        j = 1
        idxs = [i]
        while i+j < len(array):
            if np.abs(array[i] - array[i+j]) < atol:
                idxs.append(i+j)
                j += 1
            else: 
                break
        i = i + j
        slices.append(np.s_[idxs[0]:idxs[-1]+1])
    new_array = np.array([array[s].mean() for s in slices])
    return new_array, slices

def check_causal(se, log=None, verbose=False):
    couplings_l, couplings_r = se.unpack_couplings()
    energies = se.energies
    ret = True
    for i, e in enumerate(energies):
        m = np.einsum('pi,qi->pq', couplings_l, couplings_r.conj())
        val, vec = np.linalg.eig(m)
        if np.any(val < 0):
            if log and verbose:
                log.debug("Non-causal pole at %s"%e)
            ret = False
    return ret

def fit_hermitian(se):
    """
    Fit a causal self-energy

    Parameters
    ----------
    se : Lehmann
        Self-energy in Lehmann representation
    
    Returns
    -------
    se : Lehmann
        Fitted causal self-energy
    """

    energies = se.energies.copy()
    couplings_l, couplings_r = se.unpack_couplings()
    couplings_l, couplings_r = couplings_l.copy(), couplings_r.copy().conj()
    def f(w):
        denom = 1 / (1j*w - energies + 1j)
        return np.einsum('pa,qa,a->pq', couplings_l, couplings_r, denom)

    def obj(x):
        x = x.reshape(shape)
        V, e = x[:-1], x[-1]
        def integrand(w):
            denom = 1 / (1j*w - energies)
            a = np.einsum('pa,qa,a->pq', couplings_l, couplings_r, denom)

            denom = 1 / (1j*w - e)
            b = np.einsum('pa,qa,a->pq', V, V, denom)
            c = (np.abs(a - b) ** 2).sum()
            return c
        lim = np.inf
        val, err = scipy.integrate.quad(integrand, -lim, lim)
        return val
    
    def grad(x):
        x = x.reshape(shape)
        V, e = x[:-1], x[-1]
        def integrand_V(w):
            a = np.einsum('pa,qa,a->pq', couplings_l, couplings_r, 1 / (1j*w - energies))
            b = np.einsum('pa,qa,a->pq', V, V, 1 / (1j*w - e))
            d = b - a
            omegaRe = e/(w**2 + e**2)
            omegaIm = w/(w**2 + e**2)

            ret  = np.einsum('rq,qb,b->rb', d.real, V, omegaRe)
            ret += np.einsum('pr,pb,b->rb', d.real, V, omegaRe)
            ret += np.einsum('rq,qb,b->rb', d.imag, V, omegaIm)
            ret += np.einsum('pr,pb,b->rb', d.imag, V, omegaIm)
            return -2 * ret
        
        def integrand_e(w):
            a = np.einsum('pa,qa,a->pq', couplings_l, couplings_r, 1 / (1j*w - energies))
            b = np.einsum('pa,qa,a->pq', V, V, 1 / (1j*w - e))
            d = b - a
            omegaRe = (e**2 - w**2)/(w**2 + e**2)**2
            omegaIm = 2*e*w/(w**2 + e**2)**2

            ret = 2*np.einsum('pq,pb,qb,b->b', d.real, V, V, omegaRe)
            ret += 2*np.einsum('pq,pb,qb,b->b', d.imag, V, V, omegaIm)
            return ret


        integrand = lambda w: np.hstack([integrand_V(w).flatten(), integrand_e(w)])
        lim = np.inf
        jac, err_V = scipy.integrate.quad_vec(lambda x: integrand(x), -lim, lim)
        return jac
        

    x0 = np.vstack([couplings_l, energies])
    shape = x0.shape
    x0 = x0.flatten()

    xgrad = grad(x0)
    res = scipy.optimize.minimize(obj, x0, jac=grad, method='Newton-CG')
    x = res.x.reshape(shape)
    return Lehmann(x[-1], x[:-1])
