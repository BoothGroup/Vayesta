"""Routines to reconstruct the full system self-energy from cluster spectral moments"""

import numpy as np
import scipy

from vayesta.core.util import NotCalculatedError, Object, dot, einsum
try:
    from dyson import Lehmann, MBLGF, MixedMBLGF, MBLSE, MixedMBLSE, AuxiliaryShift, AufbauPrinciple
except ImportError as e:
    print(e)
    print("Dyson required for self-energy calculations")


def gf_moments_block_lanczos(moments, hermitian=True, sym_moms=True, shift=None, nelec=None, log=None, **kwargs):
    """
    Compute the Green's function moments from the spectral moments using the block Lanczos algorithm.

    Parameters
    ----------
    moments : ndarray (nmom, nmo, nmo)
        Spectral moments
    hermitian : bool
        Use Hermitian block Lanczos solver
    sym_moms : bool
        Symmetrise moments
    shift : string ('None', 'auxiliary' or 'aufbau')
        Type of shift to apply to self-energy
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
        th = moments[0].copy()
        tp = moments[1].copy()
        th = 0.5 * (th + th.transpose(0,2,1))
        tp = 0.5 * (tp + tp.transpose(0,2,1))
    else:
        th, tp = moments[0].copy(), moments[1].copy()

    solverh = MBLGF(th, hermitian=hermitian, log=log)
    solverp = MBLGF(tp, hermitian=hermitian, log=log)
    solver = MixedMBLGF(solverh, solverp)
    solver.kernel()
    gf = solver.get_greens_function()
    se = solver.get_self_energy()

    dm = gf.occupied().moment(0) * 2
    nelec = np.trace(dm)
    #log.info("Fragment %s: Electron target %f %f without shift"%(f.id, f.nelectron, nelec))

    if shift is not None:
        if nelec is None:
            raise ValueError("Number of electrons must be provided for shift")
        Shift = AuxiliaryShift if shift == 'aux' else AufbauPrinciple
        shift = Shift(th[1]+tp[1], se, nelec, occupancy=2, log=log)
        shift.kernel()
        se = shift.get_self_energy()
        gf = shift.get_greens_function()
    
    return se, gf

def se_moments_block_lanczos(se_static, se_moments, hermitian=True, sym_moms=True, shift=None, nelec=None, log=None, **kwargs):
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
        solverh = MBLSE(se_static, th, hermitian=hermitian, log=log)
        solverp = MBLSE(se_static, tp, hermitian=hermitian, log=log)
        solver = MixedMBLSE(solverh, solverp)
        solver.kernel()
    else:
        solver = MBLSE(se_static, se_moments, hermitian=hermitian, log=log)
        solver.kernel()

    gf = solver.get_greens_function()
    se = solver.get_self_energy()

    dm = gf.occupied().moment(0) * 2
    nelec = np.trace(dm)
    #log.info("Fragment %s: Electron target %f %f without shift"%(f.id, f.nelectron, nelec))

    if shift is not None:
        if nelec is None:
            raise ValueError("Number of electrons must be provided for shift")
        Shift = AuxiliaryShift if shift == 'aux' else AufbauPrinciple
        shift = Shift(se_static, se, nelec, occupancy=2, log=log)
        shift.kernel()
        se = shift.get_self_energy()
        gf = shift.get_greens_function()
    
    return se, gf
    

def make_self_energy_moments(emb, nmom_se=None, nmom_gf=None, use_sym=True, proj=1, hermitian=True, sym_moms=True, eta=1e-1, debug_gf_moms=None, debug_se_moms=None):
    """
    Construct full system self-energy moments from cluster spectral moments

    Parameters
    ----------
    emb : EWF object
        Embedding object
    n_se_mom : int
        Number of self-energy moments
    use_sym : bool
        Use symmetry to reconstruct self-energy
    proj : int
        Number of projectors to use (1 or 2)
    eta : float
        Broadening factor for static potential

    Returns
    -------
    self_energy_moms : ndarry (n_se_mom, nmo, nmo)
        Full system self-energy moments (MO basis)
    static_self_energy : ndarray (nmo,nmo)
        Static part of self-energy (MO basis)
    static_potential : ndarray (nao,nao)
        Static potential (AO basis)
    """

    fock = emb.get_fock()
    static_self_energy = np.zeros_like(fock)
    static_potential = np.zeros_like(fock)
    

    fragments = emb.get_fragments(sym_parent=None) if use_sym else emb.get_fragments()
    if nmom_se is None:
        nmom_se = fragments[0].results.se_moments[0].shape[0]
    assert nmom_se <= fragments[0].results.se_moments.shape[0]
    self_energy_moms = np.zeros((nmom_se, fock.shape[1], fock.shape[1]))
    for i, f in enumerate(fragments):
        if debug_se_moms is None and debug_gf_moms is None:
            se = f.results.self_energy
            gf = f.results.greens_function
            se_static = f.results.static_self_energy
            se_moms = se.occupied().moment(range(nmom_se)), se.virtual().moment(range(nmom_se)) if ph_separation else se.moment(range(nmom_se))
        elif debug_se_moms is not None:
            debug_se_moms = np.array(debug_se_moms)
            assert debug_se_moms.shape[0] == 2 and debug_se_moms.shape[1] == len(fragments)
            se_static, se_moms = debug_se_moms[0][i], debug_se_moms[1][i] 
        elif debug_gf_moms is not None:
            gf_moms = debug_gf_moms
            se_static = gf_moms[0][1] + gf_moms[1][1]
            se_moms = gf_moms[0], gf_moms[1]
        se = f.results.self_energy
        mc = f.get_overlap('mo|cluster')
        mf = f.get_overlap('mo|frag')
        fc = f.get_overlap('frag|cluster')
        cfc = fc.T @ fc
        
        # Fock matrix in cluster basis
        fock_cls = f.cluster.c_active.T @ fock @ f.cluster.c_active
        e_cls = np.diag(fock_cls)
        
        if proj == 1:
            # Static potential
            v_cls = se.as_static_potential(e_cls, eta=eta) # Static potential (used to update MF for the self-consistnecy)
            v_frag = cfc @ v_cls  
            v_frag = 0.5 * (v_frag + v_frag.T)
            static_potential += f.cluster.c_active @ v_frag @ f.cluster.c_active.T

            # Static self-energy
            static_se_cls = f.results.static_self_energy - fock_cls
            static_self_energy_frag = cfc @ static_se_cls
            static_self_energy_frag = 0.5 * (static_self_energy_frag + static_self_energy_frag.T)
            static_self_energy += mc @ static_self_energy_frag @ mc.T

            # Self-energy moments
            se_moms_frag = np.array([0.5*(cfc @ mom + mom @ cfc) for mom in se_moms_clus])
            self_energy_moms += np.array([mc @ mom @ mc.T for mom in se_moms_frag])

            if use_sym:
                for child in f.get_symmetry_children():
                    static_potential += child.cluster.c_active @ v_frag @ child.cluster.c_active.T
                    mc_child = child.get_overlap('mo|cluster')
                    static_self_energy += mc_child @ static_self_energy_frag @ mc_child.T
                    self_energy_moms += np.array([mc_child @ mom @ mc_child.T for mom in se_moms_frag])
            
        elif proj == 2:
            # Static potential 
            v_cls = se.as_static_potential(e_cls, eta=eta) 
            v_frag = fc @ v_cls @ fc.T
            static_potential += f.c_frag @ v_frag @ f.c_frag.T

            # Static self-energy
            static_se_cls = f.results.static_self_energy - fock_cls
            static_se_frag = fc @ static_se_cls @ fc.T
            static_self_energy += mf @ static_se_frag @ mf.T

            # Self-energy moments
            se_moms_frag = [0.5*(fc @ mom @ fc.T) for mom in se_moms_clus]
            self_energy_moms += np.array([mf @ mom @ mf.T for mom in se_moms_frag])

            if use_sym:
                for child in f.get_symmetry_children():
                    static_potential += child.c_frag @ v_frag @ child.c_frag.T
                    mf_child = child.get_overlap('mo|frag')
                    fc_child = child.get_overlap('frag|cluster')
                    static_self_energy += mf_child @ static_se_frag @ mf_child.T
                    self_energy_moms += np.array([mf_child @ mom @ mf_child.T for mom in se_moms_frag])

    if sym_moms:
        self_energy_moms = 0.5 * (self_energy_moms + self_energy_moms.transpose(0,2,1)) 
        static_self_energy = 0.5 * (static_self_energy + static_self_energy.T)
        static_potential = 0.5 * (static_potential + static_potential.T)

    return self_energy_moms, static_self_energy, static_potential


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


def make_self_energy_1proj(emb, hermitian=True, use_sym=True, sym_moms=False, use_svd=True, eta=1e-1, nmom_gf=None, aux_shift_frag=False, se_degen_tol=1e-4, se_eval_tol=1e-6, drop_non_causal=False):
    """
    Construct full system self-energy in Lehmann representation from cluster spectral moments using 1 projector

    TODO: MPI, SVD

    Parameters
    ----------
    emb : EWF object
        Embedding object
    use_sym : bool
        Use symmetry to reconstruct self-energy
    use_svd : bool
        Use SVD to decompose the self-energy as outer product
    eta : float
        Broadening factor for static potential
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
    static_self_energy : ndarray (nmo,nmo)
        Static part of self-energy (MO basis)
    static_potential : ndarray (nao,nao)
        Static potential (AO basis)
    """

    fock = emb.get_fock()
    static_self_energy = np.zeros_like(fock)
    static_potential = np.zeros_like(fock)
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
        se_static = th[1] + tp[1]
            
        se, gf = gf_moments_block_lanczos((th,tp), hermitian=hermitian, sym_moms=sym_moms, shift=None, nelec=f.nelectron, log=emb.log)
        
        dm = gf.occupied().moment(0) * 2
        nelec = np.trace(dm)
        emb.log.info("Fragment %s: Electron target %f %f without shift"%(f.id, f.nelectron, nelec))
        if aux_shift_frag:
            aux = AuxiliaryShift(se_static, se, f.nelectron, occupancy=2, log=emb.log)
            aux.kernel()
            se = aux.get_self_energy()
            gf = aux.get_greens_function()
            dm = gf.occupied().moment(0) * 2
            nelec = np.trace(dm)
            emb.log.info("Fragment %s: Electron target %f %f with shift"%(f.id, f.nelectron, nelec))


        #se = se.physical(weight=1e-6)
        mc = f.get_overlap('mo|cluster')
        fc = f.get_overlap('frag|cluster')
        cfc = fc.T @ fc
        
        # Fock matrix in cluster basis
        fock_cls = f.cluster.c_active.T  @ fock  @ f.cluster.c_active
        e_cls = np.diag(fock_cls)
        
        # Static potential
        v_cls = se.as_static_potential(e_cls, eta=eta) # Static potential (used to update MF for the self-consistnecy)
        v_frag = cfc @ v_cls  
        v_frag = 0.5 * (v_frag + v_frag.T)
        static_potential += f.cluster.c_active @ v_frag @ f.cluster.c_active.T

        # Static self-energy
        static_se_cls = se_static - fock_cls
        static_self_energy_frag = cfc @ static_se_cls
        static_self_energy_frag = 0.5 * (static_self_energy_frag + static_self_energy_frag.T)
        static_self_energy += mc @ static_self_energy_frag @ mc.T

        # Dynamic self-energy
        coup_l, coup_r = se._unpack_couplings()
        sym_coup = 0.5*(einsum('pa,qa->apq', cfc @ coup_l , coup_r.conj()) + einsum('pa,qa->apq', coup_l , cfc @ coup_r.conj()))

        if use_svd:
            couplings_l_frag, couplings_r_frag, energies_frag = [], [], []
            for a in range(sym_coup.shape[0]):
                m = sym_coup[a]
                U, s, Vt = np.linalg.svd(m)
                idx = np.abs(s) > se_eval_tol
                assert idx.sum() <= 2
                u = U[:,idx] @ np.diag(np.sqrt(s[idx]))
                v = Vt.conj().T[:,idx] @ np.diag(np.sqrt(s[idx]))
                couplings_l_frag.append(u)
                couplings_r_frag.append(v)
                energies_frag += [se.energies[a] for e in range(idx.sum())]
            
            couplings_l_frag, couplings_r_frag = np.hstack(couplings_l_frag), np.hstack(couplings_r_frag)
            couplings_l.append(mc @ couplings_l_frag)
            couplings_r.append(mc @ couplings_r_frag)
            energies.append(energies_frag)

            mat = sym_coup.sum(axis=0)
            mat2 = np.einsum('pa,qa->pq', couplings_l_frag, couplings_r_frag.conj())
            emb.log.info("Norm diff of SE numerator %s"%np.linalg.norm(mat - mat2))
        else:
            couplings_frag, energies_frag = [], []
            for a in range(sym_coup.shape[0]):
                m = sym_coup[a]
                assert np.allclose(m, m.T.conj())
                val, vec = np.linalg.eigh(m)
                idx = np.abs(val) > se_eval_tol
                assert idx.sum() <= 2
                w = vec[:,idx] @ np.diag(np.sqrt(val[idx], dtype=np.complex64))
                couplings_frag.append(w)
                energies_frag += [se.energies[a] for e in range(idx.sum())]

            couplings_frag = np.hstack(couplings_frag)

            couplings.append(mc @ couplings_frag)
            energies.append(energies_frag)

        if use_sym:
            for child in f.get_symmetry_children():
                static_potential += child.cluster.c_active @ v_frag @ child.cluster.c_active.T
                mc_child = child.get_overlap('mo|cluster')
                static_self_energy += mc_child @ static_self_energy_frag @ mc_child.T
                energies.append(energies_frag)
                if use_svd:
                    couplings_l.append(mc_child @ couplings_l_frag)
                    couplings_r.append(mc_child @ couplings_r_frag)
                else:
                    couplings.append(mc_child @ couplings_frag)

    energies = np.concatenate(energies)
    if use_svd:
        couplings = np.hstack(couplings_l), np.hstack(couplings_r)
    else:
        couplings = np.hstack(couplings)
    self_energy = Lehmann(energies, couplings)

    self_energy = remove_se_degeneracy(self_energy, dtol=se_degen_tol, etol=se_eval_tol, drop_non_causal=drop_non_causal, log=emb.log)

    return self_energy, static_self_energy, static_potential

def make_self_energy_1proj_img(emb, hermitian=True, use_sym=True, sym_moms=False, img_space=False, use_svd=False, eta=1e-1, nmom_gf=None, aux_shift_frag=False, se_degen_tol=1e-4, se_eval_tol=1e-6, drop_non_causal=False):
    """
    Construct full system self-energy in Lehmann representation from cluster spectral moments using 1 projector

    TODO: MPI, SVD

    Parameters
    ----------
    emb : EWF object
        Embedding object
    use_sym : bool
        Use symmetry to reconstruct self-energy
    use_svd : bool
        Use SVD to decompose the self-energy as outer product
    eta : float
        Broadening factor for static potential
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
    static_self_energy : ndarray (nmo,nmo)
        Static part of self-energy (MO basis)
    static_potential : ndarray (nao,nao)
        Static potential (AO basis)
    """

    fock = emb.get_fock()
    static_self_energy = np.zeros_like(fock)
    static_potential = np.zeros_like(fock)
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
        se_static = th[1] + tp[1]
            
        se, gf = gf_moments_block_lanczos((th,tp), hermitian=hermitian, sym_moms=sym_moms, shift=None, nelec=f.nelectron, log=emb.log)
        
        dm = gf.occupied().moment(0) * 2
        nelec = np.trace(dm)
        emb.log.info("Fragment %s: Electron target %f %f without shift"%(f.id, f.nelectron, nelec))
        if aux_shift_frag:
            aux = AuxiliaryShift(se_static, se, f.nelectron, occupancy=2, log=emb.log)
            aux.kernel()
            se = aux.get_self_energy()
            gf = aux.get_greens_function()
            dm = gf.occupied().moment(0) * 2
            nelec = np.trace(dm)
            emb.log.info("Fragment %s: Electron target %f %f with shift"%(f.id, f.nelectron, nelec))


        #se = se.physical(weight=1e-6)
        mc = f.get_overlap('mo|cluster')
        fc = f.get_overlap('frag|cluster')
        cfc = fc.T @ fc
        
        # Fock matrix in cluster basis
        fock_cls = f.cluster.c_active.T  @ fock  @ f.cluster.c_active
        e_cls = np.diag(fock_cls)
        
        # Static potential
        v_cls = se.as_static_potential(e_cls, eta=eta) # Static potential (used to update MF for the self-consistnecy)
        v_frag = cfc @ v_cls  
        v_frag = 0.5 * (v_frag + v_frag.T)
        static_potential += f.cluster.c_active @ v_frag @ f.cluster.c_active.T

        # Static self-energy
        static_se_cls = se_static - fock_cls
        static_self_energy_frag = cfc @ static_se_cls
        static_self_energy_frag = 0.5 * (static_self_energy_frag + static_self_energy_frag.T)
        static_self_energy += mc @ static_self_energy_frag @ mc.T

        couplings_proj, energies_proj = [], []
        #if hermitian or 1:
        # Dynamic self-energy
        coup_l, coup_r = se._unpack_couplings()
        p_coup_l, p_coup_r = cfc @ coup_l, cfc @ coup_r
        sym_coup = 0.5*(einsum('pa,qa->apq', cfc @ coup_l , coup_r.conj()) + einsum('pa,qa->apq', coup_l , cfc @ coup_r.conj()))
        nmo, naux = coup_l.shape
        couplings_l_frag, couplings_r_frag, energies_frag = [], [], []
            
        for a in range(naux):
            if hermitian:
                vs, ws = [coup_l[:,a]], [p_coup_l[:,a]]
                rank = 2
                left, right = np.zeros((rank, nmo)), np.zeros((nmo, rank))
                for i in range(len(vs)):
                    left[2*i] = ws[i]
                    left[2*i+1] = vs[i]
                    right[:,2*i] = vs[i]
                    right[:,2*i+1] = ws[i]
            else:
                vs, ws = [p_coup_l[:,a], coup_l[:,a]], [coup_r[:,a].conj(), p_coup_r[:,a].conj()]
                rank = 4 # Can redo for rank 2
                left, right = np.zeros((rank, nmo), dtype=np.complex128), np.zeros((nmo, rank), dtype=np.complex128)
                for i in range(len(vs)):
                    left[2*i] = ws[i]
                    #left[2*i+1] = vs[i]
                    right[:,2*i] = vs[i]
                    right[:,2*i+1] = ws[i]
            mat = 0.5 * (left @ right)
            U, s, Vt = np.linalg.svd(mat)
            idx = s > se_eval_tol
            U = U[:,idx]
            Vt = Vt[idx,:]
            s = s[idx]
            u = U @ np.diag(np.sqrt(s))
            v = Vt.conj().T @ np.diag(np.sqrt(s))
            
            basis = right
            dbasis = np.linalg.pinv(basis)
            couplings_l_frag.append(basis @ u)
            couplings_r_frag.append(dbasis.T.conj() @ v)
            energies_frag += [se.energies[a] for e in range(idx.sum())]
            
            basis = right
            dbasis = np.linalg.pinv(basis)
            couplings_l_frag.append(basis @ u)
            couplings_r_frag.append(dbasis.T @ v)
            energies_frag += [se.energies[a] for e in range(rank)]

            
        couplings_l_frag, couplings_r_frag = np.hstack(couplings_l_frag), np.hstack(couplings_r_frag)
        couplings_l.append(mc @ couplings_l_frag)
        couplings_r.append(mc @ couplings_r_frag)
        energies.append(energies_frag)

        mat = sym_coup.sum(axis=0)
        mat2 = np.einsum('pa,qa->pq', couplings_l_frag, couplings_r_frag.conj())
        print("Norm diff of SE numerator %s"%np.linalg.norm(mat - mat2))

    energies = np.concatenate(energies)
    if use_svd:
        couplings = np.hstack(couplings_l), np.hstack(couplings_r)
    else:
        couplings = np.hstack(couplings)

    print(energies.shape)
    print(couplings.shape)
    self_energy = Lehmann(energies, couplings)

    self_energy = remove_se_degeneracy(self_energy, dtol=se_degen_tol, etol=se_eval_tol, drop_non_causal=drop_non_causal, log=emb.log)

    return self_energy, static_self_energy, static_potential
    
def make_self_energy_2proj(emb, nmom_gf=None, hermitian=True, sym_moms=False, use_sym=True, aux_shift_frag=False, eta=1e-1):
    """
    Construct full system self-energy in Lehmann representation from cluster spectral moments using 2 projectors

    TODO: MPI, SVD

    Parameters
    ----------
    emb : EWF object
        Embedding object
    use_sym : bool
        Use symmetry to reconstruct self-energy
    eta : float
        Broadening factor for static potential

    Returns
    -------
    self_energy : Lehmann object
        Reconstructed self-energy in Lehmann representation (MO basis)
    static_self_energy : ndarray (nmo,nmo)
        Static part of self-energy (MO basis)
    static_potential : ndarray (nao,nao)
        Static potential (AO basis)
    """

    fock = emb.get_fock()
    static_self_energy = np.zeros_like(fock)
    static_potential = np.zeros_like(fock)
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
            
        se, gf = gf_moments_block_lanczos((th,tp), hermitian=hermitian, sym_moms=sym_moms, shift=None, nelec=f.nelectron, log=emb.log)

        dm = gf.occupied().moment(0) * 2
        nelec = np.trace(dm)
        emb.log.info("Fragment %s: Electron target %f %f without shift"%(f.id, f.nelectron, nelec))
        if aux_shift_frag:
            aux = AuxiliaryShift(f.results.static_self_energy, se, f.nelectron, occupancy=2, log=emb.log)
            aux.kernel()
            se = aux.get_self_energy()
            gf = aux.get_greens_function()
            dm = gf.occupied().moment(0) * 2
            nelec = np.trace(dm)
            emb.log.info("Fragment %s: Electron target %f %f with shift"%(f.id, f.nelectron, nelec))


        mf = f.get_overlap('mo|frag')
        fc = f.get_overlap('frag|cluster')
        
        # Fock matrix in cluster basis
        fock_cls = f.cluster.c_active.T @ fock @ f.cluster.c_active
        e_cls = np.diag(fock_cls)
        
        # Static potential 
        v_cls = se.as_static_potential(e_cls, eta=eta) 
        v_frag = fc @ v_cls @ fc.T
        static_potential += f.c_frag @ v_frag @ f.c_frag.T

        # Static self-energy
        static_se_cls = se_static - fock_cls
        static_se_frag = fc @ static_se_cls @ fc.T
        static_self_energy += mf @ static_se_frag @ mf.T

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
                static_potential += child.c_frag @ v_frag @ child.c_frag.T
                mf_child = child.get_overlap('mo|frag')
                fc_child = child.get_overlap('frag|cluster')
                static_self_energy += mf_child @ static_se_frag @ mf_child.T
                if type(se.couplings) is tuple:
                    couplings_l, couplings_r = se.couplings
                    couplings_l = mf_child @ fc_child @ couplings_l
                    couplings_r = mf_child @ fc_child @ couplings_r
                    couplings.append((couplings_l, couplings_r))
                else:
                    couplings.append(mf_child @ fc_child @ se.couplings)

                energies.append(se.energies)

    if type(couplings[0]) is tuple:
        couplings_l, couplings_r = zip(*couplings)
        couplings = np.hstack(couplings_l), np.hstack(couplings_r)
    else:
        couplings = np.hstack(couplings)
    energies = np.concatenate(energies)
    self_energy = Lehmann(energies, couplings)
    #self_energy = remove_se_degeneracy( self_energy, dtol=se_degen_tol, etol=se_eval_tol, drop_non_causal=drop_non_causal, log=emb.log)

    return self_energy, static_self_energy, static_potential

def drop_and_reweight(se, tol=1e-12):
    couplings_l, couplings_r = se._unpack_couplings()
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
    U, V = se._unpack_couplings()
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

def eig_outer_sum(vs, ws, tol=1e-12):
    """
    Calculate the eigenvalues and eigenvectors of the sum of symmetrised outer products.
    Given lists of vectors vs and ws, the function calcualtes the eigendecomposition
    of the matrix 0.5*(sum_i outer(vs[i], ws[i]) + outer(ws[i], vs[i])) working in
    the image space of that matrix.

    Parameters
    ----------
    vs : np.ndarray (n,N)
        List of n vectors of length N
    ws : np.ndarray (n,N)
        List of n vectors of length N

    Returns
    -------
    val : np.ndarray (n,)
        Eigenvalues 
    vec : np.ndarray
        Eigenvectors (N,n)
    """
    vs, ws = np.array(vs), np.array(ws)
    assert vs.shape == ws.shape
    rank = 2 * vs.shape[0]
    N = vs.shape[1]
    left, right = np.zeros((rank, N)), np.zeros((N, rank))
    for i in range(len(vs)):
        left[2*i] = ws[i]
        left[2*i+1] = vs[i]
        right[:,2*i] = vs[i]
        right[:,2*i+1] = ws[i]
    mat = 0.5 * (left @ right)
    val, vec = np.linalg.eig(mat)
    assert np.allclose(val.imag, 0)
    val = val.real
    idx = np.abs(val) > tol
    val, vec = val[idx], vec[:,idx]
    idx = val.argsort()
    val, vec = val[idx], vec[:,idx]
    vec = right @ vec 
    vec = vec / np.linalg.norm(vec, axis=0)
    return val.real, vec

def eig_outer_slow(vs, ws, tol=1.e-10):
    #outer = 0.5*(np.einsum('ai,aj->ij', vs, ws) + np.einsum('ai,aj->ij', ws, vs))
    outer = 0.5 * (np.tensordot(vs, ws, axes=([0],[0])) + np.tensordot(ws, vs, axes=([0],[0])))
    val, vec = np.linalg.eigh(outer)
    assert np.allclose(val.imag, 0)
    val = val.real
    idx = np.abs(val) > tol
    val, vec = val[idx], vec[:,idx]
    idx = val.argsort()
    val, vec = val[idx], vec[:,idx]
    return val, vec
    
def remove_se_degeneracy(se, dtol=1e-8, etol=1e-6, drop_non_causal=False, log=None):

    if log is None:
        log.debug("Removing degeneracy in self-energy - degenerate energy tol=%e   evec tol=%e"%(dtol, etol))
    e = se.energies
    couplings_l, couplings_r = se._unpack_couplings()
    e_new, slices = get_unique(e, atol=dtol)#
    if log is not None:
        log.debug("Number of energies = %d,  unique = %d"%(len(e),len(e_new)))
    energies, couplings = [], []
    warn_non_causal = False
    for i, s in enumerate(slices):
        mat = np.einsum('pa,qa->pq', couplings_l[:,s], couplings_r[:,s].conj()).real
        #print("Hermitian: %s"%np.linalg.norm(mat - mat.T.conj()))
        #assert np.allclose(mat, mat.T.conj())
        val, vec = np.linalg.eigh(mat)
        if  drop_non_causal:
            idx = val > etol
        else:
            idx = np.abs(val) > etol
        if np.sum(val[idx] < -etol) > 0:
            warn_non_causal = True
        w = vec[:,idx] @ np.diag(np.sqrt(val[idx], dtype=np.complex64))
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
    couplings_l, couplings_r = se._unpack_couplings()
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
    couplings_l, couplings_r = se._unpack_couplings()
    couplings_l, couplings_r = couplings_l.copy(), couplings_r.copy().conj()
    def f(w):
        denom = 1 / (1j*w - energies + 1j * eta)
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
            #print(c)
            return c
        lim = np.inf
        val, err = scipy.integrate.quad(integrand, -lim, lim)
        print("obj: %s err: %s"%(val, err))
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

            #print(omegaIm)

            ret = 2*np.einsum('pq,pb,qb,b->b', d.real, V, V, omegaRe)
            ret += 2*np.einsum('pq,pb,qb,b->b', d.imag, V, V, omegaIm)
            return ret


        integrand = lambda w: np.hstack([integrand_V(w).flatten(), integrand_e(w)])
        lim = np.inf
        jac, err_V = scipy.integrate.quad_vec(lambda x: integrand(x), -lim, lim)
        print('grad norm: %s err: %s'%(np.linalg.norm(jac),err_V))
        #print(grad)
        return jac
        

    x0 = np.vstack([couplings_l, energies])
    shape = x0.shape
    x0 = x0.flatten()

    xgrad = grad(x0)
    print(shape)
    print("obj(x0) = %s"%obj(x0))
    print('grad(x0)')
    print(xgrad)
    #x0 = np.random.randn(*x0.shape)  #* 1e-2

    #x = xgrad.reshape(x0.shape)

    #return xgrad
    print(shape)
    res = scipy.optimize.minimize(obj, x0, jac=grad, method='Newton-CG')
    #res = scipy.optimize.basinhopping(obj, x0.flatten(), niter=10, minimizer_kwargs=dict(method='BFGS'))
    print("Sucess %s, Integral = %s"%(res.success, res.x))

    x = res.x.reshape(shape)
    return Lehmann(x[-1], x[:-1])
