"""Routines to reconstruct the full system self-energy from cluster spectral moments"""

import numpy as np

from vayesta.core.util import NotCalculatedError, Object, dot, einsum
from dyson import Lehmann, MBLGF, MixedMBLGF, NullLogger, AuxiliaryShift


def make_self_energy_moments(emb, n_se_mom, use_sym=True, proj=1, eta=1e-2):
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
    self_energy_moms = np.zeros((n_se_mom, fock.shape[1], fock.shape[1]))

    fragments = emb.get_fragments(sym_parent=None) if use_sym else emb.get_fragments()
    for i, f in enumerate(fragments):
        # Calculate self energy from cluster moments
        th, tp = f.results.moms

        solverh = MBLGF(th, log=NullLogger())
        solverp = MBLGF(tp, log=NullLogger())
        solver = MixedMBLGF(solverh, solverp)
        solver.kernel()
        se = solver.get_self_energy()
        se_moms_clus = [se.moment(i) for i in range(n_se_mom)]

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
            static_se_cls = th[1] + tp[1] - fock_cls
            static_self_energy_frag = cfc @ static_se_cls
            static_self_energy_frag = 0.5 * (static_self_energy_frag + static_self_energy_frag.T)
            static_self_energy += mc @ static_self_energy_frag @ mc.T

            # Self-energy moments
            se_moms_frag = [0.5*(cfc @ mom + mom @ cfc) for mom in se_moms_clus]
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
            static_se_cls = th[1] + tp[1] - fock_cls
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

    return self_energy_moms, static_self_energy, static_potential

def make_self_energy_1proj(emb, use_sym=True, use_svd=True, eta=1e-2, aux_shift_frag=False, se_degen_tol=1e-4, se_eval_tol=1e-6, drop_non_causal=False):
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
        # Calculate self energy from cluster moments
        th, tp = f.results.moms

        solverh = MBLGF(th, log=emb.log)
        solverp = MBLGF(tp, log=emb.log)
        solver = MixedMBLGF(solverh, solverp)
        solver.kernel()
        se = solver.get_self_energy()
        gf = solver.get_greens_function()
        dm = gf.occupied().moment(0) * 2
        nelec = np.trace(dm)
        emb.log.info("Fragment %s: Electron target %f %f without shift"%(f.id, f.nelectron, nelec))
        if aux_shift_frag:
            aux = AuxiliaryShift(th[0]+tp[0], se, f.nelectron, occupancy=2, log=emb.log)
            aux.kernel()
            se = aux.get_self_energy()
            gf = aux.get_greens_function()
            dm = gf.occupied().moment(0) * 2
            nelec = np.trace(dm)
            emb.log.info("Fragment %s: Electron target %f %f with shift"%(f.id, f.nelectron, nelec))

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
        static_se_cls = th[1] + tp[1] - fock_cls
        static_self_energy_frag = cfc @ static_se_cls
        static_self_energy_frag = 0.5 * (static_self_energy_frag + static_self_energy_frag.T)
        static_self_energy += mc @ static_self_energy_frag @ mc.T

        # Dynamic self-energy
        coup_l, coup_r = se._unpack_couplings()
        sym_coup = 0.5*(einsum('pa,qa->apq', cfc @ coup_l , coup_r) + einsum('pa,qa->apq', coup_l , cfc @ coup_r))

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
        else:
            couplings_frag, energies_frag = [], []
            for a in range(sym_coup.shape[0]):
                m = sym_coup[a]
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

    self_energy = remove_se_degeneracy(emb, self_energy, dtol=se_degen_tol, etol=se_eval_tol, drop_non_causal=drop_non_causal)

    return self_energy, static_self_energy, static_potential


def make_self_energy_2proj(emb, use_sym=True, eta=1e-2):
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
        # Calculate self energy from cluster moments
        th, tp = f.results.moms

        solverh = MBLGF(th, log=NullLogger())
        solverp = MBLGF(tp, log=NullLogger())
        solver = MixedMBLGF(solverh, solverp)
        solver.kernel()
        se = solver.get_self_energy()

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
        static_se_cls = th[1] + tp[1] - fock_cls
        static_se_frag = fc @ static_se_cls @ fc.T
        static_self_energy += mf @ static_se_frag @ mf.T

        # Dynamic self-energy
        couplings.append(mf @ fc @ se.couplings)
        energies.append(se.energies)

        if use_sym:
            for child in f.get_symmetry_children():
                static_potential += child.c_frag @ v_frag @ child.c_frag.T
                mf_child = child.get_overlap('mo|frag')
                fc_child = child.get_overlap('frag|cluster')
                static_self_energy += mf_child @ static_se_frag @ mf_child.T
                x = mf_child @ fc_child @ se.couplings
                couplings.append(x)
                energies.append(se.energies)

    couplings = np.hstack(couplings)
    energies = np.concatenate(energies)
    self_energy = Lehmann(energies, couplings)

    return self_energy, static_self_energy, static_potential

def remove_se_degeneracy(emb, se, dtol=1e-8, etol=1e-6, drop_non_causal=False):

    emb.log.debug("Removing degeneracy in self-energy - degenerate energy tol=%e   evec tol=%e"%(dtol, etol))
    e = se.energies
    couplings_l, couplings_r = se._unpack_couplings()
    e_new, slices = get_unique(e, atol=dtol)#
    emb.log.debug("Number of energies = %d,  unique = %d"%(len(e),len(e_new)))
    energies, couplings = [], []
    warn_non_causal = False
    for i, s in enumerate(slices):
        mat = np.einsum('pa,qa->pq', couplings_l[:,s], couplings_r[:,s]).real
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

        emb.log.debug("    | E = %e << %s"%(e_new[i],e[s]))
        emb.log.debug("       evals: %s"%val)
        emb.log.debug("       kept:  %s"%(val[idx]))
    if warn_non_causal:
        emb.log.warning("Non-causal poles found in self-energy")
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