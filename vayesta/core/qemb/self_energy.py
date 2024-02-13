"""Routines to reconstruct the full system self-energy from cluster spectral moments"""

import numpy as np

from vayesta.core.util import NotCalculatedError, Object, dot, einsum
from dyson import Lehmann, MBLGF, MixedMBLGF, NullLogger

def make_self_energy_1proj(emb, use_sym=True, eta=1e-2, se_degen_tol=1e-6, se_eval_tol=1e-6, drop_non_causal=False):
    """
    Reconstruct self energy from cluster spectral moments using 1 projector

    TODO: MPI, SVD

    Parameters
    ----------
    emb : EWF object
        Embedding object
    use_sym : bool
        Use symmetry to reconstruct self-energy
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
    couplings, energies = [], []

    fragments = emb.get_fragments(sym_parent=None) if use_sym else emb.get_fragments()
    for i, f in enumerate(fragments):
        # Calculate self energy from cluster moments
        th, tp = f.results.moms

        vth, vtp = th.copy(), tp.copy()
        solverh = MBLGF(th, log=NullLogger())
        solverp = MBLGF(tp, log=NullLogger())
        solver = MixedMBLGF(solverh, solverp)
        solver.kernel()

        se = solver.get_self_energy()

        energies_f = se.energies
        couplings_f = se.couplings

        ovlp = emb.get_ovlp()
        mc = f.get_overlap('mo|cluster')
        mf = f.get_overlap('mo|frag')
        fc = f.get_overlap('frag|cluster')
        cfc = fc.T @ fc
        
        # Fock matrix in cluster basis
        fock_cls = f.cluster.c_active.T @ ovlp @ fock @ ovlp @ f.cluster.c_active
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
        sym_coup = einsum('pa,qa->apq', np.dot(cfc, se.couplings) , se.couplings) 
        sym_coup = 0.5 * (sym_coup + sym_coup.transpose(0,2,1))
        
        rank = 2 # rank / multiplicity 
        tol = 1e-12
        couplings_cf, energies_cf = [], []
        for a in range(sym_coup.shape[0]):
            m = sym_coup[a]
            val, vec = np.linalg.eigh(m)
            idx = np.abs(val) > tol
            assert (np.abs(val) > tol).sum() <= rank
            w = vec[:,idx] @ np.diag(np.sqrt(val[idx], dtype=np.complex64))
            couplings_cf.append(w)
            energies_cf += [se.energies[a] for e in range(idx.sum())]

        couplings_cf = np.hstack(couplings_cf)

        couplings.append(mc @ couplings_cf)
        energies.append(energies_cf)

        if use_sym:
            for child in f.get_symmetry_children():
                static_potential += child.cluster.c_active @ v_frag @ child.cluster.c_active.T
                mc_child = child.get_overlap('mo|frag')
                static_self_energy += mc_child @ static_self_energy_frag @ mc_child.T
                couplings.append(mc_child @ couplings_cf.T)
                energies.append(energies_cf)

    couplings = np.hstack(couplings)
    energies = np.concatenate(energies)
    self_energy = Lehmann(energies, couplings)

    self_energy = remove_se_degeneracy(emb, self_energy, dtol=se_degen_tol, etol=se_eval_tol, drop_non_causal=drop_non_causal)

    return self_energy, static_self_energy, static_potential


def make_self_energy_2proj(emb, use_sym=True, eta=1e-2):
    """
    Reconstruct self energy from cluster spectral moments using 2 projectors

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

        vth, vtp = th.copy(), tp.copy()
        solverh = MBLGF(th, log=NullLogger())
        solverp = MBLGF(tp, log=NullLogger())
        solver = MixedMBLGF(solverh, solverp)
        solver.kernel()

        se = solver.get_self_energy()

        energies_f = se.energies
        couplings_f = se.couplings

        ovlp = emb.get_ovlp()
        mc = f.get_overlap('mo|cluster')
        mf = f.get_overlap('mo|frag')
        fc = f.get_overlap('frag|cluster')
        cfc = fc.T @ fc
        
        # Fock matrix in cluster basis
        fock_cls = f.cluster.c_active.T @ ovlp @ fock @ ovlp @ f.cluster.c_active
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
                static_self_energy += mf_child @ static_se_frag, mf_child.T
                couplings.append(mf_child @ fc_child @ se.couplings)
                energies.append(se.energies)

    couplings = np.hstack(couplings)
    energies = np.concatenate(energies)
    self_energy = Lehmann(energies, couplings)

    return self_energy, static_self_energy, static_potential

def remove_se_degeneracy(emb, se, dtol=1e-8, etol=1e-6, drop_non_causal=False):

    emb.log.info("Removing degeneracy in self-energy - degenerate energy tol=%e   evec tol=%e"%(dtol, etol))
    e, v = se.energies, se.couplings
    e_new, slices = get_unique(e, atol=dtol)#
    emb.log.info("Number of energies = %d,  unique = %d"%(len(e),len(e_new)))
    energies, couplings = [], []
    for i, s in enumerate(slices):
        mat = einsum('pa,qa->pq', v[:,s], v[:,s]).real
        val, vec = np.linalg.eigh(mat)
        if  drop_non_causal:
            idx = val > etol
        else:
            idx = np.abs(val) > etol
        if np.sum(val[idx] < -etol) > 0:
            emb.log.warning("Large negative eigenvalues - non-causal self-energy")
        w = vec[:,idx] @ np.diag(np.sqrt(val[idx], dtype=np.complex64))
        couplings.append(w)
        energies += [e_new[i] for _ in range(idx.sum())]

        emb.log.info("    | E = %e << %s"%(e_new[i],e[s]))
        emb.log.info("       evals: %s"%val)
        emb.log.info("       kept:  %s"%(val[idx]))
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