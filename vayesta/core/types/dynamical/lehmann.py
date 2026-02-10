import numpy as np
from abc import ABC, abstractmethod
from dyson import Lehmann
from vayesta.core.util import einsum

from .dynamical import Dynamical, GreensFunction, SelfEnergy
from .moment import GF_MomentRep, SE_MomentRep

class LehmannRep(Dynamical):

    def __init__(self, lehmanns):
        """Initialize Lehmann representation.

        Parameters
        ----------
        lehmanns : list of Lehmann
            List of Lehmann objects for each sector.
        """
        
        if isinstance(lehmanns, Lehmann):
            lehmanns = [lehmanns]

        self._lehmanns = lehmanns

    @property
    def lehmanns(self):
        """List of Lehmann objects for each sector."""
        return self._lehmanns

    @property
    def nsectors(self):
        """Number of sectors in the Lehmann representation."""
        return len(self.lehmanns)
    
    @property
    def hermitian(self):
        ret = True
        for s in range(self.nsectors):
            ret = ret and self.lehmanns[s].hermitian
        return ret

    @property
    def energies(self):
        """Energies of the Lehmann representation."""
        return np.array([self.lehmanns[s].energies for s in range(self.nsectors)])

    @property
    def couplings(self):
        """Couplings of the Lehmann representation."""
        return np.array([self.lehmanns[s].couplings for s in range(self.nsectors)])

    @property
    def naux(self):
        """Number of auxiliary states in the Lehmann representation."""
        return self.energies.shape[0]
    
    @property
    def nphys(self):
        """Number of physical states in the Lehmann representation."""
        return self.couplings.shape[0]

    def _project_lehmanns(self, projector, nproj, img_space=True):
        new_lehmanns = []
        for s in range(self.nsectors):
            if nproj == 1:
                if False:
                    new_energies, new_couplings = project_1_to_fragment_eig(projector, self.lehmanns[s], img_space=img_space)
                else:
                    new_energies, new_couplings_l, new_couplings_r = project_1_to_fragment_svd(projector, self.lehmanns[s], img_space=img_space)
                    new_couplings = np.array([new_couplings_l, new_couplings_r])

            elif nproj == 2:
                new_energies = self.lehmanns[s].energies
                if self.hermitian:
                    new_couplings = projector @ self.lehmanns[s].couplings
                else:
                    coup_l, coup_r = self.lehmanns[s].unpack_couplings()
                    new_couplings_l = projector @ coup_l
                    new_couplings_r = projector @ coup_r
                    new_couplings = np.array([new_couplings_l, new_couplings_r])            
            new_lehmann = Lehmann(new_energies, new_couplings)
            new_lehmanns.append(new_lehmann)
        return new_lehmanns

    def _rotate_lehmanns(self, rotation):
        new_lehmanns = []
        for s in range(self.nsectors):
            new_couplings = rotation @ self.lehmanns[s].couplings
            new_lehmann = Lehmann(self.lehmanns[s].energies, new_couplings)
            new_lehmanns.append(new_lehmann)
        return new_lehmanns


    def _remove_degeneracies(self, hermitian=None, etol=1e-6, dtol=1e-6):
        hermitian = self.hermitian if hermitian is None else hermitian
        new_lehmanns = []
        for s, l in enumerate(self.lehmanns):
            if hermitian:
                new_lehmann = remove_se_degeneracy_sym(l, etol=etol, dtol=dtol)
            else:
                new_lehmann = remove_se_degeneracy_nsym(l, etol=etol, dtol=dtol)
            
            new_lehmanns.append(new_lehmann)
        return new_lehmanns



class GF_LehmannRep(LehmannRep, GreensFunction):

    def __init__(self, lehmanns):
        """Initialize Green's function Lehmann representation.

        Parameters
        ----------
        lehmann : list of Lehmann
            List of Lehmann objects for each sector.

        """

        if isinstance(lehmanns, Lehmann):
            lehmanns = [lehmanns]

        self._lehmanns = lehmanns

    def hermitize(self):
        raise NotImplementedError("Hermitization of Lehmann representation is not implemented.")
    

    def rotate(self, rotation):
        """Change the basis of the Lehmann representation using the given unitary matrix.

        Parameters
        ----------
        rotation : ndarray (nphys, nphys)
            Unitary rotation matrix.
        """
        new_lehmanns = self._rotate_lehmanns(rotation)
        return GF_LehmannRep(new_lehmanns)
    

    def project(self, projector, nproj, img_space=True):
        """Project the Lehmann representation using the given projector.

        Parameters
        ----------
        projector : ndarray (nphys, nproj)
            Projection matrix.
        nproj : int
            Number of indices to project.
        """
        new_lehmanns = self._project_lehmanns(projector, nproj, img_space=img_space)
        return GF_LehmannRep(new_lehmanns)
    
    def to_moments(self, nmom):
        """Convert the Lehmann representation to moment representation.

        Parameters
        ----------
        nmom : int
            Number of moments to compute.

        Returns
        -------
        moments : GF_MomentRep
            Moment representation of the Green's function.
        """
        moments = []
        for s in range(self.nsectors):
            moms = self.lehmanns[s].moments(range(nmom))
            moments.append(moms)
        return GF_MomentRep(moments=np.array(moments), hermitian=self.hermitian)

    def remove_degeneracies(self, hermitian=None, etol=1e-6, dtol=1e-6):
        new_lehmanns = self._remove_degeneracies(hermitian=hermitian, etol=etol, dtol=dtol)
        return GF_LehmannRep(new_lehmanns)
 


class SE_LehmannRep(LehmannRep, SelfEnergy):
    def __init__(self, statics, lehmanns, overlaps=None):
        """Initialize self-energy Lehmann representation.

        Parameters
        ----------
        static : list of ndarray
            List of static parts for each sector.
        lehmann : list of Lehmann
            List of Lehmann objects for each sector.
        overlap : list of ndarray, optional
            List of overlap matrices for each sector. If None, identity overlap is assumed.

        """

        if isinstance(lehmanns, Lehmann):
            lehmanns = [lehmanns]

        self._lehmanns = lehmanns
        self._init_static_overlap(statics, overlaps, hermitian=self.hermitian)

    def hermitize(self):
        raise NotImplementedError("Hermitization of Lehmann representation is not implemented.")
    

    def rotate(self, rotation):
        new_statics, new_overlaps = self._rotate_static_overlap(rotation)
        new_lehmanns = self._rotate_lehmanns(rotation)
        return SE_LehmannRep(new_statics, new_lehmanns, overlaps=new_overlaps)
    
    def project(self, projector, nproj, img_space=True):
        """Project the Lehmann representation using the given projector.

        Parameters
        ----------
        projector : ndarray (nphys, nproj)
            Projection matrix.
        nproj : int
            Number of indices to project.
        """
        proj_lehmanns = self._project_lehmanns(projector, nproj, img_space=img_space)
        proj_statics, proj_overlaps = self._project_static_overlap(projector, nproj)
        return SE_LehmannRep(proj_statics, proj_lehmanns, overlaps=proj_overlaps)


    def to_moments(self, nmom):
        """Convert the Lehmann representation to moment representation.

        Parameters
        ----------
        nmom : int
            Number of moments to compute.

        Returns
        -------
        moments : SE_MomentRep
            Moment representation of the Green's function.
        """
        moments = []
        for s in range(self.nsectors):
            moms = self.lehmanns[s].moments(range(nmom))
            moments.append(moms)
        return SE_MomentRep(self.statics, np.array(moments), overlap=self.overlaps, hermitian=self.hermitian)



    def combine(self, *args):
        """Combine multiple SE_LehmannRep into a single one by summing statics and concatenating poles.

        Parameters
        ----------
        *args : list of SE_LehmannRep
            Lehmann representations to combine.

        Returns
        -------
        combined_lehmann : SE_LehmannRep
            Combined Lehmann representation.
        """
        compatible_shapes = all(arg.nsectors == self.nsectors for arg in args)
        compatible_shapes = compatible_shapes and all(arg.statics.shape == self.statics.shape for arg in args)
        compatible_shapes = compatible_shapes and all(arg.overlaps.shape == self.overlaps.shape for arg in args)
        if not compatible_shapes:
            raise ValueError("All SE_LehmannRep instances must have the same shape to be combined.")

        hermitian = self.hermitian and all(arg.hermitian for arg in args)
        dtype = np.complex128 if not hermitian else np.float64

        combined_overlaps = self.overlaps.astype(dtype)
        combined_statics = self.statics.astype(dtype)
        combined_energies = [self.lehmanns[s].energies.astype(dtype) for s in range(self.nsectors)]
        combined_couplings = [self.lehmanns[s].couplings.astype(dtype) for s in range(self.nsectors)]  #ns * naux

        for arg in args:
            combined_overlaps += arg.overlaps
            combined_statics += arg.statics

            add_energies = [arg.lehmanns[s].energies for s in range(arg.nsectors)]
            add_couplings = [arg.lehmanns[s].couplings for s in range(arg.nsectors)]  #ns * naux

            for s in range(self.nsectors):
                combined_energies[s] = np.concatenate([combined_energies[s], add_energies[s]])
                combined_couplings[s] = np.concatenate([combined_couplings[s], add_couplings[s]], axis=-1)
        
        combined_lehmanns = [Lehmann(combined_energies[s], combined_couplings[s]) for s in range(self.nsectors)]
        return SE_LehmannRep(combined_statics, combined_lehmanns, overlaps=combined_overlaps)
    

    # Sum self-energies or green's functions?
    def combine_sectors(self):
        """Combine all sectors into a single Lehmann representation by summing statics and concatenating poles.

        Returns
        -------
        combined_lehmann : SE_LehmannRep
            Combined Lehmann representation.
        """
        dtype = np.float64 if self.hermitian else np.complex128
        combined_static, combined_overlap = self._combine_static_overlap()
        combined_energies = []
        combined_couplings = []

        for s in range(self.nsectors):
            combined_energies.append(self.lehmanns[s].energies)
            combined_couplings.append(self.lehmanns[s].couplings)

        combined_energies = np.concatenate(combined_energies)
        combined_couplings = np.concatenate(combined_couplings, axis=-1)

        combined_lehmann = Lehmann(combined_energies, combined_couplings)
        return SE_LehmannRep(combined_static, combined_lehmann, overlaps=combined_overlap)

    def remove_degeneracies(self, hermitian=None, etol=1e-6, dtol=1e-6):
        new_lehmanns = self._remove_degeneracies(hermitian=hermitian, etol=etol, dtol=dtol)
        return SE_LehmannRep(self.statics, new_lehmanns, overlaps=self.overlaps)


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

    if log is not None:
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


