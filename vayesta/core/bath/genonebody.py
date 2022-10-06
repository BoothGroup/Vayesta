import numpy as np

from vayesta.core.linalg import recursive_block_svd
from vayesta.core.util import *
from vayesta.core import spinalg
from .bath import Bath
from . import helper

class Gen_1b_Bath_RHF(Bath):
    """ Bath orbitals defined from an arbitrary set of full-system one-body objects,
    in order to reproduce the space optimally required to span these objects. 

    Passing in the Fock matrix will construct the 'power' orbitals of EwDMET.

    Note that this requires the dmet space (todo in future: Allow this to be optional, 
        and include dmet space in one-body objects)
    """

    def __init__(self, fragment, dmet_bath, occtype, svd_tol=1e-8, depth=None, oneb_mats=None, \
            covariant=None, entanglement_coupling='fragment', *args, **kwargs):
        """
        Parameters
        ----------
        depth : list, optional
            Provides the depth of the recursion for each one-body matrix passed in
            (i.e. the order to which the entanglement is spanned for that object)

        oneb_mats : list[ndarray]
            The one-body objects for which the bath is constructed. Note that this is
            order dependent, since it will span the entanglement of the lower indexed
            arrays first. These should be in the AO representation.

        covariant : list[bool]
            Indicates whether the corresponding matrix in oneb_mats list is in a fully 
            co-variant repr (True, e.g. Fock or self-energy moms) or fully contra-variant
            repr (False, e.g. DMs or GF moms). Default True for all matrices.
        """
        super().__init__(fragment, *args, **kwargs)
        self.dmet_bath = dmet_bath
        if occtype not in ('occupied', 'virtual'):
            raise ValueError("Invalid occtype: %s" % occtype)
        self.occtype = occtype
        # Whether to consider coupling to previous bath spaces as well as fragment, or just fragment
        self.entanglement_coupling = entanglement_coupling
        # Set the recursion depth for the bath generation
        if oneb_mats is None:
            raise ValueError("No one-body matrices passed in for Gen_1b_Bath")
        if type(oneb_mats) is not list:
            raise ValueError("One-body matrices must be passed in to Gen_1b_Bath as list")
        self.len_oneb = len(oneb_mats)
        self.oneb_mats = oneb_mats
        if covariant is None:
            # If we are not told whether the matrices are co- or contra-variant,
            # assume that they are all covariant matrices (i.e. Fock, self-energy moms)
            self.covariant = [True]*self.len_oneb
        else:
            self.covariant = covariant
        assert(len(self.covariant)==self.len_oneb)
        if depth is None:
            # By default, change so that the recursion depth is one for the last
            # matrix passed in, and increases by one for every lower matrix
            depth = list(range(self.len_oneb,0,-1))
        self.depth = depth
        if type(self.depth) is not list:
            raise ValueError("depth must be a list in Gen_1b_Bath")
        assert (len(self.depth) == self.len_oneb)
        self.svd_tol = svd_tol # Singular value cutoff
        # Coefficients and occupations
        self.coeff, self.nbath_svd, self.sing_vals, self.orders  = self.kernel()

    @property
    def c_cluster_occ(self):
        """Occupied DMET cluster orbitals. """
        return self.dmet_bath.c_cluster_occ

    @property
    def c_cluster_vir(self):
        """Virtual DMET cluster orbitals."""
        return self.dmet_bath.c_cluster_vir

    @property
    def c_env(self):
        if self.occtype == 'occupied':
            return self.dmet_bath.c_env_occ
        if self.occtype == 'virtual':
            return self.dmet_bath.c_env_vir

    @property
    def ncluster(self):
        if self.occtype == 'occupied':
            return self.dmet_bath.c_cluster_occ.shape[-1]
        if self.occtype == 'virtual':
            return  self.dmet_bath.c_cluster_vir.shape[-1]

    def kernel(self):
        '''
        Creates generalized SVD bath spaces of self.occtype, via sequential decomposition of series of one-body operators.

        Returns:

            coeff : ndarray
                Coefficients of occupied or virtual space in AO basis, where the DMET cluster
                (of appropriate type) comes first, followed by nbath_svd new bath orbitals, then env
            nbath_svd : int
                Number of bath orbitals of appropriate type added

            svs : ndarray
                Singular values of bath states added

            orders_full : ndarray
                Order of the decomposition (of respective matrix) which gave rise to bath orbital

            mat_svd : ndarray
                Index of the matrix that was decomposed to give this bath orbital
        '''

        c_env = self.c_env
        if self.spin_restricted and (c_env.shape[-1] == 0):
            return c_env, np.zeros(0)
        if self.spin_unrestricted and (c_env[0].shape[-1] + c_env[1].shape[-1] == 0):
            return c_env, tuple(2*[np.zeros(0)])

        t_init = timer()
        self.log.info("Making %s generalized bath orbitals, with svd tol= %.3e", self.occtype.capitalize(),self.svd_tol)
        self.log.info("-------%s----------------------------------------------", len(self.occtype)*'-')
        self.log.changeIndentLevel(1)

        if self.occtype == 'occupied':
            c_clust = self.c_cluster_occ
        else:
            c_clust = self.c_cluster_vir
        c_frag = self.c_frag
        s = self.base.get_ovlp()

        if self.entanglement_coupling == 'fragment':
            # If we are just projecting to the fragment space, then we want to create a space
            # to define the matrix of the fragment \oplus occ/vir environment (not DMET or previous bath spaces)
            # Note that this does not now span the full occ/vir space
            coeff = spinalg.hstack_matrices(c_frag, c_env)
            norb = coeff.shape[-1]
            nproj_space = nproj_space0 = c_frag.shape[-1]
        elif self.entanglement_coupling == 'cluster':
            # Left projecting the operators into the full cluster space (inc. DMET and previous bath spaces found).
            # This space will therefore grow as you increase the number of operators.
            coeff = spinalg.hstack_matrices(c_clust, c_env)    # Full occ / vir of system
            norb = coeff.shape[-1]
            nproj_space = nproj_space0 = c_clust.shape[-1]
        else:
            raise NotImplementedError('entanglement_coupling= %s' % self.entanglement_coupling)
        c_svd_bath = None

        nbath_svd = 0
        mat_svd = None 
        orders_full = None 
        svs = None 

        # Loop through set of matrices provided
        for mat_ind in range(self.len_oneb):

            self.log.info("Decomposing matrix {} of {} up to recursion depth {}...".format(mat_ind+1, self.len_oneb, self.depth[mat_ind]))
            self.log.changeIndentLevel(1)

            if self.covariant[mat_ind]:
                mat_1b = dot(coeff.T, self.oneb_mats[mat_ind], coeff)
            else:
                mat_1b = dot(coeff.T, s, self.oneb_mats[mat_ind], s, coeff)
            # mo_svd returns the orbitals in the basis of the occupied environment states
            # sv returns the singular values of all bath orbitals (zero if environment)
            # orders returns the order of the bath orbitals (inf if environment)
            mo_svd, sv, orders, weights = recursive_block_svd(mat_1b, n=nproj_space,    \
                    tol=self.svd_tol, maxblock=self.depth[mat_ind])

            nbath_order = [np.count_nonzero(np.isclose(orders, float(x))) for x in range(1,self.depth[mat_ind]+1)]
            nbath = np.sum(nbath_order)
            nbath_svd += nbath
            trunc_thresh = np.full((nbath,), 0.0)
            cum_weights = np.full((nbath,), 0.0)

            for i in range(nbath):
                order = int(round(orders[i]))
                if order == 1:
                    trunc_thresh[i] = sv[i]
                    cum_weights[i] = np.dot(weights[:,i],weights[:,i])
                elif order > 1:
                    mask_prev_ord = np.isclose(orders, float(order-1))
                    for j in range(nbath_order[order-2]):
                        cum_weights[i] = cum_weights[mask_prev_ord][j] * sv[mask_prev_ord][j] * weights[j,i]**2
                        trunc_thresh[i] = cum_weights[i] * sv[i]
                else:
                    raise ValueError

            # Store the singular values, the order depth of each bath, and which matrix the bath was derived from
            if svs is None:
                svs = sv[np.isfinite(orders)]
                orders_full = orders[np.isfinite(orders)]
                mat_svd = np.asarray([mat_ind]*nbath)
            else:
                svs = np.concatenate((svs, sv[np.isfinite(orders)]))
                orders_full = np.concatenate((orders_full, orders[np.isfinite(orders)]))
                mat_svd = np.concatenate((mat_svd, np.asarray([mat_ind]*nbath)))
            assert(mat_svd.shape[0] == orders_full.shape[0])

            for i in range(self.depth[mat_ind]):
                if nbath_order[i] == 0:
                    self.log.info("At recursion level {}, we get no more bath orbitals, so cannot go to the full recursion depth desired".format(i+1))
                    self.log.info("If this is not the full space, then decrease svd_tol to continue to grow the bath space...")
                    break
                self.log.info("From recursion {}, we get {} {} bath orbitals (of possible {})".format(i+1, nbath_order[i], self.occtype, nproj_space))
                self.log.info("SVs = %r",sv[np.isclose(orders, float(i+1))])
            self.log.changeIndentLevel(-1)
            
            # Rotate to AO
            c = dot(c_env, mo_svd)
            # Update the environment space to consist of the singular right vectors, and bath space found
            c_env = c[:, nbath:]
            if c_svd_bath:
                c_svd_bath = spinalg.hstack_matrices(c_svd_bath, c[:,:nbath])
            else:
                c_svd_bath = c[:,:nbath]

            if self.entanglement_coupling == 'cluster':
                # The first nbath of these now join the cluster space, with the rest being the env
                c_clust = spinalg.hstack_matrices(c_clust, c_svd_bath)
                nproj_space += nbath # The space we are projecting onto increases with each order
                coeff = spinalg.hstack_matrices(c_clust, c_env)
                assert(coeff.shape[-1] == norb)
            elif self.entanglement_coupling == 'fragment':
                coeff = spinalg.hstack_matrices(c_frag, c_env)


        self.log.changeIndentLevel(-1)
        if self.entanglement_coupling == 'fragment':
            # Ensure that coeff returns the full space. This is always true for the 'cluster' coupling, since
            # the bath is included in the coeff definition.
            # For fragment coupling, the c_clust is still just the DMET space.
            coeff = spinalg.hstack_matrices(c_clust, c_svd_bath, c_env)
        self.log.timing("Time gen SVD bath:  total= %s", timer()-t_init)
        
        return coeff, nbath_svd, svs, orders_full

    def get_bath(self):
        ''' This will return the additional Gen SVD bath orbitals that have been constructed in the kernel,
        and return the bath orbitals, and environment orbitals in the AO basis.
        These are just constructed to be the orbitals indexed directly after the DMET bath space.'''

        if self.occtype == 'occupied':
            ndmet_clust = self.dmet_bath.c_cluster_occ.shape[-1]
        else:
            ndmet_clust = self.dmet_bath.c_cluster_vir.shape[-1]
        full_clust = self.nbath_svd + ndmet_clust
        return self.coeff[:,ndmet_clust:full_clust], self.coeff[:,full_clust:]
