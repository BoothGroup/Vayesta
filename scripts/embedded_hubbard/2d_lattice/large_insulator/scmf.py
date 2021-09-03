import copy
import numpy as np

import pyscf
import pyscf.lib
import pyscf.scf


import vayesta
import vayesta.ewf
from vayesta.core.util import einsum

import vayesta.dmet
from vayesta.misc import brueckner, PCDIIS
from vayesta.dmet import pdmet

class SelfConsistentMF:

    def __init__(self, mf, sc_type, tvecs=None, nelectron_target=None, tol=1e-6, maxiter=200, damping=0, diis=True, ab_initio=False):
        self.mf = copy.copy(mf)
        if sc_type is not None and sc_type.lower() not in('pdmet', 'brueckner'):
            raise ValueError("Unknown self-consistency type: %r", sc_type)
        self.sc_type = sc_type
        self.tvecs = tvecs
        self.nelectron_target = nelectron_target
        self.tol = tol
        self.maxiter = (1 if sc_type is None else maxiter)
        self.damping = damping
        self.diis = diis
        self.ab_initio = ab_initio

        # Results
        self.converged = False
        self.iteration = 0
        self.e_dmet = 0
        self.e_ewf = 0
        self.docc = 0

    @property
    def nsite(self):
        if (not self.ab_initio):
            return self.mf.mol.nsite
        else:
            return self.mf.mol.nelectron # Only works for ab-initio Hydrogen rings!

    @property
    def nelectron(self):
        return self.mf.mol.nelectron

    @property
    def filling(self):
        return self.nelectron / self.nsite

    def run_fragments(self, mf, nimp, *args, **kwargs):
    
        fci = None
        # Create fragment solver for EWF
        if (not self.ab_initio):
            fci = vayesta.ewf.EWF(mf, fragment_type='Site', bath_type=None, solver='FCI',
                    make_rdm1=True, make_rdm2=True, nelectron_target=self.nelectron_target, bno_threshold=np.inf, recalc_veff=True)
        else:
            fci = vayesta.ewf.EWF(mf, fragment_type='Site', bath_type=None, solver='FCI',
                    make_rdm1=True, make_rdm2=True, nelectron_target=self.nelectron_target, bno_threshold=np.inf, recalc_veff=True)
        if self.tvecs is not None:

            f = fci.make_atom_fragment(list(range(nimp)))
            fci.kernel()

            assert f.results.converged
            frags = f.make_tsymmetric_fragments(tvecs=self.tvecs)


        else:
            for s in range(0, self.nsite, nimp):
                f = fci.make_atom_fragment(list(range(s, s+nimp)))

            fci.kernel()
            for f in fci.fragments:
                assert f.results.converged
        

        assert len(fci.fragments) == self.nsite//nimp
        

        return fci
    

    def kernel(self, nimp, mo_coeff=None):
        if mo_coeff is not None:
            self.mf.mo_coeff = mo_coeff

        if self.sc_type and self.diis:
            if self.sc_type.lower() == 'pdmet':
                diis = pyscf.lib.diis.DIIS()
            elif self.sc_type.lower() == 'brueckner':
                nocc = np.count_nonzero(self.mf.mo_occ > 0)
                diis = PCDIIS(self.mf.mo_coeff[:,:nocc].copy())
        else:
            diis = None
        
        dm0 = None
        for self.iteration in range(1, self.maxiter+1):
            e_mf = self.mf.e_tot / self.nelectron
            if dm0 is not None:
                ddm = abs(self.mf.make_rdm1() - dm0).max()
            else:
                ddm = np.inf
            dm0 = self.mf.make_rdm1()

            fci = self.run_fragments(self.mf, nimp)
            e_dmet = self.get_energy(fci.fragments) / self.nelectron
            
            print(self.sc_type)
            if (self.sc_type is not None and self.sc_type.lower()=='dmet'):
                e_ewf = fci.e_tot / self.nelectron
            elif (self.sc_type is not None and self.sc_type.lower()=='dmet_ncopt'):
                e_ewf = fci.e_tot /self.nelectron
            else:
                e_ewf = fci.get_e_tot() / self.nelectron
            if self.sc_type:
                print("Iteration %3d: E(MF)= %16.8f E(tot)= %16.8f dDM= %16.8f" % (self.iteration, e_mf, e_dmet, ddm))
            if ddm < (1-self.damping)*self.tol:
                print("%s converged in %d iterations." % (self.__class__.__name__, self.iteration))
                self.converged = True
                break
            
            self.mf = self.update_mf(self.mf, fci.fragments, diis)
        else:
            if self.sc_type is not None:
                print("%s not converged!" % self.__class__.__name__)
            

        # Onsite double occupancies (for all sites)
        docc = np.zeros((self.nsite,))
        for f, s in enumerate(range(0, self.nsite, nimp)):
            imp = fci.fragments[f].c_active[s:s+nimp]
            d = einsum("ijkl,xi,xj,xk,xl->x", fci.fragments[f].results.dm2, *(4*[imp]))/2
            docc[s:s+nimp] = d

        self.e_dmet = e_dmet
        self.e_ewf = e_ewf
        self.docc = docc

        return e_dmet, docc

    def get_energy(self, frags):
        # DMET energy
        e = np.asarray([self.get_e_dmet(f) for f in frags])
        with open('Fragment_energies.txt', 'w') as f:
            f.write(str(e))
        print(e)
        assert abs(e.min()-e.max()) < 1e-6
        e = np.sum(e)
        return e

    def get_e_dmet(self, f):
        """DMET (projected DM) energy of fragment f."""
        c = f.c_active
        p = f.get_fragment_projector(c)
        # For Hubbard, only need the diagonal 2-body RDM entries
        
        pdm1 = einsum('ix,xj,ai,bj->ab', p, f.results.dm1, c, c)
        pdm2 = einsum('ix,xjkl,ai,aj,ak,al->a', p, f.results.dm2, c, c, c, c)
        
        e1 = np.einsum('ij,ij->', self.mf.get_hcore(), pdm1)
        # Calculate diagonal trace
        e2 = self.mf.mol.hubbard_u*np.einsum('i->',pdm2)/2
        return e1 + e2

    def update_mf(self, *args, **kwargs):
        if self.sc_type is None:
            return self.mf
        if self.sc_type.lower() == 'pdmet':
            return self.update_mf_pdmet(*args, **kwargs)
        if self.sc_type.lower() == 'brueckner':
            return self.update_mf_brueckner(*args, **kwargs)
        raise ValueError("Unknown self-consistency type: %r", self.sc_type)

    def update_mf_pdmet(self, mf, frags, diis):
        # Get combined DM
        dm = np.zeros((self.nsite, self.nsite))
        for f in frags:
            pf = f.get_fragment_projector(f.c_active)
            dm += einsum('px,xi,ij,qj->pq', f.c_active, pf, f.results.dm1, f.c_active)
        assert np.isclose(np.trace(dm), mf.mol.nelectron)
        dm = (dm + dm.T)/2  # Symmetrize DM
        mf = pdmet.update_mf(mf, dm1=dm, damping=self.damping, diis=diis)
        return mf


    def update_mf_brueckner(self, mf, frags, diis):
        # Get combined T1 amplitudes
        nocc = np.count_nonzero(mf.mo_occ>0)
        nvir = self.nsite - nocc
        occ = np.s_[:nocc]
        vir = np.s_[nocc:]
        t1 = np.zeros((nocc, nvir))
        for f in frags:
            p = f.get_fragment_projector(f.c_active_occ)
            pt1 = np.dot(p, f.results.c1/f.results.c0)
            # Rotate from cluster basis to MO basis
            ro = np.dot(f.c_active_occ.T, mf.mo_coeff[:,occ])
            rv = np.dot(f.c_active_vir.T, mf.mo_coeff[:,vir])
            t1 += einsum('ia,ip,aq->pq', pt1, ro, rv)
        mf = brueckner.update_mf(mf, t1=t1, damping=self.damping, diis=diis)
        return mf
