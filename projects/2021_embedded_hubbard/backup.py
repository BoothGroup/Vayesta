import copy
import numpy as np

import pyscf
import pyscf.lib

import vayesta
import vayesta.ewf
import vayesta.lattmod
from vayesta.core.util import einsum

from vayesta.misc import brueckner, PCDIIS
from vayesta.dmet import pdmet

def get_e_dmet(f):
    """DMET (projected DM) energy of fragment f."""
    c = f.c_active
    p = f.get_fragment_projector(c)
    pdm1 = einsum('ix,xj,ai,bj->ab', p, f.results.dm1, c, c)
    pdm2 = einsum('ix,xjkl,ai,bj,ck,dl->abcd', p, f.results.dm2, c, c, c, c)
    e1 = np.einsum('ij,ij->', mf.get_hcore(), pdm1)
    e2 = np.einsum('iiii,iiii->', mf.mol.get_eri(), pdm2)/2
    return e1 + e2

class SelfConsistentMF:

    def __init__(self, mf, sc_type, mo_coeff=None, tol=1e-6, maxiter=200, damping=0, diis=True):
        self.mf = copy.copy(mf)
        if mo_coeff is not None:
            self.mf.mo_coeff = mo_coeff
        if sc_type is not None and sc_type.lower() not in('pdmet', 'brueckner'):
            raise ValueError("Unknown self-consistency type: %r", sc_type)
        self.sc_type = sc_type
        self.tol = tol
        self.maxiter = (1 if sc_type is None else maxiter)
        self.damping = damping
        self.diis = diis
        self.converged = False
        self.iteration = 0

    @property
    def nsite(self):
        return self.mf.mol.nsite

    @property
    def nelectron(self):
        return self.mf.mol.nsite

    @property
    def filling(self):
        return self.nelectron / self.nsite

    def run_fci_fragments(self, mf, nimp, *args, use_symmetry=True, **kwargs):
        fci = vayesta.ewf.EWF(mf, fragment_type='Site', bath_type=None, solver='FCI', make_rdm1=True, make_rdm2=True)
        if use_symmetry:
            f = fci.make_atom_fragment(list(range(nimp)))
            fci.kernel()
            assert f.results.converged
            frags = f.make_tsymmetric_fragments(tvecs=[self.nsite//nimp, 1, 1])
        else:
            for s in range(0, self.nsite, nimp):
                f = fci.make_atom_fragment(list(range(s, s+nimp)))
            fci.kernel()
            for f in fci.fragments:
                assert f.results.converged
        assert len(fci.fragments) == self.nsite//nimp
        return fci.fragments

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

            fci_frags = self.run_fci_fragments(self.mf, nimp)
            energy = self.get_energy(fci_frags)
            if self.sc_type:
                print("Iteration %3d: E(MF)= %16.8f E(tot)= %16.8f dDM= %16.8f" % (self.iteration, e_mf, energy, ddm))
            if ddm < (1-self.damping)*self.tol:
                print("%s converged in %d iterations." % (self.__class__.__name__, self.iteration))
                self.converged = True
                break
            self.mf = self.update_mf(self.mf, fci_frags, diis)
        else:
            if self.sc_type is not None:
                print("%s not converged!" % self.__class__.__name__)

        imp = fci_frags[0].c_active[:nimp]
        docc = einsum("ijkl,xi,xj,xk,xl->x", fci_frags[0].results.dm2, *(4*[imp]))/2
        return energy, docc

    def get_energy(self, frags):
        # DMET energy
        e = np.asarray([get_e_dmet(f) for f in frags])
        assert abs(e.min()-e.max()) < 1e-6
        e = np.sum(e) / self.nelectron
        return e

    def update_mf(self, *args, **kwargs):
        if self.sc_type is None:
            return
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


# ==================================================================================================

nsite = 12
nimp = 2
doping = 0
u_min = 0
u_max = 12
u_step = 1
do_fci = (nsite <= 12)

mo_pdmet = None
mo_brueck = None

for uidx, hubbard_u in enumerate(range(u_min, u_max+1, u_step)):
    print("Hubbard-U= %4.1f" % hubbard_u)
    print("===============")
    mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nsite+doping, hubbard_u=hubbard_u)
    mf = vayesta.lattmod.LatticeMF(mol)
    mf.kernel()

    # Full system FCI
    if do_fci:
        fci = vayesta.ewf.EWF(mf, solver='FCI', make_rdm1=True, make_rdm2=True, bath_type=None, fragment_type='Site')
        f = fci.make_atom_fragment(list(range(nsite)))
        fci.kernel()
        if f.results.converged:
            e_exact = fci.e_tot / mol.nelectron
            docc_exact = np.einsum("ijkl,i,j,k,l->", f.results.dm2, *(4*[f.c_active[0]]))/2
        else:
            print("Full-system FCI not converged.")
            e_exact = docc_exact = np.nan
        print("E(exact)= %.8f" % e_exact)
        print("Docc(exact)= %.8f" % docc_exact)
    else:
        e_exact = docc_exact = np.nan

    oneshot = SelfConsistentMF(mf, sc_type=None)
    e_oneshot, docc_oneshot = oneshot.kernel(nimp=nimp)

    sc_pdmet = SelfConsistentMF(mf, sc_type='pdmet')
    e_pdmet, docc_pdmet = sc_pdmet.kernel(nimp=nimp, mo_coeff=mo_pdmet)
    mo_pdmet = sc_pdmet.mf.mo_coeff

    sc_brueck = SelfConsistentMF(mf, sc_type='brueckner')
    e_brueck, docc_brueck = sc_brueck.kernel(nimp=nimp, mo_coeff=mo_brueck)
    mo_brueck = sc_brueck.mf.mo_coeff

    with open('energies.txt', 'a') as f:
        if uidx == 0:
            f.write("%12s  %16s  %16s  %16s  %16s\n" % ("# Hubbard-U", "Exact", "Oneshot", "PDMET", "Brueckner"))
        f.write('%12.1f  %16.8f  %16.8f  %16.8f  %16.8f\n' % (hubbard_u, e_exact, e_oneshot, e_pdmet, e_brueck))

    with open('docc.txt', 'a') as f:
        if uidx == 0:
            f.write("%12s  %16s  %16s  %16s  %16s\n" % ("# Hubbard-U", "Exact", "Oneshot", "PDMET", "Brueckner"))
        s = nimp // 2 # Take site at the center
        f.write('%12.1f  %16.8f  %16.8f  %16.8f  %16.8f\n' % (hubbard_u, docc_exact, docc_oneshot[s], docc_pdmet[s], docc_brueck[s]))
