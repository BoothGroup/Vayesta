import os.path
import numpy as np

import pyscf
import pyscf.lo

from vayesta.core.util import *
from .fragmentation import Fragmentation
from .ufragmentation import Fragmentation_UHF

# Load default minimal basis set on module initialization
default_minao = {}
path = os.path.dirname(__file__)
with open(os.path.join(path, 'minao.dat'), 'r') as f:
    for line in f:
        if line.startswith('#'): continue
        (basis, minao) = line.split()
        if minao == 'none': minao = None
        default_minao[basis] = minao

def get_default_minao(basis):
    # TODO: Add more to data file
    if not isinstance(basis, str):
        return 'minao'
    bas = basis.replace('-', '').lower()
    minao = default_minao.get(bas, 'minao')
    if minao is None:
        raise ValueError("Could not chose minimal basis for basis %s automatically!", basis)
    return minao

class IAO_Fragmentation(Fragmentation):

    name = "IAO"

    def __init__(self, *args, minao='auto', **kwargs):
        super().__init__(*args, **kwargs)
        if minao.lower() == 'auto':
            minao = get_default_minao(self.mol.basis)
            self.log.info("IAO:  computational basis= %s  minimal reference basis= %s (automatically chosen)", self.mol.basis, minao)
        else:
            self.log.debug("IAO:  computational basis= %s  minimal reference basis= %s", self.mol.basis, minao)
        self.minao = minao
        self.refmol = pyscf.lo.iao.reference_mol(self.mol, minao=self.minao)

    @property
    def n_iao(self):
        return self.refmol.nao

    def get_coeff(self, mo_coeff=None, mo_occ=None, add_virtuals=True):
        """Make intrinsic atomic orbitals (IAOs).

        Returns
        -------
        c_iao : (n(AO), n(IAO)) array
            Orthonormalized IAO coefficients.
        """
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ
        ovlp = self.get_ovlp()

        c_occ = mo_coeff[:,mo_occ>0]
        c_iao = pyscf.lo.iao.iao(self.mol, c_occ, minao=self.minao)
        n_iao = c_iao.shape[-1]
        self.log.info("n(AO)= %4d  n(MO)= %4d  n(occ-MO)= %4d  n(IAO)= %4d",
                mo_coeff.shape[0], mo_coeff.shape[-1], c_occ.shape[-1], n_iao)

        # Orthogonalize IAO using symmetric (Lowdin) orthogonalization
        x, e_min = self.symmetric_orth(c_iao, ovlp)
        self.log.debug("Lowdin orthogonalization of IAOs: n(in)= %3d -> n(out)= %3d , e(min)= %.3e", x.shape[0], x.shape[1], e_min)
        if e_min < 1e-12:
            self.log.warning("Small eigenvalue in Lowdin orthogonalization: %.3e !", e_min)
        c_iao = np.dot(c_iao, x)
        # Check that all electrons are in IAO space
        self.check_nelectron(c_iao, c_occ)
        if add_virtuals:
            c_vir = self.get_virtual_coeff(c_iao, mo_coeff=mo_coeff)
            c_iao = np.hstack((c_iao, c_vir))
        # Test orthogonality of IAO
        self.check_orthonormal(c_iao)
        return c_iao

    def check_nelectron(self, c_iao, c_occ):
        dm = np.dot(c_occ, c_occ.T)
        ovlp = self.get_ovlp()
        #print(c_occ.shape)
        #print(c_iao.shape)
        ne_iao = einsum('ai,ab,bc,cd,di->', c_iao, ovlp, dm, ovlp, c_iao)
        ne_tot = einsum('ab,ab->', dm, ovlp)
        if abs(ne_iao - ne_tot) > 1e-8:
            self.log.error("IAOs do not contain the correct number of electrons: IAO= %.8f  total= %.8f", ne_iao, ne_tot)
        else:
            self.log.debugv("Number of electrons: IAO= %.8f  total= %.8f", ne_iao, ne_tot)
        return ne_iao

    def get_labels(self):
        """Get labels of IAOs.

        Returns
        -------
        iao_labels : list of length nIAO
            Orbital label (atom-id, atom symbol, nl string, m string) for each IAO.
        """
        iao_labels_refmol = self.refmol.ao_labels(None)
        self.log.debugv('iao_labels_refmol: %r', iao_labels_refmol)
        if self.refmol.natm == self.mol.natm:
            iao_labels = iao_labels_refmol
        # If there are ghost atoms in the system, they will be removed in refmol.
        # For this reason, the atom IDs of mol and refmol will not agree anymore.
        # Here we will correct the atom IDs of refmol to agree with mol
        # (they will no longer be contiguous integers).
        else:
            ref2mol = []
            for refatm in range(self.refmol.natm):
                ref_coords = self.refmol.atom_coord(refatm)
                for atm in range(self.mol.natm):
                    coords = self.mol.atom_coord(atm)
                    if np.allclose(coords, ref_coords):
                        self.log.debugv('reference cell atom %r maps to atom %r', refatm, atm)
                        ref2mol.append(atm)
                        break
                else:
                    raise RuntimeError("No atom found with coordinates %r" % ref_coords)
            iao_labels = []
            for iao in iao_labels_refmol:
                iao_labels.append((ref2mol[iao[0]], iao[1], iao[2], iao[3]))
        self.log.debugv('iao_labels: %r', iao_labels)
        assert (len(iao_labels_refmol) == len(iao_labels))
        return iao_labels

    def search_labels(self, labels):
        return self.refmol.search_ao_label(labels)

    def get_virtual_coeff(self, c_iao, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        ovlp = self.get_ovlp()
        # Add remaining virtual space, work in MO space, so that we automatically get the
        # correct linear dependency treatment, if n(MO) < n(AO)
        c_iao_mo = dot(mo_coeff.T, ovlp, c_iao)
        # Get eigenvectors of projector into complement
        p_iao = np.dot(c_iao_mo, c_iao_mo.T)
        p_rest = np.eye(p_iao.shape[-1]) - p_iao
        e, c = np.linalg.eigh(p_rest)

        # Corresponding expression in AO basis (but no linear-dependency treatment):
        # p_rest = ovlp - ovlp.dot(c_iao).dot(c_iao.T).dot(ovlp)
        # e, c = scipy.linalg.eigh(p_rest, ovlp)
        # c_rest = c[:,e>0.5]

        # Ideally, all eigenvalues of P_env should be 0 (IAOs) or 1 (non-IAO)
        # Error if > 1e-3
        mask_iao, mask_rest = (e <= 0.5), (e > 0.5)
        e_iao, e_rest = e[mask_iao], e[mask_rest]
        if np.any(abs(e_iao) > 1e-3):
            self.log.error("CRITICAL: Some IAO eigenvalues of 1-P_IAO are not close to 0:\n%r", e_iao)
        elif np.any(abs(e_iao) > 1e-6):
            self.log.warning("Some IAO eigenvalues e of 1-P_IAO are not close to 0: n= %d max|e|= %.2e",
                    np.count_nonzero(abs(e_iao) > 1e-6), abs(e_iao).max())
        if np.any(abs(1-e_rest) > 1e-3):
            self.log.error("CRITICAL: Some non-IAO eigenvalues of 1-P_IAO are not close to 1:\n%r", e_rest)
        elif np.any(abs(1-e_rest) > 1e-6):
            self.log.warning("Some non-IAO eigenvalues e of 1-P_IAO are not close to 1: n= %d max|1-e|= %.2e",
                    np.count_nonzero(abs(1-e_rest) > 1e-6), abs(1-e_rest).max())

        if not (np.sum(mask_rest) + c_iao.shape[-1] == mo_coeff.shape[-1]):
            self.log.critical("Error in construction of remaining virtual orbitals! Eigenvalues of projector 1-P_IAO:\n%r", e)
            self.log.critical("Number of eigenvalues above 0.5 = %d", np.sum(mask_rest))
            self.log.critical("Total number of orbitals = %d", mo_coeff.shape[-1])
            raise RuntimeError("Incorrect number of remaining virtual orbitals")
        c_rest = np.dot(mo_coeff, c[:,mask_rest])        # Transform back to AO basis
        c_rest = fix_orbital_sign(c_rest)[0]

        self.check_orthonormal(np.hstack((c_iao, c_rest)), "IAO+virtual orbital")
        return c_rest

class IAO_Fragmentation_UHF(Fragmentation_UHF, IAO_Fragmentation):

    def get_coeff(self, mo_coeff=None, mo_occ=None, add_virtuals=True):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ

        self.log.info("Alpha-IAOs:")
        c_iao_a = IAO_Fragmentation.get_coeff(self, mo_coeff=mo_coeff[0], mo_occ=mo_occ[0], add_virtuals=add_virtuals)
        self.log.info(" Beta-IAOs:")
        c_iao_b = IAO_Fragmentation.get_coeff(self, mo_coeff=mo_coeff[1], mo_occ=mo_occ[1], add_virtuals=add_virtuals)
        return (c_iao_a, c_iao_b)
