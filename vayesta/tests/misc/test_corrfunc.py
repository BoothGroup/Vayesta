import pytest
import numpy as np
import pyscf.fci

from vayesta.misc import corrfunc
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase


def _fci_rdms(mf):
    """Spin-resolved FCI 1- and 2-RDMs in the MO basis."""
    fci = pyscf.fci.FCI(mf)
    fci.kernel()
    norb = (
        mf.mo_coeff[0].shape[-1]
        if np.ndim(mf.mo_coeff[0]) == 2
        else mf.mo_coeff.shape[-1]
    )
    nelec = mf.mol.nelec
    return fci.make_rdm12s(fci.ci, norb, nelec)


@pytest.mark.fast
class TestSpinSpinZ(TestCase):
    def test_unrestricted_doublet(self):
        """<S_z S_z> of a doublet with M_s=1/2 is 1/4."""
        mf = testsystems.h3_sto3g.uhf()
        dm1, dm2 = _fci_rdms(mf)
        ssz = corrfunc.spinspin_z_unrestricted(dm1, dm2)
        self.assertAlmostEqual(ssz, 0.25)

    def test_unrestricted_identity_projector(self):
        mf = testsystems.h3_sto3g.uhf()
        dm1, dm2 = _fci_rdms(mf)
        norb = dm1[0].shape[-1]
        proj = np.eye(norb)
        ssz = corrfunc.spinspin_z_unrestricted(dm1, dm2)
        ssz_proj = corrfunc.spinspin_z_unrestricted(dm1, dm2, proj1=(proj, proj))
        self.assertAlmostEqual(ssz, ssz_proj)

    def test_unrestricted_singlet(self):
        """<S_z S_z> of a singlet is 0."""
        mf = testsystems.h4_sto3g.rhf()
        dm1, dm2 = _fci_rdms(mf)
        ssz = corrfunc.spinspin_z_unrestricted(dm1, dm2)
        self.assertAlmostEqual(ssz, 0.0)
