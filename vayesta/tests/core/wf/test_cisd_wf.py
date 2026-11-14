import pytest
import numpy as np

from vayesta.core.types import SpinOrbitals, UCISD_WaveFunction


@pytest.mark.fast
@pytest.mark.parametrize("nocc", [(3, 2), (2, 2)], ids=["unequal-spin-dimensions", "equal-spin-dimensions"])
def test_ucisd_as_ccsd_four_spin_blocks(nocc):
    """Convert BA independently of AB, even when their shapes happen to match."""
    rng = np.random.default_rng(241)
    norb = 5
    nocca, noccb = nocc
    nvira, nvirb = norb - nocca, norb - noccb
    mo = SpinOrbitals((np.eye(norb), np.eye(norb)), occ=nocc)

    t1a = rng.random((nocca, nvira))
    t1b = rng.random((noccb, nvirb))
    t2ab = rng.random((nocca, noccb, nvira, nvirb))
    t2ba = t2ab.transpose(1, 0, 3, 2).copy()

    # Construct CI coefficients from known CC amplitudes in both mixed-spin orders.
    c0 = 0.8
    c1 = (c0 * t1a, c0 * t1b)
    c2 = (
        np.zeros((nocca, nocca, nvira, nvira)),
        c0 * (t2ab + np.einsum("ia,jb->ijab", t1a, t1b)),
        c0 * (t2ba + np.einsum("ia,jb->ijab", t1b, t1a)),
        np.zeros((noccb, noccb, nvirb, nvirb)),
    )
    cc = UCISD_WaveFunction(mo, c0, c1, c2).as_ccsd()

    assert len(cc.t2) == 4
    np.testing.assert_allclose(cc.t1a, t1a)
    np.testing.assert_allclose(cc.t1b, t1b)
    np.testing.assert_allclose(cc.t2ab, t2ab)
    np.testing.assert_allclose(cc.t2ba, t2ba)
