import pyscf.cc
import pyscf.cc.ccsd_rdm
import pyscf.ci

import vayesta
import vayesta.ewf
from vayesta.core.util import einsum
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


class TestProjectedCIWaveFunction(TestCase):
    """Regression tests for issue #222.

    CISD and FCI cluster wave functions have to be converted to CCSD-type amplitudes *before* the
    fragment projection is applied: the conversion T2 = C2/C0 - T1 T1 is non-linear and does not
    commute with the projection, so converting the stored projected wave function afterwards
    corrupts the amplitudes, the global amplitudes and the density matrices derived from them.
    """

    systems = {"CISD": testsystems.water_631g, "FCI": testsystems.water_sto3g}

    @classmethod
    def setUpClass(cls):
        cls.embs = {}

    @classmethod
    def tearDownClass(cls):
        del cls.embs

    @classmethod
    def get_emb(cls, solver):
        """Embedding with full bath, such that each cluster spans the full system."""
        if solver not in cls.embs:
            mf = cls.systems[solver].rhf()
            emb = vayesta.ewf.EWF(
                mf, solver=solver, bath_options=dict(bathtype="full"), solver_options=dict(conv_tol=1e-12)
            )
            with emb.iao_fragmentation() as f:
                f.add_all_atomic_fragments()
            emb.kernel()
            cls.embs[solver] = emb
        return cls.embs[solver]

    def get_cisd_amplitudes(self):
        ci = pyscf.ci.CISD(self.systems["CISD"].rhf())
        ci.conv_tol = 1e-12
        ci.kernel()
        c0, c1, c2 = ci.cisdvec_to_amplitudes(ci.ci)
        t1 = c1 / c0
        t2 = c2 / c0 - einsum("ia,jb->ijab", t1, t1)
        return ci, t1, t2

    def test_pwf_equals_convert_then_project(self):
        """The stored projected wave function must equal `wf.as_ccsd().project(p)`."""
        for solver in ("CISD", "FCI"):
            emb = self.get_emb(solver)
            for x in emb.fragments:
                proj = x.get_overlap("frag|cluster-occ")
                ref = x.results.wf.as_ccsd().project(proj)
                pwf = x.results.pwf.as_ccsd()
                self.assertAllclose(pwf.t1, ref.t1)
                self.assertAllclose(pwf.t2, ref.t2)
                self.assertAllclose(pwf.l1, ref.l1)
                self.assertAllclose(pwf.l2, ref.l2)

    def test_cisd_full_bath(self):
        """With a full bath, global amplitudes, energies and 1-RDM must reproduce the full-system CISD."""
        ci, t1, t2 = self.get_cisd_amplitudes()
        emb = self.get_emb("CISD")
        # Global amplitudes
        self.assertAllclose(emb.get_global_t1(), t1)
        self.assertAllclose(emb.get_global_t2(), t2)
        # Energies
        self.assertAllclose(emb.e_tot, ci.e_tot)
        self.assertAllclose(emb.get_ccsd_energy(), ci.e_tot)
        self.assertAllclose(emb.get_ccsd_energy(full_wf=True), ci.e_tot)
        # Global 1-RDM (CCSD expression with Lambda = T)
        dm1 = pyscf.cc.ccsd_rdm.make_rdm1(pyscf.cc.CCSD(ci._scf), t1, t2, t1, t2)
        self.assertAllclose(emb._make_rdm1_ccsd_global_wf(), dm1)
        self.assertAllclose(emb._make_rdm1_ccsd_global_wf(slow=True), dm1)

    def test_fci_full_bath(self):
        """With a full bath, the projected T-amplitude energies must reproduce the full-system FCI."""
        fci = self.systems["FCI"].rfci()
        emb = self.get_emb("FCI")
        self.assertAllclose(emb.e_tot, fci.e_tot)
        self.assertAllclose(emb.get_ccsd_energy(), fci.e_tot)
        self.assertAllclose(emb.get_ccsd_energy(full_wf=True), fci.e_tot)

    def test_as_ccsd_of_projected_cisd_raises(self):
        """Converting a projected CISD wave function to CCSD cannot be done correctly and must raise."""
        emb = self.get_emb("CISD")
        x = emb.fragments[0]
        proj = x.get_overlap("frag|cluster-occ")
        pwf = x.results.wf.as_cisd().project(proj)
        with self.assertRaises(NotImplementedError):
            pwf.as_ccsd()
