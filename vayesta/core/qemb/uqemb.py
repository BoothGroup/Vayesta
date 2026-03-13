import numpy as np

import pyscf
import pyscf.mp
import pyscf.ci
import pyscf.cc

from vayesta.core.qemb.qemb import Embedding
from vayesta.core.qemb.ufragment import UFragment

from vayesta.core.ao2mo.postscf_ao2mo import postscf_ao2mo
from vayesta.core.util import dot, einsum, log_method, with_doc
from vayesta.core import spinalg
from vayesta.core.ao2mo import kao2gmo_cderi
from vayesta.core.ao2mo import postscf_kao2gmo_uhf
from vayesta.mpi import mpi
from vayesta.core.qemb.corrfunc import get_corrfunc_unrestricted

from vayesta.core.qemb.rdm import make_rdm1_demo_uhf
from vayesta.core.qemb.rdm import make_rdm2_demo_uhf


class UEmbedding(Embedding):
    """Spin unrestricted quantum embedding."""

    # Shadow this in inherited methods:
    Fragment = UFragment

    # Deprecated:
    is_rhf = False
    is_uhf = True
    # Use instead:
    spinsym = "unrestricted"

    def _check_orthonormal(self, *mo_coeff, mo_name="", **kwargs):
        mo_coeff = spinalg.hstack_matrices(*mo_coeff)
        results = []
        for s, spin in enumerate(("alpha", " beta")):
            name_s = "-".join([spin, mo_name])
            res_s = super()._check_orthonormal(mo_coeff[s], mo_name=name_s, **kwargs)
            results.append(res_s)
        return tuple(zip(*results))

    @log_method()
    def get_eris_array_uhf(self, mo_coeff, mo_coeff2=None, compact=False):
        """Get electron-repulsion integrals in MO basis as a NumPy array.

        Parameters
        ----------
        mo_coeff: tuple(2) of (n(AO), n(MO)) array
            MO coefficients.

        Returns
        -------
        eris:
            Electron-repulsion integrals in MO basis.
        """
        if mo_coeff2 is None:
            mo_coeff2 = mo_coeff
        moa, mob = mo_coeff
        mo2a, mo2b = mo_coeff2
        # PBC with k-points:
        if self.kdf is not None:
            if compact:
                raise NotImplementedError
            cderia, cderia_neg = kao2gmo_cderi(self.kdf, (moa, mo2a))
            cderib, cderib_neg = kao2gmo_cderi(self.kdf, (mob, mo2b))
            eris_aa = einsum("Lij,Lkl->ijkl", cderia.conj(), cderia)
            eris_ab = einsum("Lij,Lkl->ijkl", cderia.conj(), cderib)
            eris_bb = einsum("Lij,Lkl->ijkl", cderib.conj(), cderib)
            if cderia_neg is not None:
                eris_aa -= einsum("Lij,Lkl->ijkl", cderia_neg.conj(), cderia_neg)
                eris_ab -= einsum("Lij,Lkl->ijkl", cderia_neg.conj(), cderib_neg)
                eris_bb -= einsum("Lij,Lkl->ijkl", cderib_neg.conj(), cderib_neg)
            return (eris_aa, eris_ab, eris_bb)

        eris_aa = super().get_eris_array((moa, mo2a, moa, mo2a), compact=compact)
        eris_ab = super().get_eris_array((moa, mo2a, mob, mo2b), compact=compact)
        eris_bb = super().get_eris_array((mob, mo2b, mob, mo2b), compact=compact)
        return (eris_aa, eris_ab, eris_bb)

    @log_method()
    def get_eris_object(self, postscf, fock=None):
        """Get ERIs for post-SCF methods.

        For folded PBC calculations, this folds the MO back into k-space
        and contracts with the k-space three-center integrals..

        Parameters
        ----------
        postscf: one of the following post-SCF methods: MP2, CCSD, RCCSD, DFCCSD
            Post-SCF method with attribute mo_coeff set.

        Returns
        -------
        eris: _ChemistsERIs
            ERIs which can be used for the respective post-SCF method.
        """
        if fock is None:
            if isinstance(postscf, pyscf.mp.mp2.MP2):
                fock = self.get_fock()
            elif isinstance(postscf, (pyscf.ci.cisd.CISD, pyscf.cc.ccsd.CCSD)):
                fock = self.get_fock(with_exxdiv=False)
            else:
                raise ValueError("Unknown post-HF method: %r", type(postscf))
        # For MO energies, always use get_fock():
        act = postscf.get_frozen_mask()
        mo_act = (postscf.mo_coeff[0][:, act[0]], postscf.mo_coeff[1][:, act[1]])
        mo_energy = (
            einsum("ai,ab,bi->i", mo_act[0], self.get_fock()[0], mo_act[0]),
            einsum("ai,ab,bi->i", mo_act[1], self.get_fock()[1], mo_act[1]),
        )
        e_hf = self.mf.e_tot

        # 1) Fold MOs into k-point sampled primitive cell, to perform efficient AO->MO transformation:
        if self.kdf is not None:
            eris = postscf_kao2gmo_uhf(postscf, self.kdf, fock=fock, mo_energy=mo_energy, e_hf=e_hf)
            return eris
        # 2) Regular AO->MO transformation
        eris = postscf_ao2mo(postscf, fock=fock, mo_energy=mo_energy, e_hf=e_hf)
        return eris

    @log_method()
    @with_doc(Embedding.build_screened_eris)
    def build_screened_eris(self, *args, **kwargs):
        if self.opts.ext_rpa_correction is not None:
            raise NotImplementedError("External RPA correlation energy only implemented for restricted references!")
        super().build_screened_eris(*args, **kwargs)

    def check_fragment_symmetry(self, dm1, charge_tol=1e-6, spin_tol=1e-6):
        frags = self.get_symmetry_child_fragments(include_parents=True)
        for group in frags:
            parent, children = group[0], group[1:]
            for child in children:
                charge_err, spin_err = parent.get_symmetry_error(child, dm1=dm1)
                if (charge_err > charge_tol) or (spin_err > spin_tol):
                    raise RuntimeError(
                        "%s and %s not symmetric: charge error= %.3e spin error= %.3e !"
                        % (parent.name, child.name, charge_err, spin_err)
                    )
                self.log.debugv(
                    "Symmetry between %s and %s: charge error= %.3e spin error= %.3e",
                    parent.name,
                    child.name,
                    charge_err,
                    spin_err,
                )

    def _check_fragment_nelectron(self):
        nelec_frags = (
            sum([f.sym_factor * f.nelectron[0] for f in self.loop()]),
            sum([f.sym_factor * f.nelectron[1] for f in self.loop()]),
        )
        self.log.info("Total number of mean-field electrons over all fragments= %.8f , %.8f", *nelec_frags)
        if abs(nelec_frags[0] - np.rint(nelec_frags[0])) > 1e-4 or abs(nelec_frags[1] - np.rint(nelec_frags[1])) > 1e-4:
            self.log.warning("Number of electrons not integer!")
        return nelec_frags

    # --- Other
    # ---------

    @log_method()
    @with_doc(make_rdm1_demo_uhf)
    def make_rdm1_demo(self, *args, **kwargs):
        return make_rdm1_demo_uhf(self, *args, **kwargs)

    @log_method()
    @with_doc(make_rdm2_demo_uhf)
    def make_rdm2_demo(self, *args, **kwargs):
        return make_rdm2_demo_uhf(self, *args, **kwargs)

    def pop_analysis(self, dm1, mo_coeff=None, local_orbitals="lowdin", write=True, minao="auto", mpi_rank=0, **kwargs):
        # IAO / PAOs are spin dependent - we need to build them here:
        if isinstance(local_orbitals, str) and local_orbitals.lower() == "iao+pao":
            local_orbitals = self.get_lo_coeff("iao+pao", minao=minao)
        pop = []
        for s, spin in enumerate(("alpha", "beta")):
            mo = mo_coeff[s] if mo_coeff is not None else None
            lo = local_orbitals if isinstance(local_orbitals, str) else local_orbitals[s]
            pop.append(super().pop_analysis(dm1[s], mo_coeff=mo, local_orbitals=lo, write=False, **kwargs))
        if write and (mpi.rank == mpi_rank):
            self.write_population(pop, **kwargs)
        return pop

    def get_atomic_charges(self, pop):
        charges = np.zeros(self.mol.natm)
        spins = np.zeros(self.mol.natm)
        for i, label in enumerate(self.mol.ao_labels(fmt=None)):
            charges[label[0]] -= pop[0][i] + pop[1][i]
            spins[label[0]] += pop[0][i] - pop[1][i]
        charges += self.mol.atom_charges()
        return charges, spins

    get_corrfunc = log_method()(get_corrfunc_unrestricted)
