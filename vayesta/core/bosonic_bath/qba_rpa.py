from vayesta.core.bath.bath import Bath
from vayesta.core.bosonic_bath.bbath import Bosonic_Bath
from vayesta.core.util import dot, einsum
import numpy as np


class RPA_Boson_Target_Space(Bath):
    """Class to obtain the target excitation space from which we'll construct our bosonic bath.
    This can either start from either the DMET or fully extended cluster, and can be optionally projected onto
    excitations local to the fragment in either just the occupied space or both the occupied and virtual spaces.
    """

    def __init__(self, fragment, target_orbitals="full", local_projection="fragment"):
        self.target_orbitals = target_orbitals
        self.local_projection = local_projection
        super().__init__(fragment)
        if not self.spin_restricted:
            raise NotImplementedError("Spin-unrestricted RPA bosonic bath not yet implemented.")

    @property
    def mo_coeff_occ(self):
        return self.fragment.base.mo_coeff_occ

    @property
    def mo_coeff_vir(self):
        return self.fragment.base.mo_coeff_vir

    @property
    def ovlp(self):
        return self.fragment.base.get_ovlp()

    def get_c_target(self):
        if self.target_orbitals == "full":
            return dot(self.mo_coeff_occ.T, self.ovlp, self.fragment.cluster.c_active_occ), dot(
                self.mo_coeff_vir.T, self.ovlp, self.fragment.cluster.c_active_vir
            )
        elif self.target_orbitals == "dmet":
            return dot(self.mo_coeff_occ.T, self.ovlp, self.fragment._dmet_bath.c_cluster_occ), dot(
                self.mo_coeff_vir.T, self.ovlp, self.fragment._dmet_bath.c_cluster_vir
            )
        else:
            raise ValueError("Unknown target orbital requested.")

    def get_c_loc(self):
        if self.local_projection is None:
            return None, None
        if "fragment" in self.local_projection:
            if len(self.local_projection) == 8 or self.local_projection[-3:] == "occ":
                return dot(self.mo_coeff_occ.T, self.ovlp, self.fragment.c_frag), None
            elif self.local_projection[-2:] == "ov":
                return dot(self.mo_coeff_occ.T, self.ovlp, self.fragment.c_frag), dot(
                    self.mo_coeff_vir.T, self.ovlp, self.fragment.c_frag
                )
        raise ValueError("Unknown fragment projection requested.")

    def gen_target_excitation(self):
        """Generate the targeted excitation space for a given fragment"""
        self.fragment.log.debug(
            "Fragment %s generating RPA bosonic bath using %s excitations and projection onto %s.",
            self.fragment.id,
            self.target_orbitals,
            self.local_projection,
        )

        # Obtain all values in the equivalent global space.
        c_occ, c_vir = self.get_c_target()
        c_loc_occ, c_loc_vir = self.get_c_loc()

        if c_loc_occ is not None:
            s_occ = dot(c_loc_occ.T, c_occ)
            c_occ = dot(c_occ, s_occ.T, s_occ)
        if c_loc_vir is not None:
            s_vir = dot(c_loc_vir.T, c_vir)
            c_vir = dot(c_vir, s_vir.T, s_vir)

        tar_ss = einsum("iI,aA->IAia", c_occ, c_vir).reshape(-1, c_occ.shape[0] * c_vir.shape[0])
        return np.hstack((tar_ss, tar_ss))

    def _get_target_orbitals(self):
        if self.target_orbitals == "full":
            return self.fragment.c_occ, self.fragment.c_vir
        elif self.target_orbitals == "dmet":
            dmet_bath = self.fragment._dmet_bath
            return dmet_bath.c_cluster_occ, dmet_bath.c_cluster_vir
        raise ValueError("Unknown target orbital requested.")


class RPA_QBA_Bath(Bosonic_Bath):
    # With equal alpha and beta blocks in target_m0 (all restricted targets), the projection, Gram matrix and
    # coefficients are formed on the spatial half, with the cluster projection applied in occupied-virtual product
    # form, and the result duplicated into both spin blocks. Mathematically identical to the spin-orbital path.
    singlet_projection = False
    # Largest difference between the spin blocks, relative to their largest element, treated as equal. The
    # spin-stacked moment path rebuilds the two blocks with round-off differences.
    singlet_projection_rtol = 1e-10

    def __init__(self, fragment, target_m0, singlet_projection: bool | None = None) -> None:
        self.target_m0 = target_m0
        if singlet_projection is not None:
            self.singlet_projection = singlet_projection
        super().__init__(fragment)

    def make_boson_coeff(self):
        if self.singlet_projection:
            out = self._make_boson_coeff_singlet()
            if out is not None:
                return out
        # Generate full local fermionic excitation space.
        clus_ov = self.cluster_excitations
        # Remove any contributions within the fermionic excitation space of the fragment.
        m0_env = self.target_m0 - dot(dot(self.target_m0, clus_ov.T), clus_ov)
        # Now we can construct the bosonic bath by diagonalising these contributions.
        contribs = dot(m0_env, m0_env.T)
        sort = np.s_[::-1]
        occup, c = np.linalg.eigh(contribs)
        occup = occup[sort]
        c = c[:, sort]

        occuprtinv = occup ** (-0.5)
        coeff = dot((c * occuprtinv[None]).T, m0_env)

        self.coeff = coeff
        self.occup = occup

        return coeff, occup

    def _make_boson_coeff_singlet(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Spatial version of `make_boson_coeff` for targets with equal alpha and beta blocks (Restricted).

        With target_m0 = (M, M) and cluster excitations (C, C) / sqrt(2), the environment rows are (E, E) with
        E = M - M C^T C, the Gram matrix is 2 E E^T, and the coefficients are (W^T E, W^T E). C = C_o (x) C_v is
        applied in product form.

        Returns
        -------
        (coeff, occup) or None
            Bath coefficients and coupling weights, or None if the target's spin blocks differ.
        """
        m0 = self.target_m0
        ov = m0.shape[1] // 2
        ma, mb = m0[:, :ov], m0[:, ov:]
        if m0.shape[1] != 2 * ov or np.abs(ma - mb).max(initial=0.0) > self.singlet_projection_rtol * np.abs(m0).max(
            initial=0.0
        ):
            self.log.debug("Spin blocks of the target moment differ; using the spin-orbital bath construction.")
            return None
        co = self.fragment.get_overlap("cluster[occ]|mo[occ]")  # (o_c, o), orthonormal rows
        cv = self.fragment.get_overlap("cluster[vir]|mo[vir]")  # (v_c, v), orthonormal rows
        m = (0.5 * (ma + mb)).reshape(-1, co.shape[1], cv.shape[1])
        proj = einsum("nia,Ii,Aa->nIA", m, co, cv)
        e = (m - einsum("nIA,Ii,Aa->nia", proj, co, cv)).reshape(-1, ov)
        contribs = 2.0 * dot(e, e.T)
        sort = np.s_[::-1]
        occup, c = np.linalg.eigh(contribs)
        occup = occup[sort]
        c = c[:, sort]

        occuprtinv = occup ** (-0.5)
        coeff = dot((c * occuprtinv[None]).T, e)
        coeff = np.hstack((coeff, coeff))

        self.coeff = coeff
        self.occup = occup

        return coeff, occup
