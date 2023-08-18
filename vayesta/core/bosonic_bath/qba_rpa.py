from vayesta.core.bath.bath import Bath
from vayesta.core.bosonic_bath.bbath import Bosonic_Bath
from vayesta.core.util import dot, einsum
import numpy as np


class RPA_Boson_Target_Space(Bath):
    """Class to obtain the target excitation space from which we'll construct our bosonic bath.
    This can either start from either the DMET or fully extended cluster, and can be optionally projected onto
    excitations local to the fragment in either just the occupied space or both the occupied and virtual spaces.
    """
    def __init__(self, fragment, target_orbitals="full", local_projection='fragment'):
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
            return dot(self.mo_coeff_occ.T, self.ovlp, self.fragment.cluster.c_active_occ), \
                dot(self.mo_coeff_vir.T, self.ovlp, self.fragment.cluster.c_active_vir)
        elif self.target_orbitals == "dmet":
            return dot(self.mo_coeff_occ.T, self.ovlp, self.fragment._dmet_bath.c_cluster_occ), \
                dot(self.mo_coeff_vir.T, self.ovlp, self.fragment._dmet_bath.c_cluster_vir)
        else:
            raise ValueError("Unknown target orbital requested.")

    def get_c_loc(self):
        if "fragment" in self.local_projection:
            if len(self.local_projection) == 8 or self.local_projection[-3:] == "occ":
                return dot(self.mo_coeff_occ.T, self.ovlp, self.fragment.cluster.c_active_occ), None
            elif self.local_projection[-2:] == "ov":
                return dot(self.mo_coeff_occ.T, self.ovlp, self.fragment.cluster.c_active_occ), \
                    dot(self.mo_coeff_vir.T, self.ovlp, self.fragment.cluster.c_active_vir)
        elif self.local_projection is None:
            return None, None
        raise ValueError("Unknown fragment projection requested.")

    def gen_target_excitation(self):
        """Generate the targeted excitation space for a given fragment"""
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

    def _get_local_orbitals(self):
        c_loc_occ = None
        c_loc_vir = None

        if "fragment" in self.local_projection:
            if len(self.local_projection) == 8 or self.local_projection[-3:] == "occ":
                c_loc_occ = self.fragment.cluster.c_active_occ
            elif self.local_projection[-2:] == "ov":
                c_loc_occ = self.fragment.cluster.c_active_occ
                c_loc_vir = self.fragment.cluster.c_active_vir
            else:
                raise ValueError("Unknown local projection requested.")
        return c_loc_occ, c_loc_vir


class RPA_QBA_Bath(Bosonic_Bath):
    def __init__(self, fragment, target_m0):
        self.target_m0 = target_m0
        super().__init__(fragment)

    def make_boson_coeff(self):
        # Generate full local fermionic excitation space.
        clus_ov = self.cluster_excitations
        # Remove any contributions within the fermionic excitation space of the fragment.
        m0_env = self.target_m0 - dot(dot(self.target_m0, clus_ov), clus_ov.T)
        # Now we can construct the bosonic bath by diagonlising these contributions.
        contribs = dot(m0_env, m0_env.T)
        occup, c = np.linalg.eigh(contribs)
        occuprtinv = occup ** (-0.5)

        coeff = dot(m0_env.T, c * occuprtinv[None])
        self.coeff = coeff
        self.occup = occup
        return coeff, occup
