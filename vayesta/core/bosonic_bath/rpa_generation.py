from vayesta.core.bath.bath import Bath
from vayesta.core.util import dot, einsum


class RPA_Boson_Bath(Bath):
    def __init__(self, fragment, target_orbitals = "full", local_projection='fragment'):
        self.target_orbitals = target_orbitals
        self.local_projection = local_projection
        super().__init__(fragment)

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
            else:
                raise ValueError("Unknown fragment projection requested.")
        return None, None

    def make_target_excitations(self):
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

        return einsum("iI,aA->iaIA", c_occ, c_vir).reshape(c_occ.shape[0] * c_vir.shape[0], -1)

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
