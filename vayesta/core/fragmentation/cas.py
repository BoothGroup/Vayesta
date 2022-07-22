import numpy as np

from .fragmentation import Fragmentation
from .ufragmentation import Fragmentation_UHF


class CAS_Fragmentation(Fragmentation):
    """Fragmentation into mean-field states."""

    name = "CAS"

    def get_coeff(self):
        return self.mo_coeff.copy()

    def get_labels(self):
        return [("", "", "MF", str(x)) for x in range(0, self.nmo)]

    def get_atom_indices_symbols(self, *args, **kwargs):
        raise NotImplementedError("Atomic fragmentation is not compatible with CAS fragmentation")

    # Need to overload this function since only accept integer specification in this case.
    def get_orbital_indices_labels(self, orbitals):
        if isinstance(orbitals[0], (int, np.integer)):
            orbital_indices = orbitals
            orbital_labels = (np.asarray(self.labels, dtype=object)[orbitals]).tolist()
            orbital_labels = [('%s%3s %s%-s' % tuple(l)).strip() for l in orbital_labels]
            return orbital_indices, orbital_labels
        raise ValueError("A list of integers is required! orbitals= %r" % orbitals)

    def add_cas_fragment(self, ncas, nelec, name=None, degen_tol=1e-10):
        """Create a single fragment containing a CAS.

        Parameters
        ----------
        ncas: int
            Number of spatial orbitals within the fragment.
        nelec: int
            Number of electrons within the fragment.
        name: str, optional
            Name for the fragment. If None, a name is automatically generated from the orbital indices. Default: None.
        """

        if self.emb.is_rhf:
            occ = self.emb.mo_occ
        else:
            occ = sum(self.emb.mo_occ)

        if nelec > sum(occ):
            raise ValueError("CAS specified with more electrons than present in system.")
        if ncas > len(occ):
            raise ValueError("CAS specified with more orbitals than present in system.")
        # Search for how many orbital pairs we have to include to obtain desired number of electrons.
        # This should be stable regardless of occupancies etc.
        anyocc = np.where(occ > 0)[0]
        offset, nelec_curr = -1, 0
        while nelec_curr < nelec:
            offset += 1
            nelec_curr += int(occ[anyocc[-1] - offset])

        if nelec_curr > nelec or offset > ncas:
            raise ValueError(
                "Cannot create CAS with required properties around Fermi level with current MO occupancy.")

        def check_for_degen(energies, po, pv, name=""):
            if po > 0:
                ogap = energies[po] - energies[po - 1]
                self.log.info(name+"CAS occupied energy gap: %e (= %e - %e)", ogap, energies[po], energies[po - 1])
            elif po == 0:
                self.log.info(name+"CAS contains all occupied orbitals.")
                ogap = np.inf
            else:
                # Shouldn't reach this as would require CAS to have more electrons than the full system.
                raise ValueError("CAS would contain more electrons than full system.")

            if ogap < degen_tol:
                raise ValueError("Requested {:s}CAS splits degenerate occupied orbitals.".format(name))

            try:
                vgap = energies[pv] - energies[pv - 1]
            except IndexError:
                assert(pv == len(energies))
                self.log.info(" "*len(name)+"    contains all virtual orbitals.")
                vgap = np.inf
            else:
                self.log.info(" "*len(name)+"    virtual energy gap: %e (= %e - %e)", vgap, energies[pv], energies[pv - 1])
            if vgap < degen_tol:
                raise ValueError("Requested CAS splits degenerate virtual orbitals.")

        if self.emb.is_rhf:
            check_for_degen(self.emb.mo_energy, anyocc[-1] - offset, anyocc[-1] - offset + ncas)
        else:
            check_for_degen(self.emb.mo_energy[0], anyocc[-1] - offset, anyocc[-1] - offset + ncas, "alpha ")
            check_for_degen(self.emb.mo_energy[1], anyocc[-1] - offset, anyocc[-1] - offset + ncas, "beta ")

        orbs = list(range(anyocc[-1] - offset, anyocc[-1] - offset + ncas))
        self.add_orbital_fragment(orbs, name=name)


class CAS_Fragmentation_UHF(Fragmentation_UHF, CAS_Fragmentation):

    def get_labels(self):
        return [("", "", "HF", str(x)) for x in range(0, self.nmo[0])]
