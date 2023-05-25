import numpy as np
import vayesta
from vayesta.core.util import energy_string
from vayesta.core.fragmentation.fragmentation import Fragmentation
from vayesta.core.fragmentation.ufragmentation import Fragmentation_UHF


class CAS_Fragmentation(Fragmentation):
    """Fragmentation into mean-field states."""

    name = "CAS"

    def get_coeff(self):
        """Return MO coefficients as "fragment" orbitals."""
        return self.mo_coeff

    def get_labels(self):
        return [("", "", "MO", str(x)) for x in range(0, self.nmo)]

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

    def add_cas_fragment(self, ncas, nelec, name=None, degen_tol=1e-8, **kwargs):
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
            # Log orbital energies
            for i in range(max(po-2, 0), min(pv+2, len(energies))):
                if i < po:
                    orbtype = 'core'
                elif i >= pv:
                    orbtype = 'external'
                else:
                    orbtype = 'CAS'
                if i == po:
                    self.log.info(62*'-')
                self.log.info("MO %4d:  %-8s  occupation= %1d  energy= %s", i, orbtype, occ[i], energy_string(energies[i]))
                if i == (pv-1):
                    self.log.info(62*'-')

            if po > 0:
                ogap = energies[po] - energies[po-1]
                self.log.info("%sCAS occupied energy gap: %s", name, energy_string(ogap))
            elif po == 0:
                self.log.info("%sCAS contains all occupied orbitals.", name)
                ogap = np.inf
            else:
                # Shouldn't reach this as would require CAS to have more electrons than the full system.
                raise ValueError("CAS would contain more electrons than full system.")

            if ogap < degen_tol:
                raise ValueError("Requested %sCAS splits degenerate occupied orbitals." % name)

            try:
                vgap = energies[pv] - energies[pv - 1]
            except IndexError:
                assert(pv == len(energies))
                self.log.info("%sCAS contains all virtual orbitals.", name)
                vgap = np.inf
            else:
                self.log.info("%sCAS virtual  energy gap: %s", name, energy_string(vgap))
            if vgap < degen_tol:
                raise ValueError("Requested CAS splits degenerate virtual orbitals.")

        if self.emb.is_rhf:
            check_for_degen(self.emb.mo_energy, anyocc[-1]-offset, anyocc[-1]-offset+ncas)
        else:
            check_for_degen(self.emb.mo_energy[0], anyocc[-1]-offset, anyocc[-1]-offset+ncas, "alpha ")
            check_for_degen(self.emb.mo_energy[1], anyocc[-1]-offset, anyocc[-1]-offset+ncas, "beta ")

        orbs = list(range(anyocc[-1]-offset, anyocc[-1]-offset+ncas))
        return self.add_orbital_fragment(orbs, name=name, **kwargs)


class CAS_Fragmentation_UHF(Fragmentation_UHF, CAS_Fragmentation):

    def get_labels(self):
        return [("", "", "MO", str(x)) for x in range(0, self.nmo[0])]
