import numpy as np

from vayesta.core.fragmentation.fragmentation import Fragmentation

# TODO: Allow different indices for alpha and beta

class Fragmentation_UHF(Fragmentation):
    """Fragmentation for unrestricted HF."""

    @property
    def nmo(self):
        return (self.mo_coeff[0].shape[-1],
                self.mo_coeff[1].shape[-1])

    #def check_orth(self, mo_coeff, mo_name=None, *args, **kwargs):
    #    results = []
    #    for s, spin in enumerate(('alpha', 'beta')):
    #        results.append(super().check_orth(mo_coeff[s], '%s-%s' % (spin[0], mo_name), *args, **kwargs))
    #    return tuple(zip(*results))

    def get_frag_coeff(self, indices):
        """Get fragment coefficients for a given set of orbital indices."""
        c_frag = (self.coeff[0][:,indices].copy(),
                  self.coeff[1][:,indices].copy())
        return c_frag

    def get_env_coeff(self, indices):
        """Get environment coefficients for a given set of orbital indices."""
        env = [np.ones((self.coeff[0].shape[-1]), dtype=bool),
               np.ones((self.coeff[1].shape[-1]), dtype=bool)]
        env[0][indices] = False
        env[1][indices] = False
        c_env = (self.coeff[0][:,env[0]].copy(),
                 self.coeff[1][:,env[1]].copy())
        return c_env
