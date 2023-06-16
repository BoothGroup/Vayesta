"""Routines specialised for computation of zeroth moment in the case of an RHF reference with dRPA interactions."""


from vayesta.rpa.rirpa.momzero_NI import (NIMomZero, MomzeroDeductNone, MomzeroDeductD, MomzeroDeductHigherOrder)


class NIMomZero(NIMomZero):
    """All provided quantities are now in spatial orbitals. This actually only requires an additional factor in the
    get_Q method."""
    def get_Q(self, freq):
        # Have equal contributions from both spin channels.
        return 2 * super().get_Q(freq)

class MomzeroDeductNone_dRHF(MomzeroDeductNone):
    """All provided quantities are now in spatial orbitals. This actually only requires an additional factor in the
    get_Q method."""
    def get_Q(self, freq):
        # Have equal contributions from both spin channels.
        return 2 * super().get_Q(freq)

class MomzeroDeductD_dRHF(MomzeroDeductD):
    """All provided quantities are now in spatial orbitals. This actually only requires an additional factor in the
    get_Q method."""
    def get_Q(self, freq):
        # Have equal contributions from both spin channels.
        return 2 * super().get_Q(freq)

class MomzeroDeductHigherOrder_dRHF(MomzeroDeductHigherOrder):
    """All provided quantities are now in spatial orbitals. This actually only requires an additional factor in the
    get_Q method."""
    def get_Q(self, freq):
        # Have equal contributions from both spin channels.
        return 2 * super().get_Q(freq)
