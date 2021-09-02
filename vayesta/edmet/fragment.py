from vayesta.dmet.fragment import DMETFragment





class EDMETFragmentExit(Exception):
    pass


class EDMETFragment(DMETFragment):

    @dataclasses.dataclass
    class Options(DMETFragment.Options):
        pass

    def __init__(self, *args, solver=None, **kwargs):
        super().__init__(*args, solver, **kwargs)

    def construct_bosons(self, rpa_moms):

        m0 = rpa_moms[0]
        m1 = rpa_moms[1]




    def kernel(self, rpa_moms, bno_threshold = None, bno_number = None, solver=None, init_guess=None, eris=None, construct_bath = False):
        mo_coeff, mo_occ, nocc_frozen, nvir_frozen, nactive = \
                            self.set_up_orbitals(bno_threshold, bno_number, construct_bath)




