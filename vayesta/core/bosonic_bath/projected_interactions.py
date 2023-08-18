



class BosonicHamiltonianProjector:
    def __init__(self, cluster, df):
        self.cluster = cluster
        self.df = df
        assert(self.cluster.inc_bosons)
        self._cderi_ov = None

    @property
    def cderi_mo(self):
        if self._cderi_mo is None:
            self._cderi_ov = self.df.cderi_ov


        pass


    def project_hamiltonian(self, freq_exchange=False):
        pass


    def project_couplings(self, inc_exchange=True):
        pass

    def project_freqs(self, inc_exchange=False):
        pass





