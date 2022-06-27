import dataclasses
from .solver import ClusterSolver
import h5py


class DumpSolver(ClusterSolver):

    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
        filename: str = None

    def kernel(self, *args, eris=None, **kwargs):
        if isinstance(self.opts.filename, str):
            h5file = h5py.File(self.opts.filename, 'a')
        else:
            h5file = self.opts.filename
        if eris is None:
            eris = self.get_eris()
        with h5file as f:
            grp = h5file.create_group('fragment_%d' % self.fragment.id)
            # Attributes
            grp.attrs['name'] = self.fragment.name
            grp.attrs['norb'] = self.cluster.norb_active
            grp.attrs['nocc'] = self.cluster.nocc_active
            grp.attrs['nvir'] = self.cluster.nvir_active
            # Orbital coefficients
            grp.create_dataset('c_cluster', data=self.cluster.c_active)
            grp.create_dataset('c_frag', data=self.fragment.c_frag)
            # Integrals
            eris = self.base.get_eris_array(self.cluster.c_active)
            grp.create_dataset('hcore', data=self.get_hcore())
            grp.create_dataset('heff', data=self.get_heff(eris=eris))
            grp.create_dataset('fock', data=self.get_fock())
            grp.create_dataset('eris', data=eris)

    def get_eris(self, *args, **kwargs):
        return self.base.get_eris_array(self.cluster.c_active)
