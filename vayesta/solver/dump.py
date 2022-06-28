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
            grp.attrs['id'] = self.fragment.id
            grp.attrs['name'] = self.fragment.name
            grp.attrs['norb'] = self.cluster.norb_active
            grp.attrs['nocc'] = self.cluster.nocc_active
            grp.attrs['nvir'] = self.cluster.nvir_active
            if self.spinsym == 'restricted':
                # Orbital coefficients
                grp.create_dataset('c_cluster', data=self.cluster.c_active)
                grp.create_dataset('c_frag', data=self.fragment.c_frag)
                # Integrals
                grp.create_dataset('hcore', data=self.get_hcore())
                grp.create_dataset('heff', data=self.get_heff(eris=eris))
                grp.create_dataset('fock', data=self.get_fock())
                grp.create_dataset('eris', data=eris)
            elif self.spinsym == 'unrestricted':
                # Orbital coefficients
                grp.create_dataset('c_cluster_a', data=self.cluster.c_active[0])
                grp.create_dataset('c_cluster_b', data=self.cluster.c_active[1])
                grp.create_dataset('c_frag_a', data=self.fragment.c_frag[0])
                grp.create_dataset('c_frag_b', data=self.fragment.c_frag[1])
                # Integrals
                hcorea, hcoreb = self.get_hcore()
                grp.create_dataset('hcore_a', data=hcorea)
                grp.create_dataset('hcore_b', data=hcoreb)
                heffa, heffb = self.get_heff(eris=eris)
                grp.create_dataset('heff_a', data=heffa)
                grp.create_dataset('heff_b', data=heffb)
                focka, fockb = self.get_fock()
                grp.create_dataset('fock_a', data=focka)
                grp.create_dataset('fock_b', data=fockb)
                erisaa, erisab, erisbb = eris
                grp.create_dataset('eris_aa', data=erisaa)
                grp.create_dataset('eris_ab', data=erisab)
                grp.create_dataset('eris_bb', data=erisbb)
            else:
                raise NotImplementedError
