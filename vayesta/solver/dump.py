import dataclasses
import h5py
import vayesta
from vayesta.core import spinalg
from .solver import ClusterSolver


class DumpSolver(ClusterSolver):

    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
        filename: str = None

    def kernel(self, *args, eris=None, **kwargs):
        fragment = self.fragment
        cluster = self.cluster

        if isinstance(self.opts.filename, str):
            h5file = h5py.File(self.opts.filename, 'a')
        else:
            h5file = self.opts.filename
        if eris is None:
            eris = self.get_eris()
        with h5file as f:
            grp = h5file.create_group('fragment_%d' % fragment.id)
            # Attributes
            grp.attrs['id'] = fragment.id
            grp.attrs['name'] = fragment.name
            grp.attrs['norb'] = cluster.norb_active
            grp.attrs['nocc'] = cluster.nocc_active
            grp.attrs['nvir'] = cluster.nvir_active
            c_dmet_cluster_occ = fragment._dmet_bath.c_cluster_occ
            c_dmet_cluster_vir = fragment._dmet_bath.c_cluster_vir
            c_dmet_cluster = spinalg.hstack_matrices(c_dmet_cluster_occ, c_dmet_cluster_vir)
            if self.spinsym == 'restricted':
                grp.attrs['norb_dmet_cluster'] = c_dmet_cluster.shape[-1]
                grp.attrs['nocc_dmet_cluster'] = c_dmet_cluster_occ.shape[-1]
                grp.attrs['nvir_dmet_cluster'] = c_dmet_cluster_vir.shape[-1]
                # Orbital coefficients
                grp.create_dataset('c_frag', data=fragment.c_frag)
                grp.create_dataset('c_dmet_cluster', data=c_dmet_cluster)
                grp.create_dataset('c_cluster', data=cluster.c_active)
                # Integrals
                grp.create_dataset('hcore', data=self.get_hcore())
                grp.create_dataset('heff', data=self.get_heff(eris=eris))
                grp.create_dataset('fock', data=self.get_fock())
                grp.create_dataset('eris', data=eris)
            elif self.spinsym == 'unrestricted':
                grp.attrs['norb_dmet_cluster'] = [c_dmet_cluster[s].shape[-1] for s in range(2)]
                grp.attrs['nocc_dmet_cluster'] = [c_dmet_cluster_occ[s].shape[-1] for s in range(2)]
                grp.attrs['nvir_dmet_cluster'] = [c_dmet_cluster_vir[s].shape[-1] for s in range(2)]
                # Orbital coefficients
                grp.create_dataset('c_frag_a', data=fragment.c_frag[0])
                grp.create_dataset('c_frag_b', data=fragment.c_frag[1])
                grp.create_dataset('c_dmet_cluster_a', data=c_dmet_cluster[0])
                grp.create_dataset('c_dmet_cluster_b', data=c_dmet_cluster[1])
                grp.create_dataset('c_cluster_a', data=cluster.c_active[0])
                grp.create_dataset('c_cluster_b', data=cluster.c_active[1])
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
