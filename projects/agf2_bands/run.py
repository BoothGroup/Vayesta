"""Get bands for AGF2
"""

from pyscf.pbc import gto, scf, dft
from pyscf.data.nist import HARTREE2EV
from vayesta.misc import gdf
from vayesta.eagf2 import KRAGF2
from gmtkn import GAPS
from ase.dft import kpoints
from vayesta import log
import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
log.setLevel(40)

run = 'run' in sys.argv
debug = 'debug' in sys.argv
plot = 'plot' in sys.argv

# System options:
key = 'LiH'
mesh = [3, 3, 3]
basis = 'gth-szv'

# Path options:
path = 'WGXWLG'
cell_type = 'fcc'
npts = 101
out = '%s_%s_%d%d%d.chk' % (key, basis, *mesh)
ktol = 1e-8

# Plot options:
plot_hf = False
plot_lda = True
plot_agf2 = True
nfreq = 1024
eta_dos = 1.0
eta_bands = 5.0
window = (-30, 60)
cmap = 'afmhot_r'

# AGF2 options:
opts = {
    'damping': 0.66,
    'fock_basis': 'ao',
    'weight_tol': 0.0,
    'extra_cycle': True,
    'conv_tol': 1e-5,
    'conv_tol_rdm1': 1e-10,
    'conv_tol_nelec': 1e-10,
    'conv_tol_nelec_factor': 1e-3,
    'max_cycle': 20,
    'max_cycle_inner': 100,
    'max_cycle_outer': 5,
}

cell = gto.Cell()
cell.atom = list(zip(GAPS[key]['atoms'], GAPS[key]['coords']))
cell.a = GAPS[key]['a']
cell.basis = basis
cell.pseudo = 'gth-pade'
cell.verbose = 0
cell.max_memory = 1e9
cell.build()

# Debug system:
if debug:
    r = 0.75
    cell = gto.Cell()
    cell.atom = f'H 0 0 0; H 0 0 {r}'
    cell.a = [[25, 0, 0], [0, 25, 0], [0, 0, r*2]]
    cell.basis = 'sto3g'
    cell.verbose = 0
    cell.build()
    mesh = [1, 1, 3]
    out = 'h2_debug.chk'
    npts = 103
    path = 'XGX'
    cell_type = 'cubic'

def to_1bz(kpts):
    scaled_kpts = cell.get_scaled_kpts(kpts)
    #scaled_kpts = (scaled_kpts + 0.5) % 1 - 0.5
    scaled_kpts[scaled_kpts >= 1] -= 1
    scaled_kpts[scaled_kpts <  0] += 1
    kpts = cell.get_abs_kpts(scaled_kpts)
    return kpts

points = kpoints.sc_special_points[cell_type]
bandpath = kpoints.get_bandpath([points[p] for p in path], cell.a, npoints=npts)
kpath, sp_points, labels = bandpath.get_linear_kpoint_axis()
band_kpts = bandpath.kpts
band_kpts[:, [1, 2]] = band_kpts[:, [2, 1]]
band_kpts = cell.get_abs_kpts(band_kpts)
band_kpts = to_1bz(band_kpts)

band_khs = [gdf._get_kpt_hash(k, tol=ktol) for k in band_kpts]
print("Total k-points in path: %d" % len(band_kpts))
print("Unique k-points in path: %d" % len(list(set(band_khs))))

if run:
    class PathData:
        def __init__(self, band_kpts):
            self.band_kpts = band_kpts
            self.band_khs = band_khs
            self._data = {k: None for k in self.band_khs}
            self._fh5 = h5py.File(out, 'w')
            self._push_count = 0

        def push(self, kpts=None, data=None):
            """Push the gf which correspond to kpts that are in band_kpts, and
            return the next element in band_kpts not met yet. Returns False if
            all have been recorded.
            """

            if kpts is not None and data is not None:
                kpts = np.array(kpts).reshape(-1, 3)
                kpts = to_1bz(kpts)
                khs = [gdf._get_kpt_hash(k, tol=ktol) for k in kpts]

                for i, kh in enumerate(khs):
                    if kh in self._data:
                        if self._data[kh] is None:
                            self._data[kh] = data[i]
                            self._push_count += 1
                            #TODO this can be done in O(nk) with another dict...
                            for j, kh_j in enumerate(self.band_khs):
                                if kh == kh_j:
                                    for k1 in data[i].keys():
                                        for k2 in data[i][k1].keys():
                                            self._fh5['%d/%s/%s' % (j, k1, k2)] = data[i][k1][k2]

            for i, kh in enumerate(self.band_khs):
                if self._data[kh] is None:
                    return self.band_kpts[i]
            return False

        @property
        def gf(self):
            return [self._data[kh] for kh in self.band_khs]

        def close(self):
            if hasattr(self, '_fh5'):
                self._fh5.close()

    path_data = PathData(band_kpts)
    kpt = band_kpts[0]

    while True:
        print("Running calculations shifted to %s" % kpt)
        kpts = cell.make_kpts(mesh, scaled_center=cell.get_scaled_kpts(kpt))
        kpts = to_1bz(kpts)

        with_df = gdf.GDF(cell, kpts)
        with_df.build()

        mf = scf.KRHF(cell, kpts)
        mf.exxdiv = None
        mf.with_df = with_df
        mf.kernel()

        lda = dft.KRKS(cell, kpts, 'lda,vwn')
        lda.exxdiv = None
        lda.with_df = with_df
        lda.kernel()

        gf2 = KRAGF2(mf, **opts)
        gf2.kernel()

        data = [{
            'agf2': {
                'energy': gf2.gf[i].energy,
                'coupling': gf2.gf[i].coupling,
                'chempot': gf2.gf[i].chempot,
                'converged': gf2.converged,
            },
            'hf': {
                'energy': mf.mo_energy[i],
                'coupling': np.eye(mf.mo_energy[i].size),
                'chempot': 0.5 * (mf.mo_energy[i][mf.mo_occ[i] > 0].max() + mf.mo_energy[i][mf.mo_occ[i] == 0].min()),
                'converged': mf.converged,
            },
            'lda': {
                'energy': lda.mo_energy[i],
                'coupling': np.eye(lda.mo_energy[i].size),
                'chempot': 0.5 * (lda.mo_energy[i][lda.mo_occ[i] > 0].max() + lda.mo_energy[i][lda.mo_occ[i] == 0].min()),
                'converged': lda.converged,
            },
        } for i in range(len(kpts))]

        kpt = path_data.push(kpts=kpts, data=data)
        if kpt is False:
            break

    print("Traced %d k-points in %d calculations." % (len(band_kpts), path_data._push_count))

    path_data.close()


if plot:
    fh5 = h5py.File(out, 'r')

    def stack(k1, k2):
        arr = np.array([np.array(fh5['%d/%s/%s' % (i, k1, k2)]) for i in range(len(band_kpts))])
        return arr

    bands = {key: HARTREE2EV * stack(key, 'energy') for key in ['hf', 'lda', 'agf2']}
    bands = {key: bands[key] - np.max(bands[key][bands[key] < HARTREE2EV * stack(key, 'chempot')[:, None]]) for key in bands.keys()}

    if window is None or window == (None, None):
        minpt = min([np.min(val) for val in bands.values()])
        maxpt = max([np.max(val) for val in bands.values()])
    else:
        minpt, maxpt = window
    grid = np.linspace(minpt*1.05, maxpt*1.05, nfreq)
    def spectrum(key, k, freq=None):
        if freq is None:
            freq = grid
        energy = bands[key][k]
        coupling = stack(key, 'coupling')[k]
        d = freq[:, None] - energy[None] + 1.0j * eta_bands
        spec = np.einsum('xk,xk,wk->w', coupling, coupling.conj(), 1./d).imag
        return spec
    def spectrum_ksummed(key):
        energy = bands[key]
        coupling = stack(key, 'coupling')
        spec = 0
        for e, v in zip(energy, coupling):
            d = grid[:, None] - e[None] + 1.0j * eta_dos
            s = np.einsum('xk,xk,wk->w', v, v.conj(), 1./d).imag
            spec -= s
        return spec

    # Setup:
    plt.style.use('seaborn-talk')
    plt.rc('axes', facecolor='whitesmoke')
    plt.rc('figure', facecolor='white')
    plt.rc('lines', markeredgecolor='k', markeredgewidth=1.0)
    plt.figure(figsize=(11, 6))
    axs = [
        plt.subplot2grid((1, 11), (0, 0), colspan=8),
        plt.subplot2grid((1, 11), (0, 8), colspan=3),
    ]

    # HF:
    if plot_hf:
        for i, band in enumerate(bands['hf'].T):
            axs[0].plot(kpath, band, 'C0-', label='HF' if i == 0 else None)
        axs[1].plot(spectrum_ksummed('hf'), grid, 'C0-', label='HF')

    # LDA:
    if plot_lda:
        for i, band in enumerate(bands['lda'].T):
            axs[0].plot(kpath, band, 'C2-', label='LDA' if i == 0 else None)
        axs[1].plot(spectrum_ksummed('lda'), grid, 'C2-', label='LDA')
    
    # AGF2:
    if plot_agf2:
        gr = np.concatenate([grid, [grid[-1]+grid[-1]-grid[-2]]])
        kp = np.concatenate([kpath, [kpath[-1]+kpath[-1]-kpath[-2]]])
        spec = np.zeros((gr.size, kp.size))
        converged = stack('agf2', 'converged')
        for i in range(len(kp)):
            if not converged[i%len(converged)]:
                spec[:, i] = np.nan
            else:
                spec[:, i] -= spectrum('agf2', i%len(converged), freq=gr)
        spec[:, -1] = spec[:, 0]
        for i, freq in enumerate(gr):
            mask = np.isnan(spec[i])
            x = lambda z: z.nonzero()[0]
            spec[i, mask] = np.interp(x(mask), x(~mask), spec[i, ~mask])
        cm = plt.cm.get_cmap(cmap)
        cm._init()
        cm._lut[:, -1] = np.linspace(0, 1, 255+4)
        axs[0].pcolormesh(
                kp-0.5*(kp[1]-kp[0]),
                gr-0.5*(gr[1]-gr[0]),
                spec,
                cmap=cm,
                edgecolors='',
                zorder=-1,
                label='AGF2',
                antialiased=True,
        )
        axs[1].plot(spectrum_ksummed('agf2'), grid, 'C1-', label='AGF2')

    # Ticks:
    axs[0].set_xticks(sp_points)
    axs[0].set_xticklabels(labels)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    for point in sp_points:
        axs[0].plot([point]*2, window, 'k-', lw=1)

    # Limits:
    axs[0].set_xlim(min(sp_points), max(sp_points))
    axs[0].set_ylim(window)
    axs[1].set_ylim(window)

    # Axes:
    axs[0].set_ylabel(r'$E - E_\mathrm{fermi}$ (eV)')
    axs[0].set_xlabel('Point in Brillouin zone')
    axs[1].set_xlabel('Density of states')

    # Formatting:
    axs[1].legend()
    plt.tight_layout()

    fh5.close()
    plt.show()
