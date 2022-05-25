import numpy as np

from vayesta.core.util import *

def plot_histogram(values, bins=None, maxbarlength=50):
    if bins is None:
        bins = np.hstack([np.inf, np.logspace(-3, -12, 10), -np.inf])
    bins = bins[::-1]
    hist = np.histogram(values, bins)[0]
    bins, hist = bins[::-1], hist[::-1]
    cumsum = 0
    lines = ["  {:^13s}  {:^4s}   {:^51s}".format("Interval", "Sum", "Histogram").rstrip()]
    for i, hval in enumerate(hist):
        cumsum += hval
        barlength = int(maxbarlength * hval/hist.max())
        if hval == 0:
            bar = ""
        else:
            barlength = max(barlength, 1)
            bar = ((barlength-1) * "|") + "]" + ("  (%d)" % hval)
        #log.info("  %5.0e - %5.0e  %4d   |%s", bins[i+1], bins[i], cumsum, bar)
        lines.append("  %5.0e - %5.0e  %4d   |%s" % (bins[i+1], bins[i], cumsum, bar))
    return lines

def horizontal_histogram(values, bins=None, binwidth=6, height=10):
    if bins is None:
        bins = np.hstack([np.inf, np.logspace(-3, -12, 10), -np.inf])[::-1]
    hist = np.histogram(values, bins)[0]
    bins, hist = bins[::-1], hist[::-1]
    cumsum = 0
    width = (binwidth+1)*len(hist) + 1
    plot = np.zeros((height, width), dtype=str)
    plot[:] = ' '
    for i, hval in enumerate(hist):
        wslice = np.s_[i*(binwidth+1)+1 : (i+1)*(binwidth+1)]
        cumsum += hval
        barheight = int(height * hval/hist.max())
        if hval == 0:
            bar = ''
        else:
            barheight = max(barheight, 1)
            plot[-barheight:,wslice] = '|'
    lines = [''.join(plot[r,:].tolist()) for r in range(height)]
    lines.append(len(hist) * ('+' + (binwidth) * '-'))
    #lines.append(len(hist) * ('[' + ((binwidth-2) * '-') + ']'))
    # Print number
    #lines += [''.join(['%*d' % (binwidth, h) for h in hist])]
    lines += [''.join(['{:^{w}}'.format(h, w=binwidth+1) for h in hist])]

    return lines

def transform_mp2_eris(eris, c_occ, c_vir, ovlp):  # pragma: no cover
    """Transform eris of kind (ov|ov) (occupied-virtual-occupied-virtual)

    OBSOLETE: replaced by transform_eris
    """
    assert (eris is not None)
    assert (eris.ovov is not None)

    c_occ0, c_vir0 = np.hsplit(eris.mo_coeff, [eris.nocc])
    nocc0, nvir0 = c_occ0.shape[-1], c_vir0.shape[-1]
    nocc, nvir = c_occ.shape[-1], c_vir.shape[-1]

    transform_occ = (nocc != nocc0 or not np.allclose(c_occ, c_occ0))
    if transform_occ:
        r_occ = np.linalg.multi_dot((c_occ.T, ovlp, c_occ0))
    else:
        r_occ = np.eye(nocc)
    transform_vir = (nvir != nvir0 or not np.allclose(c_vir, c_vir0))
    if transform_vir:
        r_vir = np.linalg.multi_dot((c_vir.T, ovlp, c_vir0))
    else:
        r_vir = np.eye(nvir)
    r_all = np.block([
        [r_occ, np.zeros((nocc, nvir0))],
        [np.zeros((nvir, nocc0)), r_vir]])

    # eris.ovov may be hfd5 dataset on disk -> allocate in memory with [:]
    govov = eris.ovov[:].reshape(nocc0, nvir0, nocc0, nvir0)
    if transform_occ and transform_vir:
        govov = einsum("iajb,xi,ya,zj,wb->xyzw", govov, r_occ, r_vir, r_occ, r_vir)
    elif transform_occ:
        govov = einsum("iajb,xi,zj->xazb", govov, r_occ, r_occ)
    elif transform_vir:
        govov = einsum("iajb,ya,wb->iyjw", govov, r_vir, r_vir)
    eris.ovov = govov.reshape((nocc*nvir, nocc*nvir))
    eris.mo_coeff = np.hstack((c_occ, c_vir))
    eris.fock = np.linalg.multi_dot((r_all, eris.fock, r_all.T))
    eris.mo_energy = np.diag(eris.fock)
    return eris

if __name__ == '__main__':
    vals = sorted(np.random.rand(30))
    print(vals)
    for line in plot_histogram(vals):
        print(line)

    print('')

    bins = np.linspace(0, 1, 12)
    print(bins)
    for line in horizontal_histogram(vals, bins):
        print(line)
