import numpy as np

from vayesta.core.util import einsum


def make_histogram(
    values, bins, labels=None, binwidth=5, height=6, fill=":", show_number=False, invertx=True, rstrip=True
):
    hist = np.histogram(values, bins)[0]
    if invertx:
        bins, hist = bins[::-1], hist[::-1]
    hmax = hist.max()

    width = binwidth * len(hist)
    plot = np.zeros((height + show_number, width), dtype=str)
    plot[:] = " "
    if hmax > 0:
        for i, hval in enumerate(hist):
            colstart = i * binwidth
            colend = (i + 1) * binwidth
            barheight = int(np.rint(height * hval / hmax))
            if barheight == 0:
                continue
            # Top
            plot[-barheight, colstart + 1 : colend - 1] = "_"
            if show_number:
                number = " {:^{w}s}".format("(%d)" % hval, w=binwidth - 1)
                for idx, i in enumerate(range(colstart, colend)):
                    plot[-barheight - 1, i] = number[idx]

            if barheight == 1:
                continue
            # Fill
            if fill:
                plot[-barheight + 1 :, colstart + 1 : colend] = fill
            # Left/right border
            plot[-barheight + 1 :, colstart] = "|"
            plot[-barheight + 1 :, colend - 1] = "|"

    lines = ["".join(plot[r, :].tolist()) for r in range(height)]
    # Baseline
    lines.append("+" + ((width - 2) * "-") + "+")

    if labels:
        if isinstance(labels, str):
            lines += [labels]
        else:
            lines += ["".join(["{:^{w}}".format(l, w=binwidth) for l in labels])]

    if rstrip:
        lines = [line.rstrip() for line in lines]
    txt = "\n".join(lines)
    return txt


def make_horizontal_histogram(values, bins=None, maxbarlength=50, invertx=True):
    if bins is None:
        bins = np.hstack([-np.inf, np.logspace(-3, -12, 10)[::-1], np.inf])
    hist = np.histogram(values, bins)[0]
    if invertx:
        bins, hist = bins[::-1], hist[::-1]
    cumsum = 0
    lines = ["  {:^13s}  {:^4s}   {:^51s}".format("Interval", "Sum", "Histogram").rstrip()]
    for i, hval in enumerate(hist):
        cumsum += hval
        barlength = int(maxbarlength * hval / hist.max())
        if hval == 0:
            bar = ""
        else:
            barlength = max(barlength, 1)
            bar = ((barlength - 1) * "|") + "]" + ("  (%d)" % hval)
        # log.info("  %5.0e - %5.0e  %4d   |%s", bins[i+1], bins[i], cumsum, bar)
        lines.append("  %5.0e - %5.0e  %4d   |%s" % (bins[i + 1], bins[i], cumsum, bar))
    txt = "\n".join(lines)
    return txt


def transform_mp2_eris(eris, c_occ, c_vir, ovlp):  # pragma: no cover
    """Transform eris of kind (ov|ov) (occupied-virtual-occupied-virtual)

    OBSOLETE: replaced by transform_eris
    """
    assert eris is not None
    assert eris.ovov is not None

    c_occ0, c_vir0 = np.hsplit(eris.mo_coeff, [eris.nocc])
    nocc0, nvir0 = c_occ0.shape[-1], c_vir0.shape[-1]
    nocc, nvir = c_occ.shape[-1], c_vir.shape[-1]

    transform_occ = nocc != nocc0 or not np.allclose(c_occ, c_occ0)
    if transform_occ:
        r_occ = np.linalg.multi_dot((c_occ.T, ovlp, c_occ0))
    else:
        r_occ = np.eye(nocc)
    transform_vir = nvir != nvir0 or not np.allclose(c_vir, c_vir0)
    if transform_vir:
        r_vir = np.linalg.multi_dot((c_vir.T, ovlp, c_vir0))
    else:
        r_vir = np.eye(nvir)
    r_all = np.block([[r_occ, np.zeros((nocc, nvir0))], [np.zeros((nvir, nocc0)), r_vir]])

    # eris.ovov may be hfd5 dataset on disk -> allocate in memory with [:]
    govov = eris.ovov[:].reshape(nocc0, nvir0, nocc0, nvir0)
    if transform_occ and transform_vir:
        govov = einsum("iajb,xi,ya,zj,wb->xyzw", govov, r_occ, r_vir, r_occ, r_vir)
    elif transform_occ:
        govov = einsum("iajb,xi,zj->xazb", govov, r_occ, r_occ)
    elif transform_vir:
        govov = einsum("iajb,ya,wb->iyjw", govov, r_vir, r_vir)
    eris.ovov = govov.reshape((nocc * nvir, nocc * nvir))
    eris.mo_coeff = np.hstack((c_occ, c_vir))
    eris.fock = np.linalg.multi_dot((r_all, eris.fock, r_all.T))
    eris.mo_energy = np.diag(eris.fock)
    return eris


if __name__ == "__main__":
    vals = sorted(np.random.rand(30))
    print(make_vertical_histogram(vals))
    print("")
    bins = np.linspace(0, 1, 12)
    # for line in horizontal_histogram(vals, bins):
    labels = "    " + "".join("{:{w}}".format("E-%d" % d, w=6) for d in range(3, 13))
    print(make_histogram(vals, bins, labels=labels))
