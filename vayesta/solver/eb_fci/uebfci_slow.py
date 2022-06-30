"""Module for slow, exact diagonalisation-based coupled electron-boson FCI code.
Based on the fci_slow.py code within pyscf.
"""

import numpy
import numpy as np
from pyscf import lib
from pyscf import ao2mo
from pyscf.fci import cistring
from pyscf.fci import rdm
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.fci import direct_uhf


def contract_all(h1e, g2e, hep, hpp, ci0, norbs, nelec, nbosons, max_occ,
                 ecore=0.0, adj_zero_pho=False):
    # ci1  = contract_1e(h1e, ci0, norbs, nelec, nbosons, max_occ)
    contrib1 = contract_2e(g2e, ci0, norbs, nelec, nbosons, max_occ)
    incbosons = (nbosons > 0 and max_occ > 0)

    if incbosons:
        contrib2 = contract_ep(hep, ci0, norbs, nelec, nbosons, max_occ,
                               adj_zero_pho=adj_zero_pho)
        contrib3 = contract_pp(hpp, ci0, norbs, nelec, nbosons, max_occ)

        return contrib1 + contrib2 + contrib3
    else:
        return contrib1


def make_shape(norbs, nelec, nbosons, max_occ):
    """Construct the shape of a single FCI vector in the coupled electron-boson space.
    """
    neleca, nelecb = _unpack_nelec(nelec)
    na = cistring.num_strings(norbs, neleca)
    nb = cistring.num_strings(norbs, nelecb)
    # print(na,nb,max_occ,nbosons)
    return (na, nb) + (max_occ + 1,) * nbosons


# Contract 1-electron integrals with fcivec.
def contract_1e(h1e, fcivec, norb, nelec, nbosons, max_occ):
    raise NotImplementedError("1 electron contraction is currently"
                              "bugged for coupled electron-boson systems."
                              "This should instead be folded into a two-body operator.")
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec // 2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)

    cishape = make_shape(norb, nelec, nbosons, max_occ)

    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros(cishape, dtype=fcivec.dtype)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            fcinew[str1] += sign * ci0[str0] * h1e[a, i]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            fcinew[:, str1] += sign * ci0[:, str0] * h1e[a, i]
    return fcinew.reshape(fcivec.shape)


# Contract 2-electron integrals with fcivec.
def contract_2e(eri, fcivec, norb, nelec, nbosons, max_occ):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec // 2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)

    cishape = make_shape(norb, nelec, nbosons, max_occ)

    ci0 = fcivec.reshape(cishape)
    t1a = numpy.zeros((norb, norb) + cishape, dtype=fcivec.dtype)
    t1b = numpy.zeros_like(t1a)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t1a[a, i, str1] += sign * ci0[str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            t1b[a, i, :, str1] += sign * ci0[:, str0]

    # t1 = lib.einsum('bjai,aiAB...->bjAB...', eri.reshape([norb]*4), t1)
    t1a_ = numpy.tensordot(eri[0].reshape([norb] * 4), t1a, 2) + numpy.tensordot(eri[1].reshape([norb] * 4), t1b, 2)
    t1b_ = numpy.tensordot(eri[2].reshape([norb] * 4), t1b, 2) + numpy.tensordot(eri[1].reshape([norb] * 4), t1a,
                                                                                [[0,1], [0,1]])
    t1a, t1b = t1a_, t1b_

    fcinew = numpy.zeros_like(ci0)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            fcinew[str1] += sign * t1a[a, i, str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            fcinew[:, str1] += sign * t1b[a, i, :, str0]
    return fcinew.reshape(fcivec.shape)


# Contract electron-phonon portion of the Hamiltonian.
def contract_ep(heb, fcivec, norb, nelec, nbosons, max_occ, adj_zero_pho=False):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec // 2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)

    cishape = make_shape(norb, nelec, nbosons, max_occ)

    ci0 = fcivec.reshape(cishape)
    t1a = numpy.zeros((norb, norb) + cishape, dtype=fcivec.dtype)
    t1b = numpy.zeros((norb, norb) + cishape, dtype=fcivec.dtype)

    if adj_zero_pho:
        zfac = float(neleca + nelecb) / norb
        # print("Zfac=",zfac)
        adj_val = zfac * ci0
        for i in range(norb):
            t1a[i, i] -= adj_val
            t1b[i, i] -= adj_val

    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t1a[a, i, str1] += sign * ci0[str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            t1b[a, i, :, str1] += sign * ci0[:, str0]
    # Now have contribution to a particular state via given excitation
    # channel; just need to apply bosonic (de)excitations.
    # Note that while the contribution {a^{+} i p} and {i a^{+} p^{+}}
    # are related via the Hermiticity of the Hamiltonian, no other properties
    # are guaranteed, so we cannot write this as p + p^{+}.

    # Contract intermediate with the electron-boson coupling.
    # If we need to remove zero phonon mode also need factor of -<N>
    heb_a, heb_b = heb if type(heb) == tuple else (heb, heb)
    # First, bosonic excitations.
    tex = numpy.tensordot(heb_a, t1a, 2) + numpy.tensordot(heb_b, t1b, 2)
    # tex = numpy.einsum("nai,ai...->n...", g, t1)
    # Then bosonic deexcitations.
    tdex = numpy.tensordot(heb_a, t1a, ((1, 2), (1, 0))) + numpy.tensordot(heb_b, t1b, ((1, 2), (1, 0)))
    # tdex = numpy.einsum("nia,ai...->n...", g, t1)

    # print(norb,nelec, nbosons)
    # print(tex.shape,"Ex:",numpy.sum(tex**2))
    # print(tex)
    # print(tdex.shape, "Deex:",numpy.sum(tdex**2))
    # print(tdex)
    # The leading index tells us which bosonic degree of freedom is coupled
    # to in each case.
    fcinew = numpy.zeros_like(ci0)

    bos_cre = numpy.sqrt(numpy.arange(1, max_occ + 1))

    for ibos in range(nbosons):
        for iocc in range(0, max_occ):
            ex_slice = slices_for_cre(ibos, nbosons, iocc)
            norm_slice = slices_for(ibos, nbosons, iocc)
            # NB bos_cre[iocc] = sqrt(iocc+1)
            fcinew[ex_slice] += tex[ibos][norm_slice] * bos_cre[iocc]

        for iocc in range(1, max_occ + 1):
            dex_slice = slices_for_des(ibos, nbosons, iocc)
            norm_slice = slices_for(ibos, nbosons, iocc)
            # NB bos_cre[iocc] = sqrt(iocc+1)
            fcinew[dex_slice] += tdex[ibos][norm_slice] * bos_cre[iocc - 1]

    return fcinew.reshape(fcivec.shape)


# Contract phonon-phonon portion of the Hamiltonian.

def contract_pp(hpp, fcivec, norb, nelec, nbosons, max_occ):
    """Arbitrary phonon-phonon coupling.
    """
    cishape = make_shape(norb, nelec, nbosons, max_occ)
    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros_like(ci0)

    phonon_cre = numpy.sqrt(numpy.arange(1, max_occ + 1))
    # t1 = numpy.zeros((nbosons,)+cishape, dtype=fcivec.dtype)
    # for psite_id in range(nbosons):
    #    for i in range(max_occ):
    #        slices1 = slices_for_cre(psite_id, nbosons, i)
    #        slices0 = slices_for    (psite_id, nbosons, i)
    #        t1[(psite_id,)+slices0] += ci0[slices1] * phonon_cre[i]     # annihilation

    t1 = apply_bos_annihilation(ci0, nbosons, max_occ)

    t1 = lib.dot(hpp, t1.reshape(nbosons, -1)).reshape(t1.shape)

    for psite_id in range(nbosons):
        for i in range(max_occ):
            slices1 = slices_for_cre(psite_id, nbosons, i)
            slices0 = slices_for(psite_id, nbosons, i)
            fcinew[slices1] += t1[(psite_id,) + slices0] * phonon_cre[i]  # creation
    return fcinew.reshape(fcivec.shape)


def apply_bos_annihilation(ci0, nbosons, max_occ):
    phonon_cre = numpy.sqrt(numpy.arange(1, max_occ + 1))
    res = numpy.zeros((nbosons,) + ci0.shape, dtype=ci0.dtype)
    for psite_id in range(nbosons):
        for i in range(max_occ):
            slices1 = slices_for_cre(psite_id, nbosons, i)
            slices0 = slices_for(psite_id, nbosons, i)
            res[(psite_id,) + slices0] += ci0[slices1] * phonon_cre[i]  # annihilation
    return res


def apply_bos_creation(ci0, nbosons, max_occ):
    phonon_cre = numpy.sqrt(numpy.arange(1, max_occ + 1))
    res = numpy.zeros((nbosons,) + ci0.shape, dtype=ci0.dtype)
    for psite_id in range(nbosons):
        for i in range(max_occ):
            slices1 = slices_for_cre(psite_id, nbosons, i)
            slices0 = slices_for(psite_id, nbosons, i)
            res[(psite_id,) + slices1] += ci0[slices0] * phonon_cre[i]  # creation
    return res


def contract_pp_for_future(hpp, fcivec, norb, nelec, nbosons, max_occ):
    """Our bosons are decoupled; only have diagonal couplings,
    ie. to the boson number.
    """
    cishape = make_shape(norb, nelec, nbosons, max_occ)
    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros_like(ci0)

    for ibos in range(nbosons):
        for iocc in range(max_occ + 1):
            slice1 = slices_for(ibos, nbosons, iocc)
            # This may need a sign change?
            # Two factors sqrt(iocc) from annihilation then creation.
            fcinew[slice1] += ci0[slice1] * iocc * hpp[ibos]
    return fcinew.reshape(fcivec.shape)


def slices_for(b_id, nbos, occ):
    slices = [slice(None, None, None)] * (2 + nbos)  # +2 for electron indices
    slices[2 + b_id] = occ
    return tuple(slices)


def slices_for_cre(b_id, nbos, occ):
    return slices_for(b_id, nbos, occ + 1)


def slices_for_des(b_id, nbos, occ):
    return slices_for(b_id, nbos, occ - 1)


def slices_for_occ_reduction(nbos, new_max_occ):
    slices = [slice(None, None, None)] * 2
    slices += [slice(0, new_max_occ + 1)] * nbos
    return tuple(slices)


def make_hdiag(h1e, g2e, hep, hpp, norb, nelec, nbosons, max_occ):
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    occslista = [tab[:neleca, 0] for tab in link_indexa]
    occslistb = [tab[:nelecb, 0] for tab in link_indexb]

    nelec_tot = neleca + nelecb

    # electron part
    cishape = make_shape(norb, nelec, nbosons, max_occ)
    hdiag = numpy.zeros(cishape)
    g2e_aa = ao2mo.restore(1, g2e[0], norb)
    g2e_ab = ao2mo.restore(1, g2e[1], norb)
    g2e_bb = ao2mo.restore(1, g2e[2], norb)

    diagj_aa = numpy.einsum('iijj->ij', g2e_aa)
    diagj_ab = numpy.einsum('iijj->ij', g2e_ab)
    diagj_bb = numpy.einsum('iijj->ij', g2e_bb)

    diagk_aa = numpy.einsum('ijji->ij', g2e_aa)
    diagk_bb = numpy.einsum('ijji->ij', g2e_bb)

    for ia, aocc in enumerate(occslista):
        for ib, bocc in enumerate(occslistb):
            e1 = h1e[0][aocc, aocc].sum() + h1e[1][bocc, bocc].sum()
            e2 = diagj_aa[aocc][:, aocc].sum() + diagj_ab[aocc][:, bocc].sum() \
                 + diagj_ab.T[bocc][:, aocc].sum() + diagj_bb[bocc][:, bocc].sum() \
                 - diagk_aa[aocc][:, aocc].sum() - diagk_bb[bocc][:, bocc].sum()
            hdiag[ia, ib] += e1 + e2 * .5

    # No electron-phonon part?

    # phonon part
    if len(hpp.shape) == 2:
        hpp = hpp.diagonal()
    for b_id in range(nbosons):
        for i in range(max_occ + 1):
            slices0 = slices_for(b_id, nbosons, i)
            hdiag[slices0] += hpp[b_id] * i

    return hdiag.ravel()


def kernel(h1e, g2e, hep, hpp, norb, nelec, nbosons, max_occ,
           tol=1e-12, max_cycle=100, verbose=0, ecore=0,
           returnhop=False, adj_zero_pho=False,
           **kwargs):
    h2e = tuple([ao2mo.restore(1, x, norb) for x in direct_uhf.absorb_h1e(h1e, g2e, norb, nelec, .5)])
    cishape = make_shape(norb, nelec, nbosons, max_occ)
    ci0 = numpy.zeros(cishape)
    # Add noise for initial guess, remove it if problematic
    ci0 += numpy.random.random(ci0.shape) * 1e-10
    ci0.__setitem__((0, 0) + (0,) * nbosons, 1)

    def hop(c):
        hc = contract_all(h1e, h2e, hep, hpp, c, norb,
                          nelec, nbosons, max_occ, adj_zero_pho=adj_zero_pho)
        return hc.reshape(-1)

    hdiag = make_hdiag(h1e, g2e, hep, hpp, norb, nelec, nbosons, max_occ)
    precond = lambda x, e, *args: x / (hdiag - e + 1e-4)
    if returnhop:
        return hop, ci0, hdiag

    e, c = lib.davidson(hop, ci0.reshape(-1), precond,
                        tol=tol, max_cycle=max_cycle, verbose=verbose,
                        **kwargs)
    return e + ecore, c.reshape(cishape)


def kernel_multiroot(h1e, g2e, hep, hpp, norb, nelec, nbosons, max_occ,
                     tol=1e-12, max_cycle=100, verbose=0, ecore=0, nroots=2,
                     returnhop=False, adj_zero_pho=False,
                     **kwargs):
    if nroots == 1:
        return kernel(h1e, g2e, hep, hpp, norb, nelec, nbosons, max_occ,
                      tol, max_cycle, verbose, ecore, returnhop, adj_zero_pho,
                      **kwargs)
    h2e = direct_uhf.absorb_h1e(h1e, g2e, norb, nelec, .5)

    cishape = make_shape(norb, nelec, nbosons, max_occ)
    ci0 = numpy.zeros(cishape)
    ci0.__setitem__((0, 0) + (0,) * nbosons, 1)
    # Add noise for initial guess, remove it if problematic
    ci0[0, :] += numpy.random.random(ci0[0, :].shape) * 1e-6
    ci0[:, 0] += numpy.random.random(ci0[:, 0].shape) * 1e-6

    def hop(c):
        hc = contract_all(h1e, h2e, hep, hpp, c, norb,
                          nelec, nbosons, max_occ, adj_zero_pho=adj_zero_pho)
        return hc.reshape(-1)

    hdiag = make_hdiag(h1e, g2e, hep, hpp, norb, nelec, nbosons, max_occ)
    precond = lambda x, e, *args: x / (hdiag - e + 1e-4)
    if returnhop:
        return hop, ci0, hdiag

    es, cs = lib.davidson(hop, ci0.reshape(-1), precond,
                          tol=tol, max_cycle=max_cycle, verbose=verbose, nroots=nroots, max_space=20,
                          **kwargs)
    return es, cs

# dm_pq = <|p^+ q|>
def make_rdm1(fcivec, norb, nelec):
    '''1-electron density matrix dm_pq = <|p^+ q|>'''
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)

    rdm1a = numpy.zeros((norb, norb))
    rdm1b = np.zeros_like(rdm1a)
    ci0 = fcivec.reshape(na, -1)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            rdm1a[a, i] += sign * numpy.dot(ci0[str1], ci0[str0])

    ci0 = fcivec.reshape(na, nb, -1)
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            rdm1b[a, i] += sign * numpy.einsum('ax,ax->', ci0[:, str1], ci0[:, str0])
    return rdm1a, rdm1b


def make_rdm12(fcivec, norb, nelec):
    '''1-electron and 2-electron density matrices
    dm_qp = <|q^+ p|>
    dm_{pqrs} = <|p^+ r^+ s q|>
    '''
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)

    ci0 = fcivec.reshape(na, nb, -1)
    rdm1 = numpy.zeros((norb, norb))
    rdm2 = numpy.zeros((norb, norb, norb, norb))

    for str0 in range(na):
        t1 = numpy.zeros((norb, norb, nb) + ci0.shape[2:])
        for a, i, str1, sign in link_indexa[str0]:
            t1[i, a, :] += sign * ci0[str1, :]

        for k, tab in enumerate(link_indexb):
            for a, i, str1, sign in tab:
                t1[i, a, k] += sign * ci0[str0, str1]

        rdm1 += numpy.einsum('mp,ijmp->ij', ci0[str0], t1)
        # i^+ j|0> => <0|j^+ i, so swap i and j
        #:rdm2 += numpy.einsum('ijmp,klmp->jikl', t1, t1)
        tmp = lib.dot(t1.reshape(norb ** 2, -1), t1.reshape(norb ** 2, -1).T)
        rdm2 += tmp.reshape((norb,) * 4).transpose(1, 0, 2, 3)
    rdm1, rdm2 = rdm.reorder_rdm(rdm1, rdm2, True)
    return rdm1, rdm2


def make_rdm12s(fcivec, norb, nelec, reorder=True):
    '''1-electron and 2-electron spin-resolved density matrices
    dm_qp = <|q^+ p|>
    dm_{pqrs} = <|p^+ r^+ s q|>
    '''
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)

    ci0 = fcivec.reshape(na, nb, -1)
    nbos_dof = ci0.shape[2]

    dm1a = numpy.zeros((norb, norb))
    dm1b = numpy.zeros_like(dm1a)

    dm2aa = numpy.zeros((norb,norb,norb,norb))
    dm2ab = numpy.zeros_like(dm2aa)
    dm2bb = numpy.zeros_like(dm2aa)

    for str0 in range(na):
        # Alpha excitation.
        t1a = numpy.zeros((norb, norb, nb, nbos_dof))
        t1b = numpy.zeros_like(t1a)

        for a, i, str1, sign in link_indexa[str0]:
            t1a[i, a, :] += sign * ci0[str1, :]
        # Beta excitation.
        for k, tab in enumerate(link_indexb):
            for a, i, str1, sign in tab:
                t1b[i, a, k] += sign * ci0[str0, str1]
        # Generate our spin-resolved 1rdm contribution. do fast as over the full FCI space.
        dm1a += numpy.tensordot(t1a, ci0[str0].reshape((nb, nbos_dof)), 2)
        dm1b += numpy.tensordot(t1b, ci0[str0].reshape((nb, nbos_dof)), 2)

        # rdm1 += numpy.einsum('mp,ijmp->ij', ci0[str0], t1)
        # t1[i,a] = a^+ i |0>
        # i^+ j|0> => <0|j^+ i, so swap i and j
        #:rdm2 += numpy.einsum('ijmp,klmp->jikl', t1, t1)
        # Calc <0|i^+ a b^+ j |0>
        t1a = t1a.reshape((norb ** 2, -1))
        t1b = t1b.reshape((norb ** 2, -1))

        dm2aa += lib.dot(t1a, t1a.T).reshape((norb,) * 4).transpose(1, 0, 2, 3)
        dm2ab += lib.dot(t1a, t1b.T).reshape((norb,) * 4).transpose(1, 0, 2, 3)
        dm2bb += lib.dot(t1b, t1b.T).reshape((norb,) * 4).transpose(1, 0, 2, 3)

    if reorder:
        dm1a, dm2aa = rdm.reorder_rdm(dm1a, dm2aa, inplace=True)
        dm1b, dm2bb = rdm.reorder_rdm(dm1b, dm2bb, inplace=True)
    return (dm1a, dm1b), (dm2aa, dm2ab, dm2bb)


def make_eb_rdm(fcivec, norb, nelec, nbosons, max_occ):
    """
    We calculate the value <0|b^+ p^+ q|0> and return this in value P[p,q,b]
    :param fcivec:
    :param norb:
    :param nelec:
    :param max_occ:
    :return:
    """
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec // 2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)

    cishape = make_shape(norb, nelec, nbosons, max_occ)
    nspinorb = 2 * norb

    # Just so we get a sensible result in pure fermionic case.
    if nbosons == 0:
        return numpy.zeros((nspinorb, nspinorb, 0))

    ci0 = fcivec.reshape(cishape)
    t1a = numpy.zeros((norb, norb) + cishape, dtype=fcivec.dtype)
    t1b = numpy.zeros_like(t1a)

    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in link_indexa[str0]:
            t1a[a, i, str1] += sign * ci0[str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in link_indexb[str0]:
            t1b[a, i, :, str1] += sign * ci0[:, str0]
    bos_cre = numpy.sqrt(numpy.arange(1, max_occ + 1))
    tempa = numpy.zeros((norb, norb, nbosons) + ci0.shape)
    tempb = np.zeros_like(tempa)
    for ibos in range(nbosons):
        # We could tidy this up with nicer slicing and einsum, but it works for now.
        for iocc in range(0, max_occ):
            ex_slice = (slice(None, None, None),) * 2 + (ibos,) + slices_for_cre(ibos, nbosons, iocc)
            norm_slice = (slice(None, None, None),) * 2 + slices_for(ibos, nbosons, iocc)
            tempa[ex_slice] += t1a[norm_slice] * bos_cre[iocc]
            tempb[ex_slice] += t1b[norm_slice] * bos_cre[iocc]
    rdm_fba = numpy.dot(tempa.reshape((norb ** 2 * nbosons, -1)), ci0.reshape(-1)).reshape((norb, norb, nbosons))
    rdm_fbb = numpy.dot(tempb.reshape((norb ** 2 * nbosons, -1)), ci0.reshape(-1)).reshape((norb, norb, nbosons))
    return rdm_fba, rdm_fbb


def calc_dd_resp_mom(ci0, e0, max_mom, norb, nel, nbos, h1e, eri, hbb, heb, max_boson_occ, rdm1,
                     trace=False,
                     coeffs=None, **kwargs):
    """
    Calculate up to the mth moment of the dd response, dealing with all spin components separately. To replace
    preceding function.
    :param m: maximum moment order of response to return.
    :param hfbas: whether to return the moment in the HF basis. Otherwise returns in the basis in the underlying
            orthogonal basis hfbas is specified in (defaults to False).
    :return:
    """
    # Note that we must stay in the same spin sector in this approach; if we want to get the nonzero components of the
    # spin-density response (rather than charge-density) we'll need to use commutation relations to relate to the
    # equivalent charge-density response.
    hop = kernel(h1e, eri, heb, hbb, norb, nel, nbos, max_boson_occ, returnhop=True, **kwargs)[0]
    if isinstance(nel, (int, numpy.integer)):
        nelecb = nel // 2
        neleca = nel - nelecb
    else:
        neleca, nelecb = nel
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    cishape = make_shape(norb, nel, nbos, max_boson_occ)

    t1a = numpy.zeros((norb, norb) + cishape, dtype=numpy.dtype(numpy.float64))
    t1b = numpy.zeros_like(t1a)

    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t1a[a, i, str1] += sign * ci0[str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            t1b[a, i, :, str1] += sign * ci0[:, str0]
    # If we want not in HF basis we can perform transformation at this stage.
    t1a = t1a.reshape((norb, norb, -1))
    t1b = t1b.reshape((norb, norb, -1))
    na = nb = norb
    if not (coeffs is None):
        if type(coeffs) == tuple:
            coeffsa, coeffsb = coeffs
        else:
            coeffsa = coeffsb = coeffs
        na, nb = coeffsa.shape[1], coeffsb.shape[1]
        t1a = numpy.einsum("ia...,ip,aq->pq...", t1a, coeffsa, coeffsa)
        t1b = numpy.einsum("ia...,ip,aq->pq...", t1b, coeffsb, coeffsb)
    if trace:
        t1a = numpy.einsum("ii...->i...", t1a)
        t1b = numpy.einsum("ii...->i...", t1b)
    # From this we'll obtain our moments through dot products, thanks to the Hermiticity of our expression.
    max_intermed = numpy.ceil(max_mom / 2).astype(int)
    aintermeds = {0: t1a}
    bintermeds = {0: t1b}
    for iinter in range(max_intermed):
        aintermeds[iinter + 1] = numpy.zeros_like(t1a)
        bintermeds[iinter + 1] = numpy.zeros_like(t1b)
        for i in range(na):
            if trace:
                aintermeds[iinter + 1][i] = hop(aintermeds[iinter][i]).reshape(-1) - \
                                            e0 * aintermeds[iinter][i]
            else:
                for a in range(na):
                    aintermeds[iinter + 1][a, i] = hop(aintermeds[iinter][a, i]).reshape(-1) - \
                                                   e0 * aintermeds[iinter][a, i]
        for i in range(nb):
            if trace:
                bintermeds[iinter + 1][i] = hop(bintermeds[iinter][i]).reshape(-1) - \
                                            e0 * bintermeds[iinter][i]
            else:
                for a in range(nb):
                    bintermeds[iinter + 1][a, i] = hop(bintermeds[iinter][a, i]).reshape(-1) - \
                                                   e0 * bintermeds[iinter][a, i]

    # Need to adjust zeroth moment to remove ground state contributions; in all higher moments this is achieved by
    # deducting the reference energy.
    # Now take appropriate dot products to get moments.
    moments = {}
    for imom in range(max_mom + 1):
        r_ind = min(imom, max_intermed)
        l_ind = imom - r_ind

        aintermed_r = aintermeds[r_ind]
        bintermed_r = bintermeds[r_ind]
        aintermed_l = aintermeds[l_ind]
        bintermed_l = bintermeds[l_ind]

        if trace:
            moments[imom] = (
                numpy.dot(aintermed_l, aintermed_r.T),
                numpy.dot(aintermed_l, bintermed_r.T),
                numpy.dot(bintermed_l, bintermed_r.T)
            )
        else:
            moments[imom] = (
                numpy.tensordot(aintermed_l, aintermed_r, (2, 2)),
                numpy.tensordot(aintermed_l, bintermed_r, (2, 2)),
                numpy.tensordot(bintermed_l, bintermed_r, (2, 2))
            )
    # Need to add additional adjustment for zeroth moment, as there is a nonzero ground state
    # contribution in this case (the current value is in fact the double occupancy <0|n_{pq} n_{sr}|0>).
    if type(rdm1) == tuple:
        rdma, rdmb = rdm1
    else:
        rdma = rdmb = rdm1 / 2
    if not (coeffs is None):
        rdma = coeffsa.T.dot(rdma).dot(coeffsa)
        rdmb = coeffsb.T.dot(rdmb).dot(coeffsb)

    moments[0] = list(moments[0])
    if trace:
        moments[0][0] = moments[0][0] - numpy.einsum("pp,qq->pq", rdma, rdma)
        moments[0][1] = moments[0][1] - numpy.einsum("pp,qq->pq", rdma, rdmb)
        moments[0][2] = moments[0][2] - numpy.einsum("pp,qq->pq", rdmb, rdmb)
    else:
        moments[0][0] = moments[0][0] - numpy.einsum("pq,rs->pqrs", rdma, rdma)
        moments[0][1] = moments[0][1] - numpy.einsum("pq,rs->pqrs", rdma, rdmb)
        moments[0][2] = moments[0][2] - numpy.einsum("pq,rs->pqrs", rdmb, rdmb)
    moments[0] = tuple(moments[0])
    return moments


def run(nelec, h1e, eri, hpp, hep, max_occ, returnhop=False, nroots=1, **kwargs):
    """run a calculation using a pyscf mf object.
    """
    norb = h1e.shape[1]
    nbosons = hpp.shape[0]
    if returnhop:
        hop0 = kernel(h1e, eri, hep, hpp, norb, nelec, nbosons, max_occ,
                      returnhop=True, **kwargs)

        # hop1 = fci_slow.kernel(h1e, eri, norb, nelec)#, returnhop=True)
        return hop0  # , hop1
    if nroots > 1:
        es, cs = kernel_multiroot(h1e, eri, hep, hpp, norb, nelec, nbosons, max_occ, nroots=nroots,
                                  **kwargs)
        return es, cs

    else:
        res0 = kernel(h1e, eri, hep, hpp, norb, nelec, nbosons, max_occ,
                      **kwargs)

        return res0


def run_hub_test(returnhop=False, **kwargs):
    return run_ep_hubbard(t=1.0, u=1.5, g=0.5, pp=0.1, nsite=2, nelec=2, nphonon=3,
                          returnhop=returnhop)


def run_ep_hubbard(t, u, g, pp, nsite, nelec, nphonon, returnhop=False, **kwargs):
    """Run a calculation using a hubbard model coupled to some phonon modes.
    """
    idx = numpy.arange(nsite - 1)
    # 1 electron interactions.
    h1e = numpy.zeros((nsite, nsite))
    h1e[idx + 1, idx] = h1e[idx, idx + 1] = -t
    # Phonon coupling.
    hpp = numpy.eye(nsite) * (0.3 + pp)
    hpp[idx + 1, idx] = hpp[idx, idx + 1] = pp
    # 2 electron interactions.
    eri = numpy.zeros((nsite, nsite, nsite, nsite))
    for i in range(nsite):
        eri[i, i, i, i] = u
    # Electron-phonon coupling.
    hep = numpy.zeros((nsite, nsite, nsite))  # (phonon, orb, orb)
    # Only have onsite coupling; so only couple .
    for i in range(nsite):
        hep[i, i, i] = g
    res0 = kernel(h1e, eri, hep, hpp, nsite, nelec, nsite, nphonon,
                  adj_zero_pho=False, returnhop=returnhop, **kwargs)
    return res0
