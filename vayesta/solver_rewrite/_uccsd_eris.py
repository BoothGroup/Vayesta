#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Timothy Berkelbach <tim.berkelbach@gmail.com>
#         Qiming Sun <osirpt.sun@gmail.com>
#

'''
UCCSD with spatial integrals
'''

from pyscf import lib, ao2mo
from pyscf.cc.uccsd import _ChemistsERIs
import numpy as np


def uao2mo(self, mo_coeff=None):
    nmoa, nmob = self.get_nmo()
    nao = self.mo_coeff[0].shape[0]
    nmo_pair = nmoa * (nmoa + 1) // 2
    nao_pair = nao * (nao + 1) // 2
    mem_incore = (max(nao_pair ** 2, nmoa ** 4) + nmo_pair ** 2) * 8 / 1e6
    mem_now = lib.current_memory()[0]
    if (self._scf._eri is not None):
        assert (np.ndim(self._scf._eri[0]) == 4)
        if (mem_incore + mem_now < self.max_memory or self.incore_complete):
            ao2mofn = (
                lambda mo_coeff: ao2mo.restore(1, ao2mo.full(self._scf._eri[0], mo_coeff), mo_coeff.shape[1]),
                lambda mo_coeff: ao2mo.general(self._scf._eri[1], mo_coeff),
                lambda mo_coeff: ao2mo.restore(1, ao2mo.full(self._scf._eri[2], mo_coeff), mo_coeff.shape[1]),
            )

            return _make_eris_incore(self, mo_coeff, ao2mofn=ao2mofn)
        else:
            raise RuntimeError("Dense Cluster ERIs predicted to exceed available memory.")
    else:
        raise NotImplementedError("Current modifications to only support spin-dependent eris in UCCSD without "
                                  "density fitting.")


def _make_eris_incore(mycc, mo_coeff=None, ao2mofn=None):
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb

    moa = eris.mo_coeff[0]
    mob = eris.mo_coeff[1]
    nmoa = moa.shape[1]
    nmob = mob.shape[1]

    if callable(ao2mofn):
        eri_aa = ao2mofn(moa).reshape([nmoa]*4)
        eri_bb = ao2mofn(mob).reshape([nmob]*4)
        eri_ab = ao2mofn((moa,moa,mob,mob))
    elif isinstance(ao2mofn, tuple) and callable(ao2mofn[0]):
        eri_aa = ao2mofn[0](moa).reshape([nmoa]*4)
        eri_bb = ao2mofn[2](mob).reshape([nmob]*4)
        eri_ab = ao2mofn[1]((moa,moa,mob,mob))
    else:
        eri_aa = ao2mo.restore(1, ao2mo.full(mycc._scf._eri, moa), nmoa)
        eri_bb = ao2mo.restore(1, ao2mo.full(mycc._scf._eri, mob), nmob)
        eri_ab = ao2mo.general(mycc._scf._eri, (moa,moa,mob,mob), compact=False)
    eri_ba = eri_ab.reshape(nmoa,nmoa,nmob,nmob).transpose(2,3,0,1)

    eri_aa = eri_aa.reshape(nmoa,nmoa,nmoa,nmoa)
    eri_ab = eri_ab.reshape(nmoa,nmoa,nmob,nmob)
    eri_ba = eri_ba.reshape(nmob,nmob,nmoa,nmoa)
    eri_bb = eri_bb.reshape(nmob,nmob,nmob,nmob)
    eris.oooo = eri_aa[:nocca,:nocca,:nocca,:nocca].copy()
    eris.ovoo = eri_aa[:nocca,nocca:,:nocca,:nocca].copy()
    eris.ovov = eri_aa[:nocca,nocca:,:nocca,nocca:].copy()
    eris.oovv = eri_aa[:nocca,:nocca,nocca:,nocca:].copy()
    eris.ovvo = eri_aa[:nocca,nocca:,nocca:,:nocca].copy()
    eris.ovvv = eri_aa[:nocca,nocca:,nocca:,nocca:].copy()
    eris.vvvv = eri_aa[nocca:,nocca:,nocca:,nocca:].copy()

    eris.OOOO = eri_bb[:noccb,:noccb,:noccb,:noccb].copy()
    eris.OVOO = eri_bb[:noccb,noccb:,:noccb,:noccb].copy()
    eris.OVOV = eri_bb[:noccb,noccb:,:noccb,noccb:].copy()
    eris.OOVV = eri_bb[:noccb,:noccb,noccb:,noccb:].copy()
    eris.OVVO = eri_bb[:noccb,noccb:,noccb:,:noccb].copy()
    eris.OVVV = eri_bb[:noccb,noccb:,noccb:,noccb:].copy()
    eris.VVVV = eri_bb[noccb:,noccb:,noccb:,noccb:].copy()

    eris.ooOO = eri_ab[:nocca,:nocca,:noccb,:noccb].copy()
    eris.ovOO = eri_ab[:nocca,nocca:,:noccb,:noccb].copy()
    eris.ovOV = eri_ab[:nocca,nocca:,:noccb,noccb:].copy()
    eris.ooVV = eri_ab[:nocca,:nocca,noccb:,noccb:].copy()
    eris.ovVO = eri_ab[:nocca,nocca:,noccb:,:noccb].copy()
    eris.ovVV = eri_ab[:nocca,nocca:,noccb:,noccb:].copy()
    eris.vvVV = eri_ab[nocca:,nocca:,noccb:,noccb:].copy()

    #eris.OOoo = eri_ba[:noccb,:noccb,:nocca,:nocca].copy()
    eris.OVoo = eri_ba[:noccb,noccb:,:nocca,:nocca].copy()
    #eris.OVov = eri_ba[:noccb,noccb:,:nocca,nocca:].copy()
    eris.OOvv = eri_ba[:noccb,:noccb,nocca:,nocca:].copy()
    eris.OVvo = eri_ba[:noccb,noccb:,nocca:,:nocca].copy()
    eris.OVvv = eri_ba[:noccb,noccb:,nocca:,nocca:].copy()
    #eris.VVvv = eri_ba[noccb:,noccb:,nocca:,nocca:].copy()

    if not callable(ao2mofn):
        ovvv = eris.ovvv.reshape(nocca*nvira,nvira,nvira)
        eris.ovvv = lib.pack_tril(ovvv).reshape(nocca,nvira,nvira*(nvira+1)//2)
        eris.vvvv = ao2mo.restore(4, eris.vvvv, nvira)

        OVVV = eris.OVVV.reshape(noccb*nvirb,nvirb,nvirb)
        eris.OVVV = lib.pack_tril(OVVV).reshape(noccb,nvirb,nvirb*(nvirb+1)//2)
        eris.VVVV = ao2mo.restore(4, eris.VVVV, nvirb)

        ovVV = eris.ovVV.reshape(nocca*nvira,nvirb,nvirb)
        eris.ovVV = lib.pack_tril(ovVV).reshape(nocca,nvira,nvirb*(nvirb+1)//2)
        vvVV = eris.vvVV.reshape(nvira**2,nvirb**2)
        idxa = np.tril_indices(nvira)
        idxb = np.tril_indices(nvirb)
        eris.vvVV = lib.take_2d(vvVV, idxa[0]*nvira+idxa[1], idxb[0]*nvirb+idxb[1])

        OVvv = eris.OVvv.reshape(noccb*nvirb,nvira,nvira)
        eris.OVvv = lib.pack_tril(OVvv).reshape(noccb,nvirb,nvira*(nvira+1)//2)
    return eris