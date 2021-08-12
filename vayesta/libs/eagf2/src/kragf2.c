/*  Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
   
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
   
        http://www.apache.org/licenses/LICENSE-2.0
   
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
   
 *
 *  Author: Oliver J. Backhouse <olbackhouse@gmail.com>
 *          Alejandro Santana-Bonilla <alejandro.santana_bonilla@kcl.ac.uk>
 *          George H. Booth <george.booth@kcl.ac.uk>
 */

#include<stdlib.h>
#include<assert.h>
#include<math.h>
#include<complex.h>
#include<stdio.h>

#include "ragf2.h"



/*
 *  b_xz = a_xiz
 */
void KAGF2slice_0i2(double complex *a,
                    int x,
                    int y,
                    int z,
                    int idx,
                    double complex *b)
{
    double complex *pa, *pb;
    int i, k;

    for (i = 0; i < x; i++) {
        pb = b + i*z;
        pa = a + i*y*z + idx*z;
        for (k = 0; k < z; k++) {
            pb[k] = pa[k];
        }
    }
}


/*
 *  b_xy = a_xyi
 */
void KAGF2slice_01i(double complex *a,
                    int x,
                    int y,
                    int z,
                    int idx,
                    double complex *b)
{
    double complex *pa, *pb;
    int i, j;

    for (i = 0; i < x; i++) {
        pb = b + i*y;
        pa = a + i*y*z + idx;
        for (j = 0; j < y; j++) {
            pb[j] = pa[j*z];
        }
    }
}


/*
 *  d_xy = a + b_x - c_y
 */
void KAGF2sum_inplace_ener(double complex a,
                           double complex *b,
                           double complex *c,
                           int x,
                           int y,
                           double complex *d)
{
    double complex *pd;
    int i, j;

    for (i = 0; i < x; i++) {
        pd = d + i*y;
        for (j = 0; j < y; j++) {
            pd[j] = a + b[i] - c[j];
        }
    }
}


/*
 *  b_x = a_x * b_x
 */
void KAGF2prod_inplace(double *a,
                       double complex *b,
                       int x)
{
    int i;

    for (i = 0; i < x; i++) {
        b[i] *= a[i];
    }
}


/*
 *  b_xy = a_y * b_xy
 */
void KAGF2prod_inplace_ener(double *a,
                            double complex *b,
                            int x,
                            int y)
{
    double complex *pb;
    int i;

    for (i = 0; i < x; i++) {
        pb = b + i*y;
        KAGF2prod_inplace(a, pb, y);
    }
}


/*
 *  b_x = alpha * a_x + beta * b_x
 */
void KAGF2sum_inplace(double complex *a,
                      double complex *b,
                      int x,
                      double complex alpha,
                      double complex beta)
{
    int i;

    for (i = 0; i < x; i++) {
        b[i] *= beta;
        b[i] += alpha * a[i];
    }
}


void array_conj(double complex *a,
                int x)
{
    int i;

    for (i = 0; i < x; i++) {
        a[i] = conj(a[i]);
    }
}


/*
 *  exact ERI
 *  vv_xy = (xi|ja) [2(yi|ja) - (yj|ia)]
 *  vev_xy = (xi|ja) [2(yi|ja) - (yj|ia)] (ei + ej - ea)
 */
void KAGF2ee_vv_vev_islice(double complex *xija,
                           double complex *xjia,
                           double *e_i,
                           double *e_j,
                           double *e_a,
                           double os_factor,
                           double ss_factor,
                           int nmo,
                           int ni,
                           int nj,
                           int na,
                           int istart,
                           int iend,
                           double complex *vv,
                           double complex *vev)
{
    const double complex D1 = 1.0;
    const char TRANS_T = 'T';
    const char TRANS_N = 'N';

    const int nja = nj * na;
    const int nxj = nmo * nj;
    const double complex fpos = os_factor + ss_factor;
    const double complex fneg = -1.0 * ss_factor;

#pragma omp parallel
{
    double *eja = calloc(nja, sizeof(double));
    double complex *xia = calloc(nmo*nja, sizeof(double complex));
    double complex *xja = calloc(nmo*nja, sizeof(double complex));

    double complex *vv_priv = calloc(nmo*nmo, sizeof(double complex));
    double complex *vev_priv = calloc(nmo*nmo, sizeof(double complex));

    int i;

#pragma omp for
    for (i = istart; i < iend; i++) {
        // build xija
        KAGF2slice_0i2(xija, nmo, ni, nja, i, xja);

        // build xjia
        KAGF2slice_0i2(xjia, nxj, ni, na, i, xia);

        // inplace xjia = 2 * xija - xjia
        KAGF2sum_inplace(xja, xia, nmo*nja, fpos, fneg);
        array_conj(xia, nmo*nja);

        // build eija = ei + ej - ea
        AGF2sum_inplace_ener(e_i[i], e_j, e_a, nj, na, eja);

        // vv_xy += xija * (2 yija - yjia)
        zgemm_(&TRANS_T, &TRANS_N, &nmo, &nmo, &nja, &D1, xia, &nja, xja, &nja, &D1, vv_priv, &nmo);

        // inplace xija = eija * xija
        KAGF2prod_inplace_ener(eja, xja, nmo, nja);

        // vev_xy += xija * eija * (2 yija - yjia)
        zgemm_(&TRANS_T, &TRANS_N, &nmo, &nmo, &nja, &D1, xia, &nja, xja, &nja, &D1, vev_priv, &nmo);
    }

    free(eja);
    free(xia);
    free(xja);

#pragma omp critical
    for (i = 0; i < (nmo*nmo); i++) {
        vv[i] += vv_priv[i];
        vev[i] += vev_priv[i];
    }

    free(vv_priv);
    free(vev_priv);
}
}


/*
 *  density fitting
 *  (xi|ja) = (xi|Q)(Q|ja)
 *  vv_xy = (xi|ja) [2(yi|ja) - (yj|ia)]
 *  vev_xy = (xi|ja) [2(yi|ja) - (yj|ia)] (ei + ej - ea)
 */
void KAGF2df_vv_vev_islice(double complex *qxi,
                           double complex *qja,
                           double complex *qxj,
                           double complex *qia,
                           double *e_i,
                           double *e_j,
                           double *e_a,
                           double os_factor,
                           double ss_factor,
                           int nmo,
                           int ni,
                           int nj,
                           int na,
                           int naux,
                           int istart,
                           int iend,
                           double complex *vv,
                           double complex *vev)
{
    const double complex D0 = 0.0;
    const double complex D1 = 1.0;
    const char TRANS_T = 'T';
    const char TRANS_N = 'N';

    const int nja = nj * na;
    const int nxj = nmo * nj;
    const double complex fpos = os_factor + ss_factor;
    const double complex fneg = -1.0 * ss_factor;

#pragma omp parallel
{
    double complex *qa = calloc(naux*na, sizeof(double complex));
    double complex *qx = calloc(naux*nmo, sizeof(double complex));
    double *eja = calloc(nja, sizeof(double));
    double complex *xia = calloc(nmo*nja, sizeof(double complex));
    double complex *xja = calloc(nmo*nja, sizeof(double complex));

    double complex *vv_priv = calloc(nmo*nmo, sizeof(double complex));
    double complex *vev_priv = calloc(nmo*nmo, sizeof(double complex));

    int i;

#pragma omp for
    for (i = istart; i < iend; i++) {
        // build qx
        KAGF2slice_01i(qxi, naux, nmo, ni, i, qx);

        // build qa
        KAGF2slice_0i2(qia, naux, ni, na, i, qa);

        // build xija = xq * qja
        zgemm_(&TRANS_N, &TRANS_T, &nja, &nmo, &naux, &D1, qja, &nja, qx, &nmo, &D0, xja, &nja);

        // build xjia = xjq * qa
        zgemm_(&TRANS_N, &TRANS_T, &na, &nxj, &naux, &D1, qa, &na, qxj, &nxj, &D0, xia, &na);

        // inplace xjia = 2 * xija - xjia
        KAGF2sum_inplace(xja, xia, nmo*nja, fpos, fneg);
        array_conj(xia, nmo*nja);

        // build eija = ei + ej - ea
        AGF2sum_inplace_ener(e_i[i], e_j, e_a, nj, na, eja);

        // vv_xy += xija * (2 yija - yjia)
        zgemm_(&TRANS_T, &TRANS_N, &nmo, &nmo, &nja, &D1, xia, &nja, xja, &nja, &D1, vv_priv, &nmo);

        // inplace xija = eija * xija
        KAGF2prod_inplace_ener(eja, xja, nmo, nja);

        // vev_xy += xija * eija * (2 yija - yjia)
        zgemm_(&TRANS_T, &TRANS_N, &nmo, &nmo, &nja, &D1, xia, &nja, xja, &nja, &D1, vev_priv, &nmo);
    }

    free(qa);
    free(qx);
    free(eja);
    free(xia);
    free(xja);

#pragma omp critical
    for (i = 0; i < (nmo*nmo); i++) {
        vv[i] += vv_priv[i];
        vev[i] += vev_priv[i];
    }

    free(vv_priv);
    free(vev_priv);
}
}
