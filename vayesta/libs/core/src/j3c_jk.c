#include<stdint.h>
#include<stdbool.h>
#include<stdlib.h>
#include<complex.h>
#include<string.h>
#include<math.h>
#include "mkl.h"

#ifdef _OPENMP
#include<omp.h>
#else
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

#define SQUARE(x) ((x) * (x))


/*
 *  Perform a (1,0,2) transpose
 */
int64_t transpose_102(
        // Input:
        int64_t na,                 // Size of first dimension
        int64_t nb,                 // Size of second dimension
        int64_t nc,                 // Size of third dimension
        double complex *arr,        // Input array (na, nb, nc)
        // Output:
        double complex *out)        // Output array (nb, na, nc)
{
    int64_t ierr = 0;

    const int64_t A = na;
    const int64_t B = nb;
    const int64_t C = nc;
    const int64_t AC = A * C;
    const int64_t BC = B * C;
    size_t i, j, k;

    for (i = 0; i < A; i++) {
    for (j = 0; j < B; j++) {
    for (k = 0; k < C; k++) {
        out[j*AC + i*C + k] = arr[i*BC + j*C + k];
    }}}

    return ierr;
}


/*
 *  Compute J (Coulomb) and K (exchange) matrices for k-point sampled
 *  integrals and density matrices:
 *
 *      J_{rs} = [\sum_{L} (L|pq) (L|rs)] D_{pq}
 *      K_{rs} = [\sum_{L} (L|qs) (L|rp)] D_{pq}
 */
int64_t j3c_jk(
        // Input:
        int64_t nk,                 // Number of k-points
        int64_t nao,                // Number of AOs
        int64_t naux,               // Number of auxiliary functions
        int64_t *naux_slice,        // Slice of auxiliary functions to compute on (2,)
        double complex *cderi,      // 3c integrals (nk, nk, naux, nao, nao)
        double complex *dm,         // Density matrices (nk, nao, nao)
        bool with_j,                // If True, calculate J
        bool with_k,                // If True, calculate K
        // Output:
        double complex *vj,         // J (Coulomb) matrices (nk, nao, nao)
        double complex *vk)         // K (exchange) matrices (nk, nao, nao)
{
    int64_t ierr = 0;
    if (!(with_j || with_k)) {
        return ierr;
    }

    // Constant lengths
    const int64_t K = nk;
    const int64_t N = nao;
    const int64_t L = naux;

    // Auxiliary slice
    const int64_t l0 = naux_slice[0];
    const int64_t l1 = naux_slice[1];
    const int64_t Q = l1 - l0;

    // Constant length products
    const int64_t N2 = N * N;
    const int64_t K2 = K * K;
    const int64_t QN = Q * N;
    const int64_t LN2 = L * N2;
    const int64_t QN2 = Q * N2;
    const int64_t KN2 = K * N2;
    const int64_t KLN2 = K * LN2;

    // Constant values
    const int64_t I1 = 1;
    const double complex Z0 = 0.0;
    const double complex Z1 = 1.0;

    // Conjugate DM to avoid using CblasConjNoTrans
    double complex *dm_conj = calloc(KN2, sizeof(double complex));
    size_t i;
    for (i = 0; i < KN2; i++) {
        dm_conj[i] = conj(dm[i]);
    }

#pragma omp parallel private(i)
    {
        // Work arrays
        double complex *work1 = calloc(Q, sizeof(double complex));
        double complex *work2 = calloc(QN2, sizeof(double complex));
        double complex *work3 = calloc(QN2, sizeof(double complex));
        double complex *vj_priv = calloc(KN2, sizeof(double complex));
        double complex *vk_priv = calloc(KN2, sizeof(double complex));
        size_t i, j, ij;

        if (with_j) {
#pragma omp for reduction(+:ierr)
            for (i = 0; i < K; i++) {
                // cderi(i,i,l,p,q) dm(i,p,q)* -> work1(l)
                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Q, I1, N2, &Z1,
                            &(cderi[i*KLN2 + i*LN2 + l0*N2]), N2, &(dm_conj[i*N2]), I1, &Z0, work1, I1);

                for (j = 0; j < K; j++) {
                    // work1(l) cderi(j,j,l,r,s) -> vj_priv(j,r,s)
                    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, I1, N2, Q, &Z1,
                                work1, Q, &(cderi[j*KLN2 + j*LN2 + l0*N2]), N2, &Z1, &(vj_priv[j*N2]), N2);
                }
            }
        }

        if (with_k) {
#pragma omp for reduction(+:ierr)
            for (ij = 0; ij < K2; ij++) {
                i = ij / K;
                j = ij % K;

                // cderi(j,i,l,r,p) dm(i,p,q) -> work2(l,r,q)
                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, QN, N, N, &Z1,
                            &(cderi[j*KLN2 + i*LN2 + l0*N2]), N, &(dm[i*N2]), N, &Z0, work2, N);

                // work2(l,r,q) -> work3(r,l,q)
                transpose_102(Q, N, N, &(work2[0]), &(work3[0]));

                // work3(r,l,q) cderi(i,j,l,q,s) -> vk_priv(j,r,s)
                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, QN, &Z1,
                            work3, QN, &(cderi[i*KLN2 + j*LN2 + l0*N2]), N, &Z1, &(vk_priv[j*N2]), N);
            }
        }

        free(work1);
        free(work2);
        free(work3);

#pragma omp critical
        {
            for (i = 0; i < KN2; i++) {
                vj[i] += vj_priv[i] / K;
                vk[i] += vk_priv[i] / K;
            }
        }

        free(vj_priv);
        free(vk_priv);
    }

    free(dm_conj);

    return ierr;
}
