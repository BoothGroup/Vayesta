#include<stdint.h>
#include<stdbool.h>
#include<stdlib.h>
#include<complex.h>
#include<assert.h>
#include<string.h>
#include<math.h>
#include "cblas.h"
#include<stdio.h>

#ifdef _OPENMP
#include<omp.h>
#else
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))


void j3c_ao2mo_zgemm(
        // Input:
        int32_t np,             // Number of orbitals in P dimension
        int32_t nq,             // Number of orbitals in Q dimension
        int32_t naux,           // Number of auxiliary functions
        int32_t ni,             // Number of orbitals in I dimension
        int32_t nj,             // Number of orbitals in J dimension
        double complex *Lpq,    // Input three-centre integrals
        double complex *cpi,    // Input coefficients for P->I
        double complex *cqj,    // Input coefficients for Q->J
        // Output:
        double complex *out)    // Output array
{
    // Contract lpq,pi*,qj->lij for complex arrays

    size_t i, l, p, q;

    const double complex Z1 = 1.0;
    const double complex Z0 = 0.0;

    double complex *work1 = calloc(naux*nq*np, sizeof(double complex));
    double complex *work2 = calloc(naux*nq*ni, sizeof(double complex));
    double complex *work3 = calloc(naux*ni*nq, sizeof(double complex));

    if (cpi != NULL) {
        // Lpq->Lqp*
        for (l = 0; l < naux; l++) {
        for (p = 0; p < np; p++) {
        for (q = 0; q < nq; q++) {
            work1[l*np*nq + q*np + p] = conj(Lpq[l*np*nq + p*nq + q]);
        }}}

        // Lqp,pi->Lqi
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, naux*nq, ni, np,
                    &Z1, work1, np, cpi, ni, &Z0, work2, ni);

        // Lqi->Liq*
        for (l = 0; l < naux; l++) {
        for (q = 0; q < nq; q++) {
        for (i = 0; i < ni; i++) {
            work3[l*nq*ni + i*nq + q] = conj(work2[l*nq*ni + q*ni + i]);
        }}}
    }
    else {
        assert(ni == np);
        for (l = 0; l < naux; l++) {
        for (p = 0; p < np; p++) {
        for (q = 0; q < nq; q++) {
            work3[l*np*nq + p*nq + q] = Lpq[l*np*nq + p*nq + q];
        }}}
    }

    if (cqj != NULL) {
        // Liq,qj->Lij
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, naux*ni, nj, nq,
                    &Z1, work3, nq, cqj, nj, &Z0, out, nj);
    }
    else {
        assert(nj == nq);
    }

    free(work1);
    free(work2);
    free(work3);
}


void block_matrix_indices(
        // Input:
        int32_t np,
        int32_t nq,
        int32_t npblk,
        int32_t nqblk,
        // Output:
        size_t *ps,
        size_t *qs)
{
    size_t p, q;

    for (p = 0; p < np; p += npblk) {
        ps[p] = p;
    }
    for (q = 0; q < nq; q += nqblk) {
        qs[q] = q;
    }

    ps[npblk] = np;
    qs[nqblk] = nq;
}


void build_energy_tensor(
        // Input:
        int32_t ni,             // Number of orbitals in I dimension
        int32_t na,             // Number of orbitals in A dimension
        int32_t nj,             // Number of orbitals in J dimension
        double *ei,             // Orbital energies in I dimension
        double *ea,             // Orbital energies in A dimension
        double *ej,             // Orbital energies in J dimension
        // Output:
        double *eiaj)           // Output array
{
    // Build the energy tensor eiaj(i,a,j) = ei(i) - ea(a) + ej(j)

    const int32_t naj = na * nj;
    size_t i, a, j;

    for (i = 0; i < ni; i++) {
    for (j = 0; j < nj; j++) {
    for (a = 0; a < na; a++) {
        eiaj[i*naj + a*nj + j] = ei[i] - ea[a] + ej[j];
    }}}
}


void build_antisymm_real(
        // Input:
        int32_t q0,             // Start index in Q dimension
        int32_t q1,             // End index in Q dimension
        int32_t ni,             // Number of orbitals in I dimension
        int32_t na,             // Number of orbitals in A dimension
        int32_t nj,             // Number of orbitals in J dimension
        double *qiaj,           // (qi|aj) integrals, if NULL then use piaj
        double *qjai,           // (qj|ai) integrals, if NULL then use qjai
        // Output:
        double *out)            // Output array
{
    // Build 2 (qi|aj) - (qj|ai) for real integrals

    const int32_t naj = na * nj;
    const int32_t nai = na * ni;
    const int32_t niaj = ni * naj;
    size_t q, i, a, j;

    for (q = q0; q < q1; q++) {
    for (i = 0; i < ni; i++) {
    for (a = 0; a < na; a++) {
    for (j = 0; j < nj; j++) {
        out[(q-q0)*niaj + i*naj + a*nj + j] = 2.0 * qiaj[q*niaj + i*naj + a*nj + j]
                                                  - qjai[q*niaj + j*nai + a*ni + i];
    }}}}
}


void build_antisymm_cplx(
        // Input:
        int32_t q0,             // Start index in Q dimension
        int32_t q1,             // End index in Q dimension
        int32_t ni,             // Number of orbitals in I dimension
        int32_t na,             // Number of orbitals in A dimension
        int32_t nj,             // Number of orbitals in J dimension
        double complex *qiaj,   // (qi|aj) integrals, if NULL then use piaj
        double complex *qjai,   // (qj|ai) integrals, if NULL then use qjai
        // Output:
        double complex *out)    // Output array
{
    // Build 2 (qi|aj) - (qj|ai) for complex integrals

    const int32_t naj = na * nj;
    const int32_t nai = na * ni;
    const int32_t niaj = ni * naj;
    size_t q, i, a, j;

    for (q = q0; q < q1; q++) {
    for (i = 0; i < ni; i++) {
    for (a = 0; a < na; a++) {
    for (j = 0; j < nj; j++) {
        out[(q-q0)*niaj + i*naj + a*nj + j] = 2.0 * qiaj[q*niaj + i*naj + a*nj + j]
                                                  - qjai[q*niaj + j*nai + a*ni + i];
    }}}}
}


void dgemm_diag(
        // Input:
        int32_t m,
        int32_t k,
        const double *alpha,
        double *a,
        double *b,
        const double *beta,
        double *c)
{
    // Compute c(m,k) = a(m,k) * b(k) for real a

    size_t i, j, ij;

    for (i = 0, ij = 0; i < m; i++) {
    for (j = 0; j < k; j++, ij++) {
        c[ij] = alpha[0] * a[ij] * b[j] + beta[0] * c[ij];
    }}
}


void zgemm_diag(
        // Input:
        int32_t m,
        int32_t k,
        const double complex *alpha,
        double complex *a,
        double *b,
        const double complex *beta,
        double complex *c)
{
    // Compute c(m,k) = a(m,k) * b(k) for complex a

    size_t i, j, ij;

    for (i = 0, ij = 0; i < m; i++) {
    for (j = 0; j < k; j++, ij++) {
        c[ij] = alpha[0] * a[ij] * b[j] + beta[0] * c[ij];
    }}
}


int32_t construct_moments_real_4c(
        // Input:
        int32_t np,             // Number of orbitals in P dimension
        int32_t nq,             // Number of orbitals in Q dimension
        int32_t ni,             // Number of orbitals in I dimension
        int32_t na,             // Number of orbitals in A dimension
        int32_t nj,             // Number of orbitals in J dimension
        int32_t nmom,           // Number of moments to calculate
        int32_t npblk,          // Block size for P
        int32_t nqblk,          // Block size for Q
        double *piaj,           // (pi|aj) integrals
        double *qiaj,           // (qi|aj) integrals, if NULL then use piaj
        double *qjai,           // (qj|ai) integrals, if NULL then use qjai
        double *ei,             // Orbital energies in I dimension
        double *ea,             // Orbital energies in A dimension
        double *ej,             // Orbital energies in J dimension, if NULL then use ei
        // Output:
        double *t)              // Output array
{
    // Contract moments of the MP2 self-energy for real, 4c integrals

    int32_t ierr = 0;

    // Check for null pointers
    if (qiaj == NULL) {
        qiaj = piaj;
    }
    if (qjai == NULL) {
        qjai = qiaj;
    }
    if (ej == NULL) {
        ej = ei;
    }

    // Constant scalars
    const double D0 = 0.0;
    const double D1 = 1.0;

    // Compound indices
    const int32_t npq = np * nq;
    const int32_t npqblk = npblk * nqblk;
    const int32_t naj = na * nj;
    const int32_t niaj = ni * naj;

    // Build energy difference tensor
    double *eiaj = calloc(niaj, sizeof(double));
    build_energy_tensor(ni, na, nj, ei, ea, ej, eiaj);

    // Compute block indices
    size_t *ps = calloc(npblk+1, sizeof(size_t));
    size_t *qs = calloc(nqblk+1, sizeof(size_t));
    block_matrix_indices(np, nq, npblk, nqblk, ps, qs);

    // Begin iterations
#pragma omp parallel
    {
        size_t p, q, p0, q0, p1, q1, np0, nq0, npq0, n, xy, x, y;
        double *tblk, *work;

#pragma omp for reduction(+:ierr)
        for (xy = 0; xy < npqblk; xy++) {
            x = xy / nqblk;
            y = xy % nqblk;
            p0 = ps[x];
            q0 = qs[y];
            p1 = ps[x+1];
            q1 = qs[y+1];
            np0 = p1 - p0;
            nq0 = q1 - q0;
            npq0 = np0 * nq0;
            if (npq0 == 0) { continue; }  // shouldn't happen
            tblk = calloc(nmom*npq0, sizeof(double));
            work = calloc(nq0*niaj, sizeof(double));

            // Build work(q,i,a,j) = 2 * qiaj(q,i,a,j) - qjai(q,j,a,i)
            build_antisymm_real(q0, q1, ni, na, nj, qiaj, qjai, work);

            // Iterate over the number of desired moments
            for (n = 0; n < nmom; n++) {
                // Contract t(n,p,q) = piaj(p,i,a,j) work(q,i,a,j)
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, np0, nq0, niaj,
                            D1, &(piaj[p0*niaj]), niaj, work, niaj, D0, &(tblk[n*npq0]), nq0);

                // Contract work(q,i,a,j) = work(q,i,a,j) * (ei(i) - ea(a) - ei(j))
                if ((n+1) != nmom) {
                    dgemm_diag(nq0, niaj, &D1, work, eiaj, &D0, work);
                }
            }

            // Add block into output array
#pragma omp critical
            {
                for (n = 0; n < nmom; n++) {
                for (p = 0; p < np0; p++) {
                for (q = 0; q < nq0; q++) {
                    t[n*npq + (p0+p)*nq + (q0+q)] = tblk[n*npq0 + p*nq0 + q];
                }}}
            }

            free(tblk);
            free(work);
        }
    }

    free(eiaj);
    free(ps);
    free(qs);

    return ierr;
}


int32_t construct_moments_cplx_4c(
        // Input:
        int32_t np,             // Number of orbitals in P dimension
        int32_t nq,             // Number of orbitals in Q dimension
        int32_t ni,             // Number of orbitals in I dimension
        int32_t na,             // Number of orbitals in A dimension
        int32_t nj,             // Number of orbitals in J dimension
        int32_t nmom,           // Number of moments to calculate
        int32_t npblk,          // Block size for P
        int32_t nqblk,          // Block size for Q
        double complex *piaj,   // (pi|aj) integrals
        double complex *qiaj,   // (qi|aj) integrals, if NULL then use piaj
        double complex *qjai,   // (qj|ai) integrals, if NULL then use qiaj
        double *ei,             // Orbital energies in I dimension
        double *ea,             // Orbital energies in A dimension
        double *ej,             // Orbital energies in J dimension, if NULL then use ei
        // Output:
        double complex *t)      // Output array
{
    // Contract moments of the MP2 self-energy for complex, 4c integrals

    int32_t ierr = 0;

    // Check for null pointers
    if (qiaj == NULL) {
        qiaj = piaj;
    }
    if (qjai == NULL) {
        qjai = qiaj;
    }
    if (ej == NULL) {
        ej = ei;
    }

    // Constant scalars
    const double complex D0 = 0.0;
    const double complex D1 = 1.0;

    // Compound indices
    const int32_t npq = np * nq;
    const int32_t npqblk = npblk * nqblk;
    const int32_t naj = na * nj;
    const int32_t niaj = ni * naj;

    // Build energy difference tensor
    double *eiaj = calloc(niaj, sizeof(double));
    build_energy_tensor(ni, na, nj, ei, ea, ej, eiaj);

    // Compute block indices
    size_t *ps = calloc(npblk+1, sizeof(size_t));
    size_t *qs = calloc(nqblk+1, sizeof(size_t));
    block_matrix_indices(np, nq, npblk, nqblk, ps, qs);

    // Begin iterations
#pragma omp parallel
    {
        size_t p, q, p0, q0, p1, q1, np0, nq0, npq0, n, xy, x, y;
        double complex *tblk, *work;

#pragma omp for reduction(+:ierr)
        for (xy = 0; xy < npqblk; xy++) {
            x = xy / nqblk;
            y = xy % nqblk;
            p0 = ps[x];
            q0 = qs[y];
            p1 = ps[x+1];
            q1 = qs[y+1];
            np0 = p1 - p0;
            nq0 = q1 - q0;
            npq0 = np0 * nq0;
            if (npq0 == 0) { continue; }  // shouldn't happen
            tblk = calloc(nmom*npq0, sizeof(double complex));
            work = calloc(nq0*niaj, sizeof(double complex));

            // Build work(q,i,a,j) = 2 * qiaj(q,i,a,j) - qjai(q,j,a,i)
            build_antisymm_cplx(q0, q1, ni, na, nj, qiaj, qjai, work);

            // Iterate over the number of desired moments
            for (n = 0; n < nmom; n++) {
                // Contract t(n,p,q) = piaj(p,i,a,j) work(q,i,a,j)
                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, np0, nq0, niaj,
                            &D1, &(piaj[p0*niaj]), niaj, work, niaj, &D0, &(tblk[n*npq0]), nq0);

                // Contract work(q,i,a,j) = work(q,i,a,j) * (ei(i) - ea(a) - ei(j))
                if ((n+1) != nmom) {
                    zgemm_diag(nq0, niaj, &D1, work, eiaj, &D0, work);
                }
            }

            // Add block into output array   TODO does this need to be critical without +=?
#pragma omp critical
            {
                for (n = 0; n < nmom; n++) {
                for (p = 0; p < np0; p++) {
                for (q = 0; q < nq0; q++) {
                    t[n*npq + (p0+p)*nq + (q0+q)] = tblk[n*npq0 + p*nq0 + q];
                }}}
            }

            free(tblk);
            free(work);
        }
    }

    free(eiaj);
    free(ps);
    free(qs);

    return ierr;
}


struct GreensFunction
{
    int32_t nocc;
    int32_t nvir;
    double *ei;
    double *ea;
    double complex *ci;
    double complex *ca;
};


int32_t construct_moments_kagf2(
        // Input:
        int32_t nmo,                // Number of MOs
        int32_t naux,               // Number of auxiliary functions
        int32_t nkpts,              // Number of k-points
        int32_t nkptlist,           // Number of k-points to compute moments for
        int32_t nmom,               // Number of moments to compute
        struct GreensFunction *gfs, // Green's functions for each k-point
        double complex *Lpq,        // Three-center k-point dependent integrals
        int32_t *kconserv,          // Indices to satisfy momentum conservation
        int32_t *kptlist,           // List of k-points to compute moments for
        int32_t *krange,            // Range within 0 -> nkptlist * nkpts**2
        // Output:
        double complex *t_occ,      // Output array for occupied moments
        double complex *t_vir)      // Output array for virtual moments
{
    // Compute the moments required for a KAGF2 calculation.
    //TODO frozen
    //TODO Dyson
    //TODO OS, SS
    //TODO diag

    int32_t ierr = 0;

    const double complex Z0 = 0.0;
    const double complex Z1 = 1.0;

    const int32_t KK = nkpts * nkpts;
    const int32_t N2 = nmo * nmo;
    const int32_t LN2 = naux * N2;
    const int32_t KLN2 = nkpts * LN2;

    const size_t k0 = (krange == NULL) ? 0 : krange[0];
    const size_t k1 = (krange == NULL) ? (nkptlist * nkpts * nkpts) : krange[1];

#pragma omp parallel
    {
        size_t ka, ka_, kb, kc, kd, kbc, kabc;
        size_t nbra, nket, n;

        double complex *Lpi, *Laj, *Lpj, *Lai, *Lpa, *Lib, *Lpb, *Lia;
        double complex *piaj, *pjai, *paib, *pbia;
        double complex *ti, *ta;

        ti = calloc(nmom*N2, sizeof(double complex));
        ta = calloc(nmom*N2, sizeof(double complex));

#pragma omp for reduction(+:ierr)
        for (kabc = k0; kabc < k1; kabc++) {
            ka_ = kabc / KK;
            ka = kptlist[ka_];
            kbc = kabc % KK;
            kb = kbc / nkpts;
            kc = kbc % nkpts;
            kd = kconserv[ka*KK + kb*nkpts + kc];

            // Build (pi|aj) integrals
            nbra = nmo * gfs[kb].nocc;
            nket = gfs[kc].nvir * gfs[kb].nocc;
            Lpi = calloc(naux*nbra, sizeof(double complex));
            Laj = calloc(naux*nket, sizeof(double complex));
            piaj = calloc(nbra*nket, sizeof(double complex));

            // Lpq(ka, kb) -> Lpi
            j3c_ao2mo_zgemm(nmo, nmo, naux, nmo, gfs[kb].nocc,
                            &(Lpq[ka*KLN2 + kb*LN2]), NULL, gfs[kb].ci, Lpi);

            // Lpq(kc, kd) -> Laj
            j3c_ao2mo_zgemm(nmo, nmo, naux, gfs[kc].nvir, gfs[kb].nocc,
                            &(Lpq[kc*KLN2 + kd*LN2]), gfs[kc].ca, gfs[kd].ci, Laj);

            // Lpi,Laj -> piaj
            cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nbra, nket, naux,
                        &Z1, Lpi, nbra, Laj, nket, &Z0, piaj, nket);
            free(Lpi);
            free(Laj);


            // Build (pj|ai) integrals
            nbra = nmo * gfs[kd].nocc;
            nket = gfs[kc].nvir * gfs[kb].nocc;
            Lpj = calloc(naux*nbra, sizeof(double complex));
            Lai = calloc(naux*nket, sizeof(double complex));
            pjai = calloc(nbra*nket, sizeof(double complex));

            // Lpq(ka, kd) -> Lpj
            j3c_ao2mo_zgemm(nmo, nmo, naux, nmo, gfs[kd].nocc,
                            &(Lpq[ka*KLN2 + kd*LN2]), NULL, gfs[kd].ci, Lpj);

            // Lpq(kc, kb) -> Lai
            j3c_ao2mo_zgemm(nmo, nmo, naux, gfs[kc].nvir, gfs[kb].nocc,
                            &(Lpq[kc*KLN2 + kb*LN2]), gfs[kc].ca, gfs[kb].ci, Lai);

            // Lpj,Lai -> pjai
            cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nbra, nket, naux,
                        &Z1, Lpj, nbra, Lai, nket, &Z0, pjai, nket);
            free(Lpj);
            free(Lai);


            // Build occupied moments
            construct_moments_cplx_4c(nmo, nmo, gfs[kb].nocc, gfs[kc].nvir, gfs[kd].nocc,
                                      nmom, nmo, nmo, piaj, piaj, pjai,
                                      gfs[kb].ei, gfs[kc].ea, gfs[kd].ei, ti);
            free(piaj);
            free(pjai);


            // Build (pa|ib) integrals
            nbra = nmo * gfs[kb].nvir;
            nket = gfs[kc].nocc * gfs[kd].nvir;
            Lpa = calloc(naux*nbra, sizeof(double complex));
            Lib = calloc(naux*nket, sizeof(double complex));
            paib = calloc(nbra*nket, sizeof(double complex));

            // Lpq(ka, kb) -> Lpa
            j3c_ao2mo_zgemm(nmo, nmo, naux, nmo, gfs[kb].nvir,
                            &(Lpq[ka*KLN2 + kb*LN2]), NULL, gfs[kb].ca, Lpa);

            // Lpq(kc, kd) -> Lib
            j3c_ao2mo_zgemm(nmo, nmo, naux, gfs[kc].nocc, gfs[kd].nvir,
                            &(Lpq[kc*KLN2 + kd*LN2]), gfs[kc].ci, gfs[kd].ca, Lib);

            // Lpa,Lib -> paib
            cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nbra, nket, naux,
                        &Z1, Lpa, nbra, Lib, nket, &Z0, paib, nket);
            free(Lpa);
            free(Lib);


            // Build (pb|ia) integrals
            nbra = nmo * gfs[kd].nvir;
            nket = gfs[kc].nocc * gfs[kb].nvir;
            Lpb = calloc(naux*nbra, sizeof(double complex));
            Lia = calloc(naux*nket, sizeof(double complex));
            pbia = calloc(nbra*nket, sizeof(double complex));

            // Lpq(ka, kd) -> Lpb
            j3c_ao2mo_zgemm(nmo, nmo, naux, nmo, gfs[kd].nvir,
                            &(Lpq[ka*KLN2 + kd*LN2]), NULL, gfs[kd].ca, Lpb);

            // Lpq(kc, kb) -> Lia
            j3c_ao2mo_zgemm(nmo, nmo, naux, gfs[kc].nocc, gfs[kb].nvir,
                            &(Lpq[kc*KLN2 + kb*LN2]), gfs[kc].ci, gfs[kb].ca, Lia);

            // Lpb,Lia -> pbia
            cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nbra, nket, naux,
                        &Z1, Lpb, nbra, Lia, nket, &Z0, pbia, nket);
            free(Lpb);
            free(Lia);


            // Build virtual moments
            construct_moments_cplx_4c(nmo, nmo, gfs[kb].nvir, gfs[kc].nocc, gfs[kd].nvir,
                                      nmom, nmo, nmo, paib, paib, pbia,
                                      gfs[kb].ea, gfs[kc].ei, gfs[kd].ea, ta);
            free(paib);
            free(pbia);


            // Add moments into output arrays
#pragma omp critical
            {
                for (n = 0; n < (nmom*N2); n++) {
                    t_occ[ka*nmom*N2 + n] += ti[n];
                    t_vir[ka*nmom*N2 + n] += ta[n];
                }
            }
        }

        free(ti);
        free(ta);
    }

    return ierr;
}


