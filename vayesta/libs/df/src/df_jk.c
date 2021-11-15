#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define USE_BLAS

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "omp.h"
#include "hdf5.h"
#include "cblas.h"

///* Get size (number of elements) in HDF5 dataset */
//size_t get_dataset_size(hid_t h5dset) {
//    size_t size = 1;
//
//    hid_t dspace = H5Dget_space(h5dset);
//    size_t ndims = H5Sget_simple_extent_ndims(dspace);
//    hsize_t dims[ndims];
//    H5Sget_simple_extent_dims(dspace, dims, NULL);
//    for (int i = 0; i < ndims ; i++) {
//        size = (size * dims[i]);
//        printf("dim %d= %lld\n", i, dims[i]);
//    }
//    printf("total size= %ld\n", size);
//    return size;
//}

int unpack_tril(size_t n, size_t ntril, double *tril, double *out)
{
    if (out == NULL) return -1;

    for (size_t i = 0, idx = 0; i < n; i++) {
        for (size_t j = 0; j <= i; j++, idx++) {
            out[i*n + j] = tril[idx];
        }
    }
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i+1; j < n; j++) {
            out[i*n + j] = out[j*n + i];
        }
    }
    return 0;
}

int read_j3c(
        /* In */
        char *filename, char *group, char *kpt, char *dset,
        hsize_t offset[2], hsize_t count[2],
        /* Out */
        double *out)
{
    hid_t file_id, grp_id, kpt_id, dset_id;
    hid_t mem_id, dspace_id;
    herr_t status;

    const hsize_t stride[2] = {1, 1};
    const hsize_t block[2] = {1, 1};

    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    grp_id = H5Gopen(file_id, group, H5P_DEFAULT);
    kpt_id = H5Gopen(grp_id, kpt, H5P_DEFAULT);
    dset_id = H5Dopen(kpt_id, dset, H5P_DEFAULT);

    dspace_id = H5Dget_space(dset_id);
    mem_id = H5Screate_simple(2, count, NULL);
    status = H5Sselect_hyperslab(dspace_id, H5S_SELECT_SET, offset, stride, count, block);
    if (status != 0) return status;
    status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, mem_id, dspace_id, H5P_DEFAULT, out);
    if (status != 0) return status;
    status = H5Sclose(mem_id);
    if (status != 0) return status;
    status = H5Sclose(dspace_id);
    if (status != 0) return status;

    status = H5Dclose(dset_id);
    if (status != 0) return status;
    status = H5Gclose(kpt_id);
    if (status != 0) return status;
    status = H5Gclose(grp_id);
    if (status != 0) return status;
    status = H5Fclose(file_id);
    if (status != 0) return status;

    return 0;
}

int64_t df_rhf_jk(
        /* In */
        int64_t nao,        // Number of AOs
        int64_t nocc,       // Number of occupied MOs
        int64_t naux,       // Number of DF functions
        char *filename,     // File with 3c-integrals
        double *mo_coeff,   // (nao, nocc) occupied MO coefficients
        int64_t blksize,
        /* Out */
        double *vj,         // (nao, nao) Coulomb matrix
        double *vk          // (nao, nao) exchange matrix
        )
{
    int64_t ierr = 0;
    const int64_t naopair = nao * (nao+1) / 2;

    size_t nblks = (int) (((double)(naux + blksize - 1)) / (double) blksize);
    printf("n(Aux)= %ld  blksize= %ld  nblks= %ld\n", naux, blksize, nblks);

    if ((vj == NULL) || (vk == NULL)) return -1;

#pragma omp parallel
    {
    double *work1 = malloc(blksize*naopair * sizeof(double));
    double *work2 = malloc(nao*nao * sizeof(double));
    double *work3 = malloc(nao * sizeof(double));

#pragma omp for reduction(+:ierr)
    for (size_t iblk = 0; iblk < nblks ; iblk++) {

        /* Read block from HDF5 file */
        size_t l0 = iblk * blksize;
        size_t l1 = MIN(l0 + blksize, naux);
        printf("block %3ld [%4ld:%4ld] on OMP thread %d\n", iblk, l0, l1, omp_get_thread_num());
        hsize_t offset[2] = {l0, 0};
        hsize_t count[2] = {l1-l0, naopair};
#pragma omp critical (hdf5read)
        read_j3c(filename, "j3c", "0", "0", offset, count, work1);

        for (size_t lidx = 0, l = l0; l < l1 ; lidx++, l++) {

            memset(work2, 0, nao*nao * sizeof(double));
            unpack_tril(nao, naopair, &(work1[lidx*naopair]), work2);

            double locc = 0;
#ifdef USE_BLAS
            for (size_t i = 0; i < nocc ; i++) {
                /* (L|ab) * C_bi -> (L|ai) [work3] */
                cblas_dgemv(CblasRowMajor, CblasNoTrans, nao, nao, 1.0, work2, nao, &(mo_coeff[i]), nocc, 0.0, work3, 1);
                /* For J: (L|ai) * C_ai -> L [locc] */
                locc += cblas_ddot(nao, work3, 1, &(mo_coeff[i]), nocc);
                /* For K: (L|ai) * (L|bi) -> K_ab */
#pragma omp critical (vk)
                cblas_dger(CblasRowMajor, nao, nao, 1.0, work3, 1, work3, 1, vk, nao);
            }

            /* For J: L * (L|ab) -> J_ab */
#pragma omp critical (vj)
            cblas_daxpy(nao*nao, locc, work2, 1, vj, 1);
#else
            for (size_t i = 0; i < nocc ; i++) {
                /* (L|ab) * C_bi -> (L|ai) [work3] */
                memset(work3, 0, nao * sizeof(double));
                for (size_t a = 0; a < nao ; a++) {
                    for (size_t b = 0; b < nao ; b++) {
                        work3[a] += work2[a*nao+b] * mo_coeff[b*nocc+i];
                    }
                }
                /* For J: (L|ai) * C_ai -> L [locc] */
                for (size_t a = 0; a < nao ; a++) {
                    locc += work3[a] * mo_coeff[a*nocc+i];
                }
                /* For K: (L|ai) * (L|bi) -> K_ab */
#pragma omp critical (vk)
                for (size_t a = 0; a < nao ; a++) {
                    for (size_t b = 0; b < nao ; b++) {
                        vk[a*nao+b] += work3[a] * work3[b];
                    }
                }
            }
            /* For J: L * (L|ab) -> J_ab */
#pragma omp critical (vj)
            for (size_t a = 0; a < nao ; a++) {
                for (size_t b = 0; b < nao ; b++) {
                    vj[a*nao+b] += locc * work2[a*nao+b];
                }
            }
#endif
        }

    } /* for iblk */
    free(work1);
    free(work2);
    free(work3);
    } /* OMP parallel */

    return ierr;
}
