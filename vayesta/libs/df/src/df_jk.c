#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hdf5.h"

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
//
//
//int load_dataset_double(
//        /* In */
//        hid_t h5grp,
//        char *key,
//        /* InOut */
//        size_t *size,
//        double *data)
//{
//    int ierr = 0;
//    hid_t h5status, dset;
//
//    dset = H5Dopen(h5grp, key, H5P_DEFAULT);
//
//    /* Get size */
//    size_t size = get_dataset_size(dset);
//    printf("total size= %ld\n", size);
//
//    if (data == NULL)
//        data = malloc(size * sizeof(double));
//    if (data == NULL) {
//        ierr = -1;
//        goto EXIT;
//    }
//
//    /* Load into memory */
//    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
//
//    EXIT:
//    h5status = H5Dclose(dset);
//    return ierr;
//}
//
double * unpack_tril(size_t n, size_t ntril, double *tril, double *out)
{
    if (out == NULL) {
        out = malloc(n*n * sizeof(double));
    }
    if (out == NULL) {
        return NULL;
    }

    for (size_t i = 0, idx = 0; i < n; i++) {
        for (size_t j = 0; j <= i; j++, idx++) {
            out[i*n + j] = tril[idx];
            //out[j*n + i] = tril[idx];
        }
    }
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i+1; j < n; j++) {
            out[i*n + j] = out[j*n + i];
            //out[j*n + i] = out[i*n + j];
        }
    }
    return out;
}

int64_t df_rhf_jk(
        /* In */
        int64_t nao,        // Number of AOs
        int64_t nocc,       // Number of occupied MOs
        int64_t naux,       // Number of DF functions
        char *cderi,        // File with 3c-integrals
        double *mo_coeff,   // (nao, nocc) occupied MO coefficients
        double *dm1,        // (nao, nao) 1DM
        int64_t blksize,
        /* Out */
        double *j,          // (nao, nao) Coulomb matrix
        double *k           // (nao, nao) exchange matrix
        )
{
    int64_t ierr = 0;

    const int64_t naopair = nao * (nao+1) / 2;
    printf("naopair= %ld\n", naopair);
    hsize_t stride[2] = {1, 1};
    hsize_t block[2] = {1, 1};

    hsize_t offset[2] = {-1, 0};
    hsize_t count[2] = {-1, naopair};

    printf("starting...\n");
    printf("filename= %s\n", cderi);

    hid_t file, gj3c, gkpt, dset_id;        /* identifier */
    hid_t mem_id, dspace_id;
    herr_t status;

    file = H5Fopen(cderi, H5F_ACC_RDONLY, H5P_DEFAULT);

    gj3c = H5Gopen(file, "j3c", H5P_DEFAULT);
    gkpt = H5Gopen(gj3c, "0", H5P_DEFAULT);
    dset_id = H5Dopen(gkpt, "0", H5P_DEFAULT);
    dspace_id = H5Dget_space(dset_id);

    size_t nblks;
    if (blksize == 1) {
        nblks = naux;
    } else {
        nblks = (int) (((double) naux + 1) / (double) blksize);
    }
    printf("n(Aux)= %ld  blksize= %ld  nblks= %ld\n", naux, blksize, nblks);

    double *work1 = malloc(blksize*naopair * sizeof(double));
    double *work2 = malloc(nao*nao * sizeof(double));
    double *work3 = malloc(nao * sizeof(double));

    for (size_t iblk = 0, l0 = 0; l0 < naux ; iblk++, l0 += blksize) {
        size_t l1 = MIN(l0 + blksize, naux);

        /* Read block from HDF5 file */
        printf("block %3ld : l0= %4ld l1= %4ld\n", iblk, l0, l1);
        offset[0] = l0;
        count[0] = l1-l0;
        mem_id = H5Screate_simple(2, count, NULL);
        status = H5Sselect_hyperslab(dspace_id, H5S_SELECT_SET, offset, stride, count, block);
        status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, mem_id, dspace_id, H5P_DEFAULT, work1);
        //for (size_t e = 0; e < (l1-l0); e++) {
        //    printf("%.16e\n", work1[e]);
        //}

        for (size_t lidx = 0, l = l0; l < l1 ; lidx++, l++) {

            memset(work2, 0, nao*nao * sizeof(double));
            unpack_tril(nao, naopair, &(work1[lidx*naopair]), work2);

            //printf("l= %ld\n", l);
            //for (size_t e = 0; e < (nao*nao); e++) {
            //    printf("%.16e\n", work2[e]);
            //}

            /* J */
            if (j != NULL) {
                double lab = 0;
                for (size_t a = 0; a < nao ; a++) {
                    for (size_t b = 0; b < nao ; b++) {
                        lab += (dm1[a*nao+b] * work2[a*nao+b]);
                    }
                }
                for (size_t a = 0; a < nao ; a++) {
                    for (size_t b = 0; b < nao ; b++) {
                        j[a*nao+b] += lab * work2[a*nao+b];
                    }
                }
            }

            /* K */
            //memset(work2, 0, nao*nao * sizeof(double));
            if (k != NULL) {
                for (size_t i = 0; i < nocc ; i++) {
                    memset(work3, 0, nao * sizeof(double));
                    for (size_t a = 0; a < nao ; a++) {
                        for (size_t b = 0; b < nao ; b++) {
                            work3[a] += work2[a*nao+b] * mo_coeff[b*nocc+i];
                        }
                    }
                    for (size_t a = 0; a < nao ; a++) {
                        for (size_t b = 0; b < nao ; b++) {
                            k[a*nao+b] += 2 * work3[a] * work3[b];
                        }
                    }
                }
            }
        }

        H5Sselect_none(dspace_id);
        status = H5Sclose(mem_id);
    }

    free(work1);
    free(work2);
    free(work3);

    //double *data = NULL;
    //load_dataset_double(gkpt, "0", data);
    //for (int i = 0; i < 5; i++) {
    //    printf("%f\n", data[i]);
    //}
    //free(data);

    //status = H5Sclose(mem_id);
    status = H5Sclose(dspace_id);
    status = H5Dclose(dset_id);
    status = H5Gclose(gkpt);
    status = H5Gclose(gj3c);
    status = H5Fclose(file);

    printf("done!\n");
    return ierr;
}
