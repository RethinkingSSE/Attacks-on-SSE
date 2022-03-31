#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

void read_matrix(int** index, int** matrix, double scaling, int N_kw, char* input_fileName)
{
    FILE *fp = fopen(input_fileName, "r");
    fscanf(fp, "%*[^\n]\n");

    for (int ii = 0; ii < N_kw; ii++)
        fscanf(fp, "%d,", &((*index)[ii]));
    fscanf(fp, "%*[^\n]\n");

    int tmp;
    for (int ii = 0; ii < N_kw; ii++)
    {
        for (int jj = 0; jj < N_kw; jj++)
        {
            fscanf(fp, "%d,", &tmp);
            (*matrix)[ii*N_kw + jj] = (int) (scaling * tmp);
        }
        fscanf(fp, "%*[^\n]\n");
    }
    fclose(fp);
}


void pad_matrix(int** matrix_padded, int** matrix, int N_kw, int N_doc, int freq_max)
{
    // Initialising RNG
    const gsl_rng_type * T;
    gsl_rng * r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);

    // perform padding on the keywords
    int ii, jj;
    #pragma omp parallel for private(ii)
    for (int ii = 0; ii < N_kw; ii++)
        (*matrix_padded)[ii*N_kw + ii] = 2*freq_max;

    // perform padding
    #pragma omp parallel for private(ii, jj)
    for (ii = 0; ii < N_kw; ii++)
    {
        for (jj = 0; jj < N_kw; jj++)
        {
            if (ii > jj)
            {
                int n1 = 2*freq_max - (*matrix)[ii*N_kw + ii];
                int n2 = 2*freq_max - (*matrix)[jj*N_kw + jj];
                int x1 = gsl_ran_hypergeometric(r, n1, 2*N_doc-n1, (*matrix_padded)[jj*N_kw + jj]);
                int x2 = gsl_ran_hypergeometric(r, n2, 2*N_doc-n2, (*matrix_padded)[ii*N_kw + ii]);

                (*matrix_padded)[ii*N_kw + jj] = (*matrix)[ii*N_kw + jj] + x1 + x2;
                (*matrix_padded)[jj*N_kw + ii] = (*matrix_padded)[ii*N_kw + jj];
            }
        }
    }
    gsl_rng_free(r);
}
    


void observe_matrix(gsl_matrix* matrix_obs, int** matrix_padded, int N_kw)
{
    // perform observed count generation
    for (int ii = 0; ii < N_kw; ii++)
        for (int jj = 0; jj < N_kw; jj++)
            gsl_matrix_set(matrix_obs, ii, jj, (double) ((*matrix_padded)[ii*N_kw + jj]));
}



void permutation_generation(int* idx1, int* idx2, int** permutation_tmp, int** permutation, int** permutation_inv, int N_kw, int N_obs)
{
    *idx1 = rand() % N_obs;
    *idx2 = -1;
    int idx_old = (*permutation)[*idx1];
    int idx_new = rand() % N_kw;

    (*permutation_tmp)[*idx1] = idx_new;

    if ((*permutation_inv)[idx_new] >= 0)
    {
        *idx2 = (*permutation_inv)[idx_new];
        (*permutation_tmp)[*idx2] = idx_old;
    }
}