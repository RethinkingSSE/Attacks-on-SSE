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
            (*matrix)[ii*N_kw + jj] = (int) (tmp * scaling);
        }
        fscanf(fp, "%*[^\n]\n");
    }
    fclose(fp);
}


void pad_matrix(int** matrix_padded, int** matrix, double lambda, int N_kw, int N_doc)
{
    // Initialising RNG
    const gsl_rng_type * T;
    gsl_rng *r1, *r2;
    gsl_rng_env_setup();
    r1 = gsl_rng_alloc(gsl_rng_default);
    r2 = gsl_rng_alloc (gsl_rng_taus);

    // perform padding
    int ii, jj;

    int max_resp_len = 0;
    for (ii = 0; ii < N_kw; ii++)
        if ((*matrix)[ii*N_kw + ii] > max_resp_len)
            max_resp_len = (*matrix)[ii*N_kw + ii];

    #pragma omp parallel for private(ii)
    for (ii = 0; ii < N_kw; ii++)
        (*matrix_padded)[ii*N_kw + ii] = (int) (lambda * (max_resp_len)) + gsl_rng_uniform_int(r2, max_resp_len+1);

    #pragma omp parallel for private(ii, jj)
    for (ii = 0; ii < N_kw; ii++)
    {
        for (jj = 0; jj < N_kw; jj++)
        {
            if (ii > jj)
            {
                int n1, n2, n3;
                n3 = 0;
                if ((*matrix_padded)[ii*N_kw + ii] > (*matrix)[ii*N_kw + ii])
                {
                    n1 = (*matrix)[ii*N_kw + jj];
                    n3 += gsl_ran_hypergeometric(r1, (*matrix_padded)[ii*N_kw + ii] - (*matrix)[ii*N_kw + ii], N_doc, (*matrix_padded)[jj*N_kw + jj]);
                }
                else
                    n1 = gsl_ran_hypergeometric(r1, (*matrix)[ii*N_kw + jj], (*matrix)[ii*N_kw + ii] - (*matrix)[ii*N_kw + jj], (*matrix_padded)[ii*N_kw + ii] - (*matrix)[ii*N_kw + jj]);
                
                if ((*matrix_padded)[jj*N_kw + jj] > (*matrix)[jj*N_kw + jj])
                {
                    n2 = n1;
                    n3 += gsl_ran_hypergeometric(r1, (*matrix_padded)[jj*N_kw + jj] - (*matrix)[jj*N_kw + jj], N_doc, (*matrix_padded)[ii*N_kw + ii] - (*matrix)[ii*N_kw + jj]);
                }
                else
                    n2 = gsl_ran_hypergeometric(r1, n1, (*matrix)[jj*N_kw + jj] - n1, (*matrix_padded)[jj*N_kw + jj]);

                (*matrix_padded)[ii*N_kw + jj] = n2 + n3;
                (*matrix_padded)[jj*N_kw + ii] = n2 + n3;
            }
        }
    }
    gsl_rng_free(r1);
    gsl_rng_free(r2);
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
    int count = 0;
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

