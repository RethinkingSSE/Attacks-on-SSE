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


void pad_matrix(int** matrix_padded, int** matrix, int N_kw, int N_doc, double sec_budget, int fixed_padding)
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
        (*matrix_padded)[ii*N_kw + ii] = 2 * (*matrix)[ii*N_kw + ii] + fixed_padding + (int) gsl_ran_laplace(r, 2.0/sec_budget);

    // perform padding
    #pragma omp parallel for private(ii, jj)
    for (ii = 0; ii < N_kw; ii++)
    {
        for (jj = 0; jj < N_kw; jj++)
        {
            if (ii > jj)
            {
                int n1 = (*matrix_padded)[ii*N_kw + ii] - (*matrix)[ii*N_kw + ii];
                int n2 = (*matrix_padded)[jj*N_kw + jj] - (*matrix)[jj*N_kw + jj];
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


void permutation_generation(int* idx1, int* idx2, int** permutation_tmp, int** permutation, int** permutation_inv, gsl_matrix* matrix_obs, int** matrix, int fixed_padding, int N_kw, int N_obs, int N_doc, double sec_budget)
{
    double N1_lower, N1_upper, N2_lower, N2_upper;
    int check = -1;
    int count = 0;
    *idx1 = rand() % N_obs;
    *idx2 = -1;
    int idx_old = (*permutation)[*idx1];
    int idx_new = 0;

    do {
        idx_new = rand() % N_kw;
        count++;

        if ((*permutation_inv)[idx_new] >= 0)
        {
            *idx2 = (*permutation_inv)[idx_new];
            N1_lower = 2 * (*matrix)[idx_new*N_kw + idx_new] - 4 * sqrt((*matrix)[idx_new*N_kw + idx_new] * (N_doc - (*matrix)[idx_new*N_kw + idx_new]) / N_doc) - 4 / sec_budget + fixed_padding;
            N1_upper = 2 * (*matrix)[idx_new*N_kw + idx_new] + 4 * sqrt((*matrix)[idx_new*N_kw + idx_new] * (N_doc - (*matrix)[idx_new*N_kw + idx_new]) / N_doc) + 4 / sec_budget + fixed_padding;

            N2_lower = 2 * (*matrix)[idx_old*N_kw + idx_old] - 4 * sqrt((*matrix)[idx_old*N_kw + idx_old] * (N_doc - (*matrix)[idx_old*N_kw + idx_old]) / N_doc) - 4 / sec_budget + fixed_padding;
            N2_upper = 2 * (*matrix)[idx_old*N_kw + idx_old] + 4 * sqrt((*matrix)[idx_old*N_kw + idx_old] * (N_doc - (*matrix)[idx_old*N_kw + idx_old]) / N_doc) + 4 / sec_budget + fixed_padding;

            if ((gsl_matrix_get(matrix_obs, *idx1, *idx1) > N1_lower) && (gsl_matrix_get(matrix_obs, *idx1, *idx1) < N1_upper))
                if ((gsl_matrix_get(matrix_obs, *idx2, *idx2) > N2_lower) && (gsl_matrix_get(matrix_obs, *idx2, *idx2) > N2_lower))
                    check = 1;
        }
        else
        {
            *idx2 = -1;
            N1_lower = 2 * (*matrix)[idx_new*N_kw + idx_new] - 4 * sqrt((*matrix)[idx_new*N_kw + idx_new] * (N_doc - (*matrix)[idx_new*N_kw + idx_new]) / N_doc) - 4 / sec_budget + fixed_padding;
            N1_upper = 2 * (*matrix)[idx_new*N_kw + idx_new] + 4 * sqrt((*matrix)[idx_new*N_kw + idx_new] * (N_doc - (*matrix)[idx_new*N_kw + idx_new]) / N_doc) + 4 / sec_budget + fixed_padding;

            if ((gsl_matrix_get(matrix_obs, *idx1, *idx1) > N1_lower) && (gsl_matrix_get(matrix_obs, *idx1, *idx1) < N1_upper))
                check = 1;

        }
        
    } while ((check < 0) && (count < 200));

    if (count == 200)
        *idx2 = *idx1;

    if (*idx1 != *idx2)
    {
        (*permutation_tmp)[*idx1] = idx_new;
        if ((*permutation_inv)[idx_new] >= 0)
        {
            *idx2 = (*permutation_inv)[idx_new];
            (*permutation_tmp)[*idx2] = idx_old;
        }
    }
}