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


void pad_matrix(int** matrix_padded, int** matrix, int m, double p, double q, int N_kw, int N_doc)
{
    // Initialising RNG
    const gsl_rng_type * T;
    gsl_rng * r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);

    // perform padding
    int ii, jj;
    #pragma omp parallel for private(ii, jj)
    for (ii = 0; ii < N_kw; ii++)
    {
        for (jj = 0; jj < N_kw; jj++)
        {
            if (ii == jj)
                (*matrix_padded)[ii*N_kw + jj] = gsl_ran_binomial(r, p, m*(*matrix)[ii*N_kw + jj]) + gsl_ran_binomial(r, q, m*(N_doc - (*matrix)[ii*N_kw + jj]));


            if (ii > jj)
            {
                int N1 = (*matrix)[ii*N_kw + jj];
                int N2 = (*matrix)[ii*N_kw + ii] + (*matrix)[jj*N_kw + jj] - 2*(*matrix)[ii*N_kw + jj];
                int N3 = N_doc - N1 - N2;

                int n1 = gsl_ran_binomial(r, p*p, m*N1);
                int n2 = gsl_ran_binomial(r, p*q, m*N2);
                int n3 = gsl_ran_binomial(r, q*q, m*N3);

                (*matrix_padded)[ii*N_kw + jj] = n1 + n2 + n3;
                (*matrix_padded)[jj*N_kw + ii] = n1 + n2 + n3;
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




void solution_initial(int** permutation, gsl_matrix* matrix_obs, int** matrix, int m, double p, double q, int N_kw, int N_obs, int N_doc)
{
    int* diff = (int*) malloc(sizeof(int) * N_kw);
    int* index = (int*) malloc(sizeof(int) * N_kw);
    int* occupied = (int*) malloc(sizeof(int) * N_kw);

    for (int ii = 0; ii < N_kw; ii++)
        occupied[ii] = -1;


    for (int ii = 0; ii < N_obs; ii++)
    {
        // reset index
        for (int jj = 0; jj < N_kw; jj++)
            index[jj] = jj;

        // compute differences
        for (int jj = 0; jj < N_kw; jj++)
            diff[jj] = abs(gsl_matrix_get(matrix_obs, ii, ii) - m * p * (*matrix)[jj*N_kw+jj] - m * q * (N_doc - (*matrix)[jj*N_kw+jj]));

        // sorting
        for (int jj = 0; jj < N_kw-1; jj++)
        {
            for (int kk = 0; kk < N_kw-jj-1; kk++)
            {
                if (diff[kk] > diff[kk+1])
                {
                    int temp = diff[kk+1];
                    diff[kk+1] = diff[kk];
                    diff[kk] = temp;

                    temp = index[kk+1];
                    index[kk+1] = index[kk];
                    index[kk] = temp;
                }
            }
        }

        // use the smallest index available
        int idx = 0;
        while (occupied[index[idx]] > 0)
            idx++;

        // update the data structures
        (*permutation)[ii] = index[idx];
        occupied[index[idx]] = 1;
    }

    //for (int ii = 200; ii < 250; ii++)
    //{
    //    printf("%f\n", gsl_matrix_get(matrix_obs, ii, ii) - m * p * (*matrix)[(*permutation)[ii]*N_kw+(*permutation)[ii]] - m * q * (N_doc - (*matrix)[(*permutation)[ii]*N_kw+(*permutation)[ii]]));
    //}
}





void permutation_generation(int* idx1, int* idx2, int** permutation_tmp, int** permutation, int** permutation_inv, gsl_matrix* matrix_obs, int** matrix, int m, double p, double q, int N_kw, int N_obs, int N_doc)
{
    double N1_mean, N1_var, N1_lower, N1_upper, N2_mean, N2_var, N2_lower, N2_upper;
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

            N1_mean = m * p * (*matrix)[idx_new*N_kw + idx_new] + m * q * (N_doc - (*matrix)[idx_new*N_kw + idx_new]);
            N1_var = (*matrix)[idx_new*N_kw + idx_new] / (double) N_doc * (N_doc - (*matrix)[idx_new*N_kw + idx_new]) * 2.0 * m;
            N1_var += m * p * (1-p) * (*matrix)[idx_new*N_kw + idx_new] + m * q * (1-q) * (N_doc - (*matrix)[idx_new*N_kw + idx_new]);
            N1_var =  3*sqrt(N1_var);

            N1_lower = N1_mean - N1_var;
            N1_upper = N1_mean + N1_var;

            N2_mean = m * p * (*matrix)[idx_old*N_kw + idx_old] + m * q * (N_doc - (*matrix)[idx_old*N_kw + idx_old]);
            N2_var = (*matrix)[idx_old*N_kw + idx_old] / (double) N_doc * (N_doc - (*matrix)[idx_old*N_kw + idx_old]) * 2.0 * m ;
            N2_var += m * p * (1-p) * (*matrix)[idx_old*N_kw + idx_old] + m * q * (1-q)* (N_doc - (*matrix)[idx_old*N_kw + idx_old]);
            N2_var =  3*sqrt(N2_var);

            N2_lower = N2_mean - N2_var;
            N2_upper = N2_mean + N2_var;


            if ((gsl_matrix_get(matrix_obs, *idx1, *idx1) > N1_lower) && (gsl_matrix_get(matrix_obs, *idx1, *idx1) < N1_upper))
                if ((gsl_matrix_get(matrix_obs, *idx2, *idx2) > N2_lower) && (gsl_matrix_get(matrix_obs, *idx2, *idx2) < N2_upper))
                    check = 1;
        }
        else
        {
	    *idx2 = (*permutation_inv)[idx_new];

            N1_mean = m * p * (*matrix)[idx_new*N_kw + idx_new] + m * q * (N_doc - (*matrix)[idx_new*N_kw + idx_new]);
            N1_var = (*matrix)[idx_new*N_kw + idx_new] / (double) N_doc * (N_doc - (*matrix)[idx_new*N_kw + idx_new]) * 2.0 * m;
            N1_var += m * p * (1-p) * (*matrix)[idx_new*N_kw + idx_new] + m * q * (1-q) * (N_doc - (*matrix)[idx_new*N_kw + idx_new]);
            N1_var =  3*sqrt(N1_var);

            N1_lower = N1_mean - N1_var;
            N1_upper = N1_mean + N1_var;

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