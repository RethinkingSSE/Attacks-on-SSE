#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_blas.h>

#include <omp.h>

#include "util.h"



int main(int argc, char *argv[]) {
    char* input_fileName1 = argv[1];
    char* input_fileName2 = argv[2];
    int N_doc_bg = atoi(argv[3]);
    int N_kw = atoi(argv[4]);
    int N_obs = atoi(argv[5]);
    int N_iter = atoi(argv[6]);
    double lambda = atof(argv[7]);
    char* output_fileName = argv[8];
    int N_doc = 480000/2;

    int* matrix = (int*) malloc(sizeof(int) * N_obs * N_obs);
    int* matrix_bgi = (int*) malloc(sizeof(int) * N_kw * N_kw);
    int* matrix_padded = (int*) malloc(sizeof(int) * N_obs * N_obs);
    int* true_index = (int*) malloc(sizeof(int) * N_kw);
    int* permutation = (int*) malloc(sizeof(int) * N_obs);
    gsl_matrix* matrix_obs;


    for (int round = 0; round < 10; round++)
    {
        char input_fileName1_extend[40];
        char input_fileName2_extend[40];
        sprintf(input_fileName1_extend, "%s%d", input_fileName1, round);
        sprintf(input_fileName2_extend, "%s%d", input_fileName2, round);

        // Setup
        struct timeval tv1,tv2;
        gettimeofday(&tv1, NULL);
        read_matrix(&true_index, &matrix_bgi, 1.0*N_doc/N_doc_bg, N_kw, input_fileName2_extend);
        read_matrix(&true_index, &matrix, 1.0, N_obs, input_fileName1_extend);
        gettimeofday(&tv2, NULL);
        printf("Reading done: %f.\n", (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec));
        fflush(stdout);

        for (int iter = 0; iter < 10; iter++)
        {
            printf("Run %d\n", iter);
            matrix_obs = gsl_matrix_alloc(N_obs, N_obs);

            gettimeofday(&tv1, NULL);
            pad_matrix(&matrix_padded, &matrix, lambda, N_obs, N_doc);
            gettimeofday(&tv2, NULL);
            printf("Padding done: %f.\n", (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec));
            fflush(stdout);

            gettimeofday(&tv1, NULL);
            observe_matrix(matrix_obs, &matrix_padded, N_obs);
            gettimeofday(&tv2, NULL);
            printf("Observed matrix generated: %f.\n", (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec));
            fflush(stdout);

            
            gettimeofday(&tv1, NULL);
            attack(matrix_obs, &matrix_bgi, &permutation, N_kw, N_obs, N_doc, N_iter);
            gettimeofday(&tv2, NULL);
            printf("Main attack done: %d.\n",  (tv2.tv_sec - tv1.tv_sec));
            fflush(stdout);

            char output_fileName_full[40];
            sprintf(output_fileName_full, "%s%d-%d", output_fileName, round, iter);
            print_result(output_fileName_full, &permutation, &true_index, N_obs);
            //sprintf(output_fileName_full, "%s%d-%d-full", output_fileName, round, iter);
            //print_full_result(output_fileName_full, &permutation, &true_index, N_obs);
        }
    }

    free(matrix);
    free(matrix_padded);
    gsl_matrix_free(matrix_obs);
    return(0);
}


double log_score(int idx1, int idx2, gsl_matrix* matrix_obs, int** matrix, int** permutation, int N_kw, int N_doc)
{
    if (idx1 == idx2)
        return(0.0);

    int idx1_m = (*permutation)[idx1];
    int idx2_m = (*permutation)[idx2];

    double mean1, mean2, mean3;
    mean3 = 0;
    double var = 0;

    if (gsl_matrix_get(matrix_obs, idx1, idx1) > (*matrix)[idx1_m*N_kw + idx1_m])
    {
        mean1 = (*matrix)[idx1_m*N_kw + idx2_m];
        double x1, x2;
        x1 = gsl_matrix_get(matrix_obs, idx1, idx1) - (*matrix)[idx1_m*N_kw + idx1_m];
        x2 = gsl_matrix_get(matrix_obs, idx2, idx2) - (*matrix)[idx1_m*N_kw + idx2_m];
        mean3 = x1 / N_doc * x2;
        var += x1 / N_doc * x2 * (N_doc - x2) / N_doc;
    }
    else
    {
        mean1 = (*matrix)[idx1_m*N_kw + idx2_m] * gsl_matrix_get(matrix_obs, idx1, idx1) / (*matrix)[idx1_m*N_kw + idx1_m];
        var   += mean1 / (*matrix)[idx1_m*N_kw + idx1_m] * ((*matrix)[idx1_m*N_kw + idx1_m] - gsl_matrix_get(matrix_obs, idx1, idx1));
    }


    if (gsl_matrix_get(matrix_obs, idx2, idx2) > (*matrix)[idx2_m*N_kw + idx2_m])
    {
        mean2 = mean1;
        double x1, x2;
        x1 = gsl_matrix_get(matrix_obs, idx2, idx2) - (*matrix)[idx2_m*N_kw + idx2_m];
        x2 = gsl_matrix_get(matrix_obs, idx1, idx1) - (*matrix)[idx1_m*N_kw + idx2_m];
        mean3 += x1 / N_doc * x2;
        var += x1 / N_doc * x2 * (N_doc - x2) / N_doc;
    }
    else
    {
        mean2 = mean1 * gsl_matrix_get(matrix_obs, idx2, idx2) / (*matrix)[idx2_m*N_kw + idx2_m];
        var  += mean2 / (*matrix)[idx2_m*N_kw + idx2_m] * ((*matrix)[idx2_m*N_kw + idx2_m] - gsl_matrix_get(matrix_obs, idx2, idx2));
    }

    var += 1.0 * (*matrix)[idx1_m*N_kw + idx2_m] / N_doc * (N_doc - (*matrix)[idx2_m*N_kw + idx2_m]);

    double score = gsl_ran_gaussian_pdf(mean2 + mean3 - gsl_matrix_get(matrix_obs, idx1, idx2), sqrt(var));
    
    if (score == 0)
        return(-500.0);
    return(log(score));
}





void attack(gsl_matrix* matrix_obs, int** matrix, int** permutation, int N_kw, int N_obs, int N_doc, int N_iter)
{
    // Initialise data structures
    double* score_matrix = (double*) malloc(sizeof(double) * N_obs * N_obs);
    double* score_row1 = (double*) malloc(sizeof(double) * N_obs);
    double* score_row2 = (double*) malloc(sizeof(double) * N_obs);

    int* permutation_tmp = (int*) malloc(sizeof(int) * N_obs);
    int* permutation_inv = (int*) malloc(sizeof(int) * N_kw);


    // Initialise permutations
    for (int ii = 0; ii < N_obs; ii++)
        (*permutation)[ii] = ii;
    for (int ii = 0; ii < N_obs; ii++)
        permutation_tmp[ii] = ii;
    for (int ii = 0; ii < N_obs; ii++)
        permutation_inv[ii] = ii;
    for (int ii = N_obs; ii < N_kw; ii++)
        permutation_inv[ii] = -1;

    // Initialising RNG
    const gsl_rng_type * T;
    gsl_rng * r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);

    struct timeval tv1,tv2;
    gettimeofday(&tv1, NULL);

    // Compute initial score
    #pragma omp parallel for shared(score_matrix, matrix_obs, matrix)
    for (int ii = 0; ii < N_obs * N_obs; ii++)
        score_matrix[ii] = log_score((int) (ii / N_obs), ii % N_obs, matrix_obs, matrix, permutation, N_kw, N_doc);
    gettimeofday(&tv2, NULL);
    printf("Initial score computed: %f.\n", (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec));
    
    // Iterations of simulated annealing
    double temp = (double) N_kw;
    int N_stuck = 0;
    for (int iter = 0; iter < N_iter; iter++)
    {
	    /* Status code */
	    if (iter % (N_iter / 10) == 0)
        {
            gettimeofday(&tv1, NULL);
            printf("Iteration: %d, %d, %d.\n", iter, N_stuck, (int) (tv1.tv_sec - tv2.tv_sec));
            fflush(stdout);
            gettimeofday(&tv2, NULL);
        }

        if (N_stuck >= 20000)
            iter = N_iter;

	    /* Main code */
        int idx1, idx2;
        permutation_generation(&idx1, &idx2, &permutation_tmp, permutation, &permutation_inv, N_kw, N_obs);

        int ii = 0;
        #pragma omp parallel for shared(score_row1)
        for (ii = 0; ii < N_obs; ii++)
            score_row1[ii] = log_score(idx1, ii, matrix_obs, matrix, &permutation_tmp, N_kw, N_doc);

        if (idx2 >= 0)
            #pragma omp parallel for shared(score_row2)
            for (ii = 0; ii < N_obs; ii++)
                score_row2[ii] = log_score(idx2, ii, matrix_obs, matrix, &permutation_tmp, N_kw, N_doc);

        double score_diff = 0;
        for (int ii = 0; ii < N_obs; ii++)
            score_diff += score_row1[ii];
        for (int ii = 0; ii < N_obs; ii++)
            score_diff -= score_matrix[idx1*N_obs + ii];
        if (idx2 >= 0)
        {
            for (int ii = 0; ii < N_obs; ii++)
                score_diff += score_row2[ii];
            for (int ii = 0; ii < N_obs; ii++)
                score_diff -= score_matrix[idx2*N_obs + ii];
        }

        // compute difference in score, with exponentiation
        score_diff = score_diff / temp;
        
        if (score_diff < -40)
            score_diff = 0;
        else if (score_diff > 0)
            score_diff = 1.01;
        else
            score_diff = exp(score_diff);

        if (gsl_ran_flat(r, 0, 1) < score_diff)
        {
            // Update the scores
            for (int ii = 0; ii < N_obs; ii++)
                score_matrix[idx1*N_obs + ii] = score_row1[ii];
            for (int ii = 0; ii < N_obs; ii++)
                score_matrix[ii*N_obs + idx1] = score_row1[ii];
            if (idx2 >= 0)
            {
                for (int ii = 0; ii < N_obs; ii++)
                    score_matrix[idx2*N_obs + ii] = score_row2[ii];
                for (int ii = 0; ii < N_obs; ii++)
                    score_matrix[ii*N_obs + idx2] = score_row2[ii];
            }

            // Update the permutation
            permutation_inv[(*permutation)[idx1]] = -1;
            (*permutation)[idx1] = permutation_tmp[idx1];
            permutation_inv[permutation_tmp[idx1]] = idx1;
            if (idx2 >= 0)
            {
                (*permutation)[idx2] = permutation_tmp[idx2];
                permutation_inv[permutation_tmp[idx2]] = idx2;
            }
            N_stuck = 0;
        }
        else
        {
            // Update the permutation
            permutation_tmp[idx1] = (*permutation)[idx1];
            if (idx2 >= 0)
                permutation_tmp[idx2] = (*permutation)[idx2];
            N_stuck += 1;
        }

        temp *= 0.995;
    }

    free(score_matrix);
    free(score_row1);
    free(score_row2);
    gsl_rng_free(r);
}



void print_result(char* output_fileName, int** permutation, int** true_index, int N_obs)
{
    FILE* fp = fopen(output_fileName, "w");
    int count = 0;
    for (int ii = 0; ii < N_obs; ii++)
        if ((*permutation)[ii] == (*true_index)[ii])
            count++;

    fprintf(fp, "%d\n", count);
    fclose(fp);
    printf("Success: %d/%d.\n", count, N_obs);
}



void print_full_result(char* output_fileName, int** permutation, int** true_index, int N_obs)
{
    FILE* fp = fopen(output_fileName, "w");
    int count = 0;
    for (int ii = 0; ii < N_obs; ii++)
        fprintf(fp, "%d, %d\n", (*permutation)[ii], (*true_index)[ii]);
    fclose(fp);
}