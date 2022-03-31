void read_matrix(int** index, int** matrix, double scaling, int N_kw, char* input_fileName);
void pad_matrix(int** matrix_padded, int** matrix, int N_kw, int N_doc, double sec_budget, int fixed_padding);
void observe_matrix(gsl_matrix* matrix_obs, int** matrix_padded, int N_kw);

void initial_solution(int** permutation, gsl_matrix* matrix_obs, int** matrix, int fixed_padding, int N_kw, int N_obs);
void permutation_generation(int* idx1, int* idx2, int** permutation_tmp, int** permutation, int** permutation_inv, gsl_matrix* matrix_obs, int** matrix, int fixed_padding, int N_kw, int N_obs, int N_doc, double sec_budget);

double log_score(int idx1, int idx2, gsl_matrix* matrix_obs, int** matrix, int** permutation, int N_kw, int N_doc, double sec_budget, int fixed_padding);
void attack(gsl_matrix* matrix_obs, int** matrix, int** permutation, int N_kw, int N_obs, int N_doc, double sec_budget, int fixed_padding, int N_iter);

void print_result(char* output_fileName, int** permutation, int** true_index, int N_obs);
void print_full_result(char* output_fileName, int** permutation, int** true_index, int N_obs);
