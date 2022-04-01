# Attacks-on-SSE
This repository is an open-source implementation of the attacks in "Rethinking Searchable Symmetric Encryption".

## Overview
The repository includes attacks against four schemes, namely:
1. **PRT-EMM**: An end-to-end SSE scheme built from KamMoa19 (https://link.springer.com/chapter/10.1007/978-3-030-17656-3_7).
2. **FP-EMM** and **DP-EMM**: End-to-end SSE schemes built from PPYY19 (https://dl.acm.org/doi/10.1145/3319535.3354213).
3. **DPAP-SE**: An end-to-end SSE scheme specified in CLRZ18 (https://ieeexplore.ieee.org/abstract/document/8486381).

## Running the attacks
The code uses GSL for scientific computations and OpenMP for parallelization.

The arguments for the attacks are as follows:
1. **PRT-EMM**: `<input_target> <input_auxiliary> <N_aux> <N_KW> <N_obs_KW> <N_iter> <lambda> <output_file>`
2. **FP-EMM**:  `<input_target> <input_auxiliary> <N_aux> <N_KW> <N_obs_KW> <N_iter> <output_file>`
3. **DP-EMM**:  `<input_target> <input_auxiliary> <N_aux> <N_KW> <N_obs_KW> <N_iter> <sec_budget> <output_file>`
4. **DPAP-SE**: `<input_target> <input_auxiliary> <N_aux> <N_KW> <N_obs_KW> <N_iter> <m> <p> <q> <output_file>`
where `N_aux` and `N_KW` are the sizes of the input co-occurrence matrices; `N_iter` is the maximum number of iterations for the attack to run before aborting (and returning the best assignment found so far); `lambda`, `sec_budget`, `m`, `p` and `q` are security parameters of the respective schemes.
The structure of the input and output files are explained below.

To change how the attack loops with respect to the inputs (e.g. every attack uses freshly generated inputs), one can simply change the `main()` function.

## Structure of the input and output files
The structure of the input files are as follows.
- First line: List of keywords.
- Second line: Indices of the keywords. The indices for the auxiliary input are 0 to `N_aux-1`. The indices for the target input is shuffled with respect to the indices of the auxiliary input.
- Other lines: Lines of the co-occurrence matrices. Counts are used for the auxiliary input and normalised in the code itself (with N_aux) to reduce file size and improve precision.

The structure of the output files are as follows. Every line represents the number of correct guesses of assignmnents in an attack.
