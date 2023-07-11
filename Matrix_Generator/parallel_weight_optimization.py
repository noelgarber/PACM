# Contains functions to optimize matrix weights in a parallelized manner

import numpy as np
import pandas as pd
import os
import time
import multiprocessing
from tqdm import trange
from functools import partial
from Matrix_Generator.ConditionalMatrix import ConditionalMatrices
from Matrix_Generator.conditional_scoring import apply_motif_scores
from general_utils.general_utils import save_dataframe, save_weighted_matrices, unravel_seqs
from general_utils.weights_utils import permute_weights
from general_utils.general_vars import aa_charac_dict
from general_utils.user_helper_functions import get_possible_weights
from general_utils.statistics import optimize_threshold_fdr, fdr_for_optimizer

def process_weights_chunk(chunk, conditional_matrices, source_df, slim_length, seq_col, significance_col, score_col,
                          significant_str = "Yes", allowed_rate_divergence = 0.2, convert_phospho = True):
    '''
    Lower helper function for parallelization of position weight optimization; processes chunks of permuted weights

    Note that matrix_arrays_dict, unlike matrices_dict which appears elsewhere, contains numpy arrays instead of
    pandas dataframes. This greatly improves performance, but requires the matching pd.Index to be passed separately.

    Args:
        chunk (np.ndarray):                         the chunk of permuted weights currently being processed
        conditional_matrices (ConditionalMatrices): unweighted conditional position-weighted matrices
        source_df (pd.DataFrame):                   contains source peptide-protein binding data
        slim_length (int):                          length of the motif being studied
        seq_col (str):                              col name where sequences are stored
        significance_col (str):                     col name where significance information is found
        score_col (str):                            col name where motif scores are found
        significant_str (str):                      string representing a pass in source_df[significance_col]
        allowed_rate_divergence (float):            max allowed difference between FDR and FOR pairs
        convert_phospho (bool):                     whether to convert phospho-residues to non-phospho before lookups

    Returns:
        results_tuple (tuple):  (chunk_best_fdr, chunk_best_for, chunk_best_score_threshold, chunk_best_weights,
                                 chunk_best_weighted_matrices_dict, chunk_best_source_df)
    '''

    output_df = source_df.copy()

    passes_bools = source_df[significance_col].values == significant_str

    # Convert sequences to a 2D array upfront for performance improvement
    sequences = source_df[seq_col].to_numpy()
    sequences_2d = unravel_seqs(sequences, slim_length, convert_phospho)

    # Initialize what will become a list of tuples of (best_score, best_fdr, best_for) matching the indices of chunk
    optimal_values = []

    for weights_array in chunk:
        # Apply the current weights_array to the dict of matrices with numpy; takes almost no time
        conditional_matrices.apply_weights(weights_array, only_3d = True)

        # Get the array of scores for peptide entries in source_df using the current set of weighted matrices
        scores_array = apply_motif_scores(output_df, slim_length, conditional_matrices, sequences_2d,
                                          convert_phospho = True, return_array = True, use_weighted = True)

        # Determine the optimal threshold score that gives balanced FDR/FOR values, which are inversely correlated
        score_range_series = np.linspace(scores_array.min(), scores_array.max(), num=500)
        best_results = optimize_threshold_fdr(None, score_range_series, passes_bools = passes_bools,
                                              scores_array = scores_array)
        current_best_score, current_best_fdr, current_best_for = best_results

        optimal_values.append((current_best_score, current_best_fdr, current_best_for))

    # Find the chunk index for the weights array that produces the lowest optimal FDR & FOR values (using the mean)
    optimal_values_array = np.array(optimal_values)
    best_index = fdr_for_optimizer(optimal_values_array[:,1:], allowed_rate_divergence)

    chunk_best_score_threshold, chunk_best_fdr, chunk_best_for = optimal_values_array[best_index]
    chunk_best_weights = chunk[best_index]

    # Get the matching dict of weighted matrices and use it to apply final scores to output_df
    conditional_matrices.apply_weights(chunk_best_weights, only_3d = False)
    chunk_best_source_df = apply_motif_scores(output_df, slim_length, conditional_matrices, sequences_2d, score_col,
                                              convert_phospho = convert_phospho, add_residue_cols = True,
                                              in_place = False, return_array = False)

    results_tuple = (chunk_best_fdr, chunk_best_for, chunk_best_score_threshold,
                     chunk_best_weights, conditional_matrices, chunk_best_source_df)

    return results_tuple

def process_weights(weights_array_chunks, conditional_matrices, slim_length, source_df, sequence_col, significance_col,
                    significant_str, score_col, allowed_rate_divergence = 0.2, convert_phospho = True):
    '''
    Upper helper function for parallelization of position weight optimization; processes weights by chunking

    Args:
        weights_array_chunks (list):                list of chunks as numpy arrays for feeding to process_weights_chunk
        conditional_matrices (ConditionalMatrices): unweighted conditional position-weighted matrices
        slim_length (int):                          length of the motif being studied
        source_df (pd.DataFrame):                   source peptide-protein binding data
        sequence_col (str):                         col name in chunk where sequences are stored
        significance_col (str):                     col name in chunk where significance information is found
        significant_str (str):                      value in significance_col that denotes significance, e.g. "Yes"
        score_col (str):                            col name where motif scores are found
        allowed_rate_divergence (float):            maximum allowed difference between FDR and FOR pairs
        convert_phospho (bool):                     whether to convert phospho-aa's to non-phospho before doing lookups

    Returns:
        results_list (list):     the list of results sets for all the weights arrays
    '''

    pool = multiprocessing.Pool()

    process_partial = partial(process_weights_chunk, conditional_matrices = conditional_matrices, source_df = source_df,
                              slim_length = slim_length, seq_col = sequence_col, significance_col = significance_col,
                              score_col = score_col, significant_str = significant_str,
                              allowed_rate_divergence = allowed_rate_divergence, convert_phospho = convert_phospho)

    results = None
    best_rate_mean = 9999

    with trange(len(weights_array_chunks), desc="Processing weights") as pbar:
        for chunk_results in pool.imap_unordered(process_partial, weights_array_chunks):
            rate_mean = (chunk_results[0] + chunk_results[1]) / 2
            rate_delta = abs(chunk_results[0] - chunk_results[1])
            if rate_mean < best_rate_mean and rate_delta <= allowed_rate_divergence:
                best_rate_mean = rate_mean
                results = chunk_results
                print(f"\tNew record: FDR={results[0]} | FOR={results[1]} | weights={results[3]}")

            pbar.update()

    pool.close()
    pool.join()

    return results

def find_optimal_weights(input_df, slim_length, conditional_matrices, sequence_col, significance_col, significant_str,
                         score_col, matrix_output_folder, output_folder, possible_weights = None,
                         dict_of_aa_characs = aa_charac_dict, convert_phospho = True, save_pickled_matrix_dict = True):
    '''
    Parent function for finding optimal position weights to generate optimally weighted matrices

    Args:
        input_df (pd.DataFrame):                     contains peptide sequences, signal values, and significances
        slim_length (int):                           length of the motif being studied
        conditional_matrices (ConditionalMatrices):  unweighted conditional matrices based on type-position rules
        sequence_col (str):                          col name where sequences are stored
        significance_col (str):                      col name where significance information is found
        significant_str (str):                       value in significance_col that denotes significance, e.g. "Yes"
        score_col (str):                             col name where motif scores are found
        possible_weights (list):                     list of numpy arrays with possible weights at each position
        dict_of_aa_characs (dict):                   dictionary of chemical_class --> [amino acid members]
        matrix_output_folder (str):                  path for saving weighted matrices
        output_folder (str):                         path for saving the scored data
        convert_phospho (bool):                      whether to convert phospho-residues to non-phospho before lookups
        save_pickled_matrix_dict (bool):             whether to save a pickled version of the dict of matrices

    Returns:
        results_tuple (tuple):  (best_fdr, best_for, best_score_threshold, best_weights, best_weighted_matrices_dict, best_dens_df)
    '''

    output_df = input_df.copy()

    # Get the permuted weights and break into chunks based on CPU count, for parallelization
    possible_weights = get_possible_weights(slim_length) if possible_weights is None else possible_weights
    weights_array = permute_weights(slim_length, possible_weights)
    weights_array_count = weights_array.shape[0]
    cpu_count = os.cpu_count()
    chunk_size = round(weights_array_count / cpu_count)
    if chunk_size > 10000:
        chunk_size = 10000
    elif chunk_size < 10:
        chunk_size = 10
    weights_array_chunks = [weights_array[i:i + chunk_size] for i in range(0, len(weights_array), chunk_size)]

    # Run the parallelized optimization process
    results = process_weights(weights_array_chunks, conditional_matrices, slim_length, output_df, sequence_col,
                              significance_col, significant_str, score_col, allowed_rate_divergence = 0.2,
                              convert_phospho = convert_phospho)
    best_fdr, best_for, best_score_threshold, best_weights, best_conditional_matrices, best_scored_df = results
    print("\t---\n", f"\tOptimal weights of {best_weights} gave FDR = {best_fdr} and FOR = {best_for} at a SLiM score threshold > {best_score_threshold}")

    # Save the weighted matrices and scored data
    save_weighted_matrices(conditional_matrices.matrices_dict, os.path.join(matrix_output_folder, "Unweighted"), save_pickled_matrix_dict)
    save_weighted_matrices(best_conditional_matrices.weighted_matrices_dict, os.path.join(matrix_output_folder, "Weighted"), save_pickled_matrix_dict)
    with open(os.path.join(matrix_output_folder, "final_weights.txt"), "w") as file:
        file.write(",".join(map(str, best_weights)))

    scored_data_filename = "pairwise_scored_data_thres" + str(best_score_threshold) + ".csv"
    save_dataframe(best_scored_df, output_folder, scored_data_filename)

    print(f"Saved weighted matrices and scored data to {output_folder}")

    return results