# Contains functions to optimize matrix weights in a parallelized manner

import numpy as np
import pandas as pd
import multiprocessing
from copy import deepcopy
from tqdm import trange
from functools import partial
from Matrix_Generator.ConditionalMatrix import ConditionalMatrices
from Matrix_Generator.conditional_scoring import apply_motif_scores
from general_utils.general_utils import unravel_seqs
from general_utils.weights_utils import permute_weights

def process_weights_chunk(chunk, conditional_matrices, sequences_2d, passes_bools, source_df, motif_length):
    '''
    Lower helper function for parallelization of position weight optimization; processes chunks of permuted weights

    Note that matrix_arrays_dict, unlike matrices_dict which appears elsewhere, contains numpy arrays instead of
    pandas dataframes. This greatly improves performance, but requires the matching pd.Index to be passed separately.

    Args:
        chunk (np.ndarray):                         the chunk of permuted weights currently being processed
        conditional_matrices (ConditionalMatrices): unweighted conditional position-weighted matrices
        sequences_2d (np.ndarray):                  2D array of peptide sequences, where each row is a peptide as array
        passes_bools (np.ndarray):                  array of bools about whether the peptides are positive or not
        source_df (pd.DataFrame):                   contains source peptide-protein binding data
        motif_length (int):                          length of the motif being studied

    Returns:
        results_tuple (tuple):  (chunk_best_weights, chunk_best_threshold, chunk_best_mcc)
    '''

    output_df = source_df.copy()

    # Initialize what will become a list of tuples of (best_score, best_fdr, best_for) matching the indices of chunk

    chunk_best_mcc = 0
    chunk_best_threshold = None
    chunk_best_weights = None

    for weights_array in chunk:
        # Apply the current weights_array to the dict of matrices with numpy; takes almost no time
        conditional_matrices.apply_weights(weights_array, only_3d = True)

        # Get the array of scores for peptide entries in source_df using the current set of weighted matrices
        scores_array = apply_motif_scores(output_df, motif_length, conditional_matrices, sequences_2d,
                                          convert_phospho = True, return_array = True, use_weighted = True,
                                          return_2d = False, return_df = False)

        # Find the Matthews correlation coefficient for different thresholds and select the best of them
        sorted_scores = deepcopy(scores_array)
        sorted_scores = sorted_scores[sorted_scores>0]
        sorted_scores.sort()
        scores_above_thresholds = scores_array >= sorted_scores.reshape(-1,1)

        TPs = np.logical_and(scores_above_thresholds, passes_bools).sum(axis=1)
        TNs = np.logical_and(~scores_above_thresholds, ~passes_bools).sum(axis=1)
        FPs = np.logical_and(scores_above_thresholds, ~passes_bools).sum(axis=1)
        FNs = np.logical_and(~scores_above_thresholds, passes_bools).sum(axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            MCCs_by_threshold = (TPs * TNs - FPs * FNs) / np.sqrt((TPs + FPs) * (TPs + FNs) * (TNs + FPs) * (TNs + FNs))
        MCC_max_index = np.nanargmax(MCCs_by_threshold)
        optimized_threshold = sorted_scores[MCC_max_index]
        optimized_mcc = MCCs_by_threshold[MCC_max_index]

        if optimized_mcc > chunk_best_mcc:
            chunk_best_weights = weights_array
            chunk_best_threshold = optimized_threshold
            chunk_best_mcc = optimized_mcc

    return (chunk_best_weights, chunk_best_threshold, chunk_best_mcc)

def process_weights(weights_array_chunks, conditional_matrices, motif_length, source_df, sequence_col, significance_col,
                    significant_str, convert_phospho = True):
    '''
    Upper helper function for parallelization of position weight optimization; processes weights by chunking

    Args:
        weights_array_chunks (list):                list of chunks as numpy arrays for feeding to process_weights_chunk
        conditional_matrices (ConditionalMatrices): unweighted conditional position-weighted matrices
        motif_length (int):                         length of the motif being studied
        source_df (pd.DataFrame):                   source peptide-protein binding data
        sequence_col (str):                         col name in chunk where sequences are stored
        significance_col (str):                     col name in chunk where significance information is found
        significant_str (str):                      value in significance_col that denotes significance, e.g. "Yes"
        convert_phospho (bool):                     whether to convert phospho-aa's to non-phospho before doing lookups

    Returns:
        results_tuple (tuple):                      tuple of optimized (best_weights, best_threshold, best_mcc)
    '''

    passes_bools = source_df[significance_col].values == significant_str

    # Convert sequences to a 2D array upfront for performance improvement
    sequences = source_df[sequence_col].to_numpy()
    sequences_2d = unravel_seqs(sequences, motif_length, convert_phospho)

    pool = multiprocessing.Pool()

    process_partial = partial(process_weights_chunk, conditional_matrices = conditional_matrices,
                              sequences_2d = sequences_2d, passes_bools = passes_bools, source_df = source_df,
                              motif_length = motif_length)

    results = (None, None, None)
    best_mcc = 0

    with trange(len(weights_array_chunks), desc="Processing conditional matrix weights") as pbar:
        for chunk_results in pool.imap_unordered(process_partial, weights_array_chunks):
            if chunk_results[2] > best_mcc:
                results = chunk_results
                best_mcc = chunk_results[2]
                best_weights = chunk_results[0]
                best_weights_str = ", ".join(map(str, best_weights))
                print(f"\tNew record: Matthews correlation coefficient = {best_mcc} for weights: {best_weights_str}")

            pbar.update()

    pool.close()
    pool.join()

    return results

def optimize_conditional_weights(input_df, motif_length, conditional_matrices, sequence_col, significance_col,
                                 significant_str, possible_weights = None, convert_phospho = True, chunk_size = 1000):
    '''
    Parent function for finding optimal position weights to generate optimally weighted matrices

    Args:
        input_df (pd.DataFrame):                     contains peptide sequences, signal values, and significances
        motif_length (int):                          length of the motif being studied
        conditional_matrices (ConditionalMatrices):  unweighted conditional matrices based on type-position rules
        sequence_col (str):                          col name where sequences are stored
        significance_col (str):                      col name where significance information is found
        significant_str (str):                       value in significance_col that denotes significance, e.g. "Yes"
        possible_weights (np.ndarray):               array of possible weight values for each position
        convert_phospho (bool):                      whether to convert phospho-residues to non-phospho before lookups
        chunk_size (int):                            chunk size for parallel processing of weights

    Returns:
        conditional_matrices (ConditionalMatrices):  final conditional matrices with optimal weights applied
    '''

    output_df = input_df.copy()

    # Get the permuted weights and break into chunks for parallelization
    permuted_weights = permute_weights(motif_length, possible_weights)
    all_zero_rows = np.all(permuted_weights == 0, axis=1)
    permuted_weights = permuted_weights[~all_zero_rows]

    weights_array_chunks = [permuted_weights[i:i + chunk_size] for i in range(0, len(permuted_weights), chunk_size)]

    # Run the parallelized optimization process
    optimized_results = process_weights(weights_array_chunks, conditional_matrices, motif_length, output_df,
                                        sequence_col, significance_col, significant_str, convert_phospho)
    best_weights, best_threshold, best_mcc = optimized_results
    print("\t---\n", f"\tOptimal weights of {best_weights} gave Matthews correlation coefficient of {best_mcc}",
                     f"at a score threshold > {best_threshold}")

    # Apply best weights onto ConditionalMatrices object
    conditional_matrices.apply_weights(best_weights, only_3d=False)

    return conditional_matrices