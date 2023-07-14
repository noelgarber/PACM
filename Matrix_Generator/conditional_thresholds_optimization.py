# Contains functions to optimize matrix weights in a parallelized manner

import numpy as np
import pandas as pd
import os
import psutil
import math
import time
import multiprocessing
from tqdm import trange
from functools import partial
from Matrix_Generator.ConditionalMatrix import ConditionalMatrices
from Matrix_Generator.conditional_scoring import apply_motif_scores
from general_utils.general_utils import save_dataframe, save_weighted_matrices, unravel_seqs, permute_array
from general_utils.statistics import fdr_for_optimizer
from general_utils.permutation_utils import permutation_split_memory

def process_thresholds_chunk(thresholds_chunk, conditional_matrices, source_df, slim_length, seq_col,
                             significance_col, score_col, significant_str = "Yes", allowed_rate_divergence = 0.2,
                             convert_phospho = True):
    '''
    Lower helper function for parallelization of position weight optimization; processes chunks of permuted weights

    Note that matrix_arrays_dict, unlike matrices_dict which appears elsewhere, contains numpy arrays instead of
    pandas dataframes. This greatly improves performance, but requires the matching pd.Index to be passed separately.

    Args:
        thresholds_chunk (np.ndarray):              the chunk of permuted thresholds currently being processed
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
        results_tuple (tuple):  (chunk_best_fdr, chunk_best_for, chunk_best_thresholds, chunk_best_source_df)
    '''

    output_df = source_df.copy()

    passes_bools = source_df[significance_col].to_numpy() == significant_str

    # Convert sequences to a 2D array upfront for performance improvement
    sequences = source_df[seq_col].to_numpy()
    sequences_2d = unravel_seqs(sequences, slim_length, convert_phospho)

    # Get the motif scores for whole peptides and the residues constituting them, using unwweighted conditional matrices
    scores_array, scores_2d = apply_motif_scores(output_df, slim_length, conditional_matrices, sequences_2d,
                                                 convert_phospho = True, return_array = True, use_weighted = False,
                                                 return_2d = True)

    # Test whether residues and whole peptides pass thresholds
    residue_passes_thresholds = thresholds_chunk[:,np.newaxis,:] <= scores_2d
    peptide_passes_thresholds = residue_passes_thresholds.all(axis=2) # shape = (permutations_count, peptide_count)
    del residue_passes_thresholds # save memory by explicitly deleting large arrays

    # Calculate arguments for finding FDR and FOR
    FP_bools = np.logical_and(peptide_passes_thresholds, ~passes_bools) # passes_bools is automatically broadcasted
    FP_count = FP_bools.sum(axis=1)
    del FP_bools

    FN_bools = np.logical_and(~peptide_passes_thresholds, passes_bools)
    FN_count = FN_bools.sum(axis=1)
    del FN_bools

    positive_calls = peptide_passes_thresholds.sum(axis=1)
    negative_calls = (~peptide_passes_thresholds).sum(axis=1)

    # Calculate FDR and FOR for each thresholds array in thresholds_chunk
    with np.errstate(divide="ignore", invalid="ignore"):
        current_best_fdr = np.where(positive_calls > 0, FP_count / positive_calls, np.inf)
        current_best_for = np.where(negative_calls > 0, FN_count / negative_calls, np.inf)
    del FP_count, FN_count, positive_calls, negative_calls

    optimal_values_array = np.column_stack((current_best_fdr, current_best_for))
    del current_best_fdr, current_best_for

    # Find the chunk index for the weights array that produces the lowest optimal FDR & FOR values (using the mean)
    best_index = fdr_for_optimizer(optimal_values_array, allowed_rate_divergence)
    chunk_best_fdr, chunk_best_for = optimal_values_array[best_index]
    chunk_best_thresholds = thresholds_chunk[best_index]
    del optimal_values_array, thresholds_chunk

    # Get the matching dict of weighted matrices and use it to apply final scores to output_df
    residue_passes_thresholds = scores_2d >= chunk_best_thresholds
    chunk_best_source_df = apply_motif_scores(output_df, slim_length, conditional_matrices, sequences_2d, score_col,
                                              convert_phospho = convert_phospho, add_residue_cols = True,
                                              in_place = False, return_array = False, use_weighted = False,
                                              return_2d = True, residue_calls_2d = residue_passes_thresholds)

    return (chunk_best_fdr, chunk_best_for, chunk_best_thresholds, chunk_best_source_df)

def process_thresholds(threshold_chunks, conditional_matrices, slim_length, source_df, sequence_col, significance_col,
                       significant_str, score_col, allowed_rate_divergence = 0.2, convert_phospho = True, group = None):
    '''
    Upper helper function for parallelization of position thresholds optimization; processes thresholds arrs by chunking

    Args:
        threshold_chunks (list):                    permuted threshold array chunks
        conditional_matrices (ConditionalMatrices): unweighted conditional position-weighted matrices
        slim_length (int):                          length of the motif being studied
        source_df (pd.DataFrame):                   source peptide-protein binding data
        sequence_col (str):                         col name in chunk where sequences are stored
        significance_col (str):                     col name in chunk where significance information is found
        significant_str (str):                      value in significance_col that denotes significance, e.g. "Yes"
        score_col (str):                            col name where motif scores are found
        allowed_rate_divergence (float):            maximum allowed difference between FDR and FOR pairs
        convert_phospho (bool):                     whether to convert phospho-aa's to non-phospho before doing lookups
        group (tuple):                              optional; tuple of (group_number, group_count)

    Returns:
        results_list (list):     the list of results sets for all the weights arrays
    '''

    pool = multiprocessing.Pool()

    process_partial = partial(process_thresholds_chunk, conditional_matrices = conditional_matrices,
                              source_df = source_df, slim_length = slim_length, seq_col = sequence_col,
                              significance_col = significance_col, score_col = score_col,
                              significant_str = significant_str, allowed_rate_divergence = allowed_rate_divergence,
                              convert_phospho = convert_phospho)

    results = None
    best_rate_mean = 9999

    if group is not None:
        progress_bar_description = f"Processing thresholds for group {group[0]} of {group[1]}"
        record_str = "New group record"
    else:
        progress_bar_description = f"Processing thresholds"
        record_str = "New record"

    with trange(len(threshold_chunks), desc=progress_bar_description) as pbar:
        for chunk_results in pool.imap_unordered(process_partial, threshold_chunks):
            rate_mean = (chunk_results[0] + chunk_results[1]) / 2
            rate_delta = abs(chunk_results[0] - chunk_results[1])
            if rate_mean < best_rate_mean and rate_delta <= allowed_rate_divergence:
                best_rate_mean = rate_mean
                results = chunk_results
                print(f"\t{record_str}: FDR={results[0]} | FOR={results[1]} | thresholds={results[2]}")

            pbar.update()

    pool.close()
    pool.join()

    return results

def find_optimal_thresholds(input_df, slim_length, conditional_matrices, sequence_col, significance_col, significant_str,
                            score_col, matrix_output_folder, output_folder, steps = 50,
                            convert_phospho = True, save_pickled_matrix_dict = True):
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
        steps (int):                                 the steps between 0 and 1 for where to set residue score thres
        matrix_output_folder (str):                  path for saving weighted matrices
        output_folder (str):                         path for saving the scored data
        convert_phospho (bool):                      whether to convert phospho-residues to non-phospho before lookups
        save_pickled_matrix_dict (bool):             whether to save a pickled version of the dict of matrices

    Returns:
        results_tuple (tuple):  (best_fdr, best_for, best_score_threshold, best_weights, best_weighted_matrices_dict, best_dens_df)
    '''

    output_df = input_df.copy()

    # Permute thresholds; manage memory to ensure the available maximum is not exceeded
    position_thresholds = np.linspace(start=0, stop=1, num=steps, dtype=np.float16)
    thresholds_count = len(position_thresholds)

    available_memory = psutil.virtual_memory().available  # in bytes
    memory_limit = 0.5 * available_memory

    iteration_positions, partial_slim_length = permutation_split_memory(thresholds_count, slim_length, memory_limit,
                                                                      element_bits = 16, meshgrid_bits = 64)

    # Permute non-iterated and iterated positions
    cpus = os.cpu_count()
    if iteration_positions > 0:
        partial_arrays = permute_array(position_thresholds, partial_slim_length)
        iterated_permuted_arrays = permute_array(position_thresholds, iteration_positions)

        # Get chunk size based on cpu_count
        chunk_size = len(partial_arrays)/(cpus*100) if len(partial_arrays)>(cpus*100) else len(partial_arrays)/cpus
        chunk_size = math.ceil(chunk_size)

        iteration_count = len(iterated_permuted_arrays)
        print(f"Starting parallel threshold optimization with chunk size={chunk_size:,}",
              f"\n\tProcessing in {iteration_count:,} iterated groups due to memory constraints")

        best_rate_mean = 9999
        results = [None, None, None, None]
        for i, iterable_permuted_array in enumerate(iterated_permuted_arrays):
            iterable_permuted_arrays = np.tile(iterable_permuted_array, (partial_arrays.shape[0], 1))
            permuted_batch = np.concatenate([iterable_permuted_arrays, partial_arrays], axis=1)

            # Separate the permuted arrays into chunks for parallel processing
            batch_chunks = [permuted_batch[i:i + chunk_size] for i in range(0, len(permuted_batch), chunk_size)]
            if len(permuted_batch) % chunk_size != 0:
                # Handle cases where len(permuted_batch) is not a multiple of chunk_size
                batch_chunks.append(permuted_batch[-(len(permuted_batch) % chunk_size):])

            group = (i, iteration_count)
            group_results = process_thresholds(batch_chunks, conditional_matrices, slim_length, output_df, sequence_col,
                                         significance_col, significant_str, score_col, allowed_rate_divergence = 0.2,
                                         convert_phospho = convert_phospho, group = group)

            # Delete unnecessary objects to save memory
            del batch_chunks, iterable_permuted_arrays, permuted_batch

            # Evaluate FDR and FOR against current best mean rate value
            fdr_val, for_val, residue_thresholds, scored_df = group_results
            rate_mean = (fdr_val + for_val) / 2
            if rate_mean < best_rate_mean:
                results = group_results

    else:
        permuted_arrays = permute_array(position_thresholds, slim_length)

        # Get chunk size based on cpu_count
        chunk_size = len(permuted_arrays)/(cpus*100) if len(permuted_arrays)>(cpus*100) else len(permuted_arrays)/cpus
        chunk_size = math.ceil(chunk_size)

        # Separate the permuted arrays into chunks for parallel processing
        chunks = [permuted_arrays[i:i + chunk_size] for i in range(0, len(permuted_arrays), chunk_size)]
        if len(permuted_batch) % chunk_size != 0:
            # Handle cases where len(permuted_arrays) is not a multiple of chunk_size
            chunks.append(permuted_batch[-(len(permuted_batch) % chunk_size):])

        results = process_thresholds(chunks, conditional_matrices, slim_length, output_df, sequence_col,
                                     significance_col, significant_str, score_col, allowed_rate_divergence = 0.2,
                                     convert_phospho = convert_phospho)

        # Delete unnecessary objects to save memory
        del permuted_arrays, chunks

    print("\t---\n",
          f"\tDone! Optimal residue thresholds of {results[2]} gave FDR = {results[0]} and FOR = {results[1]}")

    # Save the weighted matrices and scored data
    save_weighted_matrices(conditional_matrices.matrices_dict, matrix_output_folder, save_pickled_matrix_dict)
    with open(os.path.join(matrix_output_folder, "final_residue_thresholds.txt"), "w") as file:
        file.write(",".join(map(str, best_residue_thresholds)))

    save_dataframe(best_scored_df, output_folder, "pairwise_scored_data.csv")

    print(f"Saved weighted matrices and scored data to {output_folder}")

    return results