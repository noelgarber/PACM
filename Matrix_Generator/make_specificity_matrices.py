#This script constructs a singular weighted matrix to predict bait-bait specificity in SLiM sequences.

import numpy as np
import pandas as pd
import multiprocessing
import os
import pickle
from tqdm import trange
from functools import partial
from Matrix_Generator.SpecificityMatrix import SpecificityMatrix
try:
    from Matrix_Generator.config_local import comparator_info, specificity_params
except:
    from Matrix_Generator.config import comparator_info, specificity_params

'''------------------------------------------------------------------------------------------------------------------
                     Define functions for parallelized optimization of specificity matrix weights
   ------------------------------------------------------------------------------------------------------------------'''

def process_weights_chunk(chunk, specificity_matrix):
    '''
    Lower helper function for parallelization of position weight optimization for the specificity matrix

    Args:
        chunk (np.ndarray):                     the chunk of permuted/randomized weights currently being processed
        specificity_matrix (SpecificityMatrix): the specificity matrix object

    Returns:
        chunk_results (tuple):                  (optimized_weights, optimized_mean_f1)
    '''

    sequence_length = len(chunk[0])

    optimized_mean_f1 = 0
    optimized_weights = np.ones(sequence_length)

    for weights in chunk:
        specificity_matrix.apply_weights(weights)
        specificity_matrix.score_source_peptides(use_weighted = True)
        specificity_matrix.set_specificity_statistics(use_weighted = True)
        current_mean_f1 = specificity_matrix.weighted_mean_f1
        if current_mean_f1 > optimized_mean_f1:
            optimized_weights = weights
            optimized_mean_f1 = current_mean_f1

    return (optimized_weights, optimized_mean_f1)

def process_weights(weights_array_chunks, specificity_matrix):
    '''
    Upper helper function for parallelization of position weight optimization; processes weights by chunking

    Args:
        weights_array_chunks (list):            list of chunks as numpy arrays for feeding to process_weights_chunk
        specificity_matrix (SpecificityMatrix): the specificity matrix object

    Returns:
        results (tuple):                        (best_weights, best_mean_f1)
    '''

    pool = multiprocessing.Pool()
    process_partial = partial(process_weights_chunk, specificity_matrix = specificity_matrix)

    best_weights = None
    best_mean_f1 = 0

    with trange(len(weights_array_chunks), desc="Processing specificity matrix weights") as pbar:
        for chunk_results in pool.imap_unordered(process_partial, weights_array_chunks):
            if chunk_results[1] > best_mean_f1:
                best_weights, best_mean_f1 = chunk_results
                formatted_weights = ", ".join(best_weights.round(2).astype(str))
                print(f"\nNew record: weighted mean f1-score = {best_mean_f1} for weights: [{formatted_weights}]")

            pbar.update()

    pool.close()
    pool.join()

    return (best_weights, best_mean_f1)

def find_optimal_weights(specificity_matrix, motif_length, chunk_size = 1000, ignore_positions = None):
    '''
    Parent function for finding optimal position weights to generate an optimally weighted specificity matrix

    Args:
        specificity_matrix (SpecificityMatrix): the specificity matrix object
        motif_length (int):                     length of the motif being studied
        chunk_size (int):                       the number of position weights to process at a time
        ignore_positions (iterable):            positions to force to 0 for weights arrays

    Returns:
        specificity_matrix (SpecificityMatrix): the fitted SpecificityMatrix object containing matrices and scored data
    '''

    # Get the permuted weights and break into chunks for parallelization
    sample_size = 1000000
    value_range = (0.0, 4.0)
    trial_weights = np.random.uniform(value_range[0], value_range[1], size=(sample_size, motif_length))
    if ignore_positions is not None:
        for position in ignore_positions:
            trial_weights[:,position] = 0

    all_zero_rows = np.all(trial_weights == 0, axis=1)
    trial_weights = trial_weights[~all_zero_rows]

    weights_array_chunks = [trial_weights[i:i + chunk_size] for i in range(0, len(trial_weights), chunk_size)]

    # Run the parallelized optimization process
    results = process_weights(weights_array_chunks, specificity_matrix)
    best_weights, best_mean_f1 = results

    # Apply the final best weights onto the specificity matrix and score the source data
    specificity_matrix.apply_weights(best_weights)
    specificity_matrix.score_source_peptides(use_weighted = True)
    specificity_matrix.set_specificity_statistics(use_weighted=True)

    return specificity_matrix

'''---------------------------------------------------------------------------------------------------------------------
                                      Define main functions and default parameters
   ------------------------------------------------------------------------------------------------------------------'''

def main(source_df, comparator_info = comparator_info, specificity_params = specificity_params, save = True):
    '''
    Main function for generating and assessing optimal specificity position-weighted matrices

    Args:
        source_df (pd.DataFrame):  dataframe containing sequences, pass/fail info, and log2fc values
        comparator_info (dict):    dict of info about comparators and data locations as described in config.py
        specificity_params (dict): dict of specificity matrix generation parameters as described in config.py
        save (bool):               whether to automatically save the results

    Returns:
        specificity_results (tuple):  (output_df, predefined_weights, score_values, weighted_matrix,
                                       equation, coef, intercept, r2)
    '''

    # Construct the specificity matrix from source data
    specificity_matrix = SpecificityMatrix(source_df, comparator_info, specificity_params)

    # Optionally optimize matrix weights; not necessary if predefined weights are given as this is automatic
    optimize_weights = specificity_params.get("optimize_weights")
    if optimize_weights:
        # Determine optimal weights by maximizing the R2 value against a large randomized array of weights arrays
        motif_length = specificity_params["motif_length"]
        chunk_size = specificity_params["chunk_size"]
        ignore_positions = specificity_params["ignore_positions"]
        specificity_matrix = find_optimal_weights(specificity_matrix, motif_length, chunk_size, ignore_positions)

    # Save the results
    output_folder = specificity_params.get("output_folder")
    if save:
        specificity_matrix.save(output_folder)
        specificity_matrix.plot_regression(output_folder, use_weighted=optimize_weights)

    # Save the SpecificityMatrix object for reloading when scoring novel motifs
    specificity_matrix_path = os.path.join(output_folder, "specificity_matrix.pkl")
    with open(specificity_matrix_path, "wb") as f:
        pickle.dump(specificity_matrix, f)

    return specificity_matrix