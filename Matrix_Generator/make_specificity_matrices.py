#This script constructs a singular weighted matrix to predict bait-bait specificity in SLiM sequences.

import numpy as np
import pandas as pd
import multiprocessing
import os
import pickle
from tqdm import trange
from functools import partial
from Matrix_Generator.SpecificityMatrix import SpecificityMatrix
from general_utils.weights_utils import permute_weights
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
        chunk (np.ndarray):                     the chunk of permuted weights currently being processed
        specificity_matrix (SpecificityMatrix): the specificity matrix object

    Returns:
        chunk_results (tuple):                  (optimized_weights, optimized_mean_mcc)
    '''

    sequence_length = len(chunk[0])

    optimized_mean_mcc = 0
    optimized_weights = np.ones(sequence_length)

    for permuted_weights in chunk:
        specificity_matrix.apply_weights(permuted_weights)
        specificity_matrix.score_source_peptides(use_weighted = True)
        specificity_matrix.set_specificity_statistics(use_weighted = True)
        current_mean_mcc = specificity_matrix.weighted_mean_mcc
        if current_mean_mcc > optimized_mean_mcc:
            optimized_weights = permuted_weights
            optimized_mean_mcc = current_mean_mcc

    return (optimized_weights, optimized_mean_mcc)

def process_weights(weights_array_chunks, specificity_matrix):
    '''
    Upper helper function for parallelization of position weight optimization; processes weights by chunking

    Args:
        weights_array_chunks (list):            list of chunks as numpy arrays for feeding to process_weights_chunk
        specificity_matrix (SpecificityMatrix): the specificity matrix object

    Returns:
        results (tuple):                        (best_weights, best_r2, best_r2_extrema)
    '''

    pool = multiprocessing.Pool()
    process_partial = partial(process_weights_chunk, specificity_matrix = specificity_matrix)

    best_weights = None
    best_mean_mcc = 0

    with trange(len(weights_array_chunks), desc="Processing specificity matrix weights") as pbar:
        for chunk_results in pool.imap_unordered(process_partial, weights_array_chunks):
            if chunk_results[1] > best_mean_mcc:
                best_weights, best_mean_mcc = chunk_results
                print(f"New record: mean_mcc = {best_mean_mcc} for weights: {best_weights}")

            pbar.update()

    pool.close()
    pool.join()

    return (best_weights, best_mean_mcc)

def find_optimal_weights(specificity_matrix, motif_length, possible_weights = None, chunk_size = 1000):
    '''
    Parent function for finding optimal position weights to generate an optimally weighted specificity matrix

    Args:
        specificity_matrix (SpecificityMatrix): the specificity matrix object
        motif_length (int):                     length of the motif being studied
        possible_weights (list):                list of arrays of possible weights at each position of the motif
        chunk_size (int):                       the number of position weights to process at a time

    Returns:
        specificity_matrix (SpecificityMatrix): the fitted SpecificityMatrix object containing matrices and scored data
    '''

    # Get the permuted weights and break into chunks for parallelization
    permuted_weights = permute_weights(motif_length, possible_weights)
    all_zero_rows = np.all(permuted_weights == 0, axis=1)
    permuted_weights = permuted_weights[~all_zero_rows]

    weights_array_chunks = [permuted_weights[i:i + chunk_size] for i in range(0, len(permuted_weights), chunk_size)]

    # Run the parallelized optimization process
    results = process_weights(weights_array_chunks, specificity_matrix)
    best_weights, best_mean_mcc = results

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
        # Determine optimal weights by maximizing the R2 value against a permuted array of weights arrays
        motif_length = specificity_params["motif_length"]
        possible_weights, chunk_size = specificity_params["possible_weights"], specificity_params["chunk_size"]
        specificity_matrix = find_optimal_weights(specificity_matrix, motif_length, possible_weights, chunk_size)

    # Save the results
    output_folder = specificity_params.get("output_folder")
    if save:
        specificity_matrix.save(output_folder)

    # Save the SpecificityMatrix object for reloading when scoring novel motifs
    specificity_matrix_path = os.path.join(output_folder, "specificity_matrix.pkl")
    with open(specificity_matrix_path, "wb") as f:
        pickle.dump(specificity_matrix, f)

    return specificity_matrix