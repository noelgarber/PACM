#This script constructs a singular weighted matrix to predict bait-bait specificity in SLiM sequences.

import numpy as np
import pandas as pd
import multiprocessing
from tqdm import trange
from functools import partial
from Matrix_Generator.config import comparator_info, specificity_params
from Matrix_Generator.SpecificityMatrix import SpecificityMatrix
from general_utils.weights_utils import permute_weights

'''------------------------------------------------------------------------------------------------------------------
                     Define functions for parallelized optimization of specificity matrix weights
   ------------------------------------------------------------------------------------------------------------------'''

def process_weights_chunk(chunk, specificity_matrix, fit_mode = "extrema", abs_extrema_threshold = 0.5):
    '''
    Lower helper function for parallelization of position weight optimization for the specificity matrix

    Args:
        chunk (np.ndarray):                     the chunk of permuted weights currently being processed
        specificity_matrix (SpecificityMatrix): the specificity matrix object
        fit_mode (str):                         if "extrema", optimizes r2 with respect to data above abs_extrema_threshold;
                                                if "all", optimizes r2 with respect to all data points
        abs_extrema_threshold (float):          only required if fit_mode is "extrema"

    Returns:
        chunk_results (tuple):                  (optimized_weights, optimized_r2, optimized_r2_extrema)
    '''

    sequence_length = len(chunk[0])
    optimized_r2 = 0.0
    optimized_r2_extrema = 0.0
    optimized_weights = np.ones(sequence_length)
    
    if fit_mode != "extrema" and fit_mode != "all": 
        raise Exception(f"process_weights_chunk fit_mode was set to {fit_mode}, but `extrema` or `all` was expected")
    use_extrema = fit_mode == "extrema"

    for permuted_weights in chunk:
        specificity_matrix.apply_weights(permuted_weights)
        specificity_matrix.score_source_peptides(use_weighted = True)
        specificity_matrix.set_specificity_statistics(abs_extrema_threshold, use_weighted = True)
        current_r2 = specificity_matrix.weighted_extrema_r2 if use_extrema else specificity_matrix.weighted_linear_r2
        if current_r2 > optimized_r2:
            optimized_weights = permuted_weights
            optimized_r2 = specificity_matrix.weighted_linear_r2
            optimized_r2_extrema = specificity_matrix.weighted_extrema_r2

    return (optimized_weights, optimized_r2, optimized_r2_extrema)

def process_weights(weights_array_chunks, specificity_matrix, fit_mode = "extrema", abs_extrema_threshold = 0.5):
    '''
    Upper helper function for parallelization of position weight optimization; processes weights by chunking

    Args:
        weights_array_chunks (list):            list of chunks as numpy arrays for feeding to process_weights_chunk
        specificity_matrix (SpecificityMatrix): the specificity matrix object
        fit_mode (str):                         if "extrema", optimizes r2 with respect to data above abs_extrema_threshold;
                                                if "all", optimizes r2 with respect to all data points
        abs_extrema_threshold (float):          only required if fit_mode is "extrema"

    Returns:
        results (tuple):                        (best_weights, best_r2, best_r2_extrema)
    '''

    pool = multiprocessing.Pool()
    process_partial = partial(process_weights_chunk, specificity_matrix = specificity_matrix,
                              fit_mode = fit_mode, abs_extrema_threshold = abs_extrema_threshold)

    best_weights = None
    best_r2 = 0
    best_r2_extrema = 0

    optimization_val = 0
    r2_optimization_index = 2 if fit_mode == "extrema" else 1

    with trange(len(weights_array_chunks), desc="Processing specificity matrix weights") as pbar:
        for chunk_results in pool.imap_unordered(process_partial, weights_array_chunks):
            if 1 >= chunk_results[r2_optimization_index] > optimization_val:
                best_weights, best_r2, best_r2_extrema = chunk_results
                optimization_val = chunk_results[r2_optimization_index]
                print(f"New record: R2={best_r2} (extrema R2={best_r2_extrema}) for weights: {best_weights}")

            pbar.update()

    pool.close()
    pool.join()

    return (best_weights, best_r2, best_r2_extrema)

def find_optimal_weights(specificity_matrix, motif_length, possible_weights = None, chunk_size = 1000,
                         fit_mode = "extrema", abs_extrema_threshold = 0.5):
    '''
    Parent function for finding optimal position weights to generate an optimally weighted specificity matrix

    Args:
        specificity_matrix (SpecificityMatrix): the specificity matrix object
        motif_length (int):                     length of the motif being studied
        possible_weights (list):                list of arrays of possible weights at each position of the motif
        chunk_size (int):                       the number of position weights to process at a time
        fit_mode (str):                         if "extrema", optimizes r2 with respect to data above abs_extrema_threshold;
                                                if "all", optimizes r2 with respect to all data points
        abs_extrema_threshold (float):          only required if fit_mode is "extrema"

    Returns:
        specificity_matrix (SpecificityMatrix): the fitted SpecificityMatrix object containing matrices and scored data
    '''

    # Get the permuted weights and break into chunks for parallelization
    permuted_weights = permute_weights(motif_length, possible_weights)
    weights_array_chunks = [permuted_weights[i:i + chunk_size] for i in range(0, len(permuted_weights), chunk_size)]

    # Run the parallelized optimization process
    results = process_weights(weights_array_chunks, specificity_matrix, fit_mode, abs_extrema_threshold)
    best_weights, best_r2, best_r2_extrema = results

    # Apply the final best weights onto the specificity matrix and score the source data
    specificity_matrix.apply_weights(best_weights)
    specificity_matrix.score_source_peptides(use_weighted = True)
    specificity_matrix.set_specificity_statistics(abs_extrema_threshold, use_weighted=True)

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
        fit_mode, abs_extrema_threshold = specificity_params["fit_mode"], specificity_params["abs_extrema_threshold"]
        specificity_matrix = find_optimal_weights(specificity_matrix, motif_length, possible_weights,
                                                 chunk_size, fit_mode, abs_extrema_threshold)

    # Save the results
    if save:
        output_folder = specificity_params.get("output_folder")
        specificity_matrix.save(output_folder)

    return specificity_matrix