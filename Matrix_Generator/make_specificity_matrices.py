#This script constructs a singular weighted matrix to predict bait-bait specificity in SLiM sequences.

import numpy as np
import pandas as pd
import multiprocessing
import os
import pickle
from copy import deepcopy
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

def process_weights_chunk(chunk, specificity_matrix, fit_mode = "f1", side = None):
    '''
    Lower helper function for parallelization of position weight optimization for the specificity matrix

    Args:
        chunk (np.ndarray):                     the chunk of permuted/randomized weights currently being processed
        specificity_matrix (SpecificityMatrix): the specificity matrix object
        fit_mode (str):                         whether to use f1-score ('f1'), linear R2 ('R2'), or MCC ('mcc'/"MCC")
        side (str|None):                        can be "upper", "lower", or None (default; takes mean of the two)

    Returns:
        chunk_results (tuple):                  (maximizable_value, optimized_f1, optimized_r2, optimized_weights)
    '''

    sequence_length = len(chunk[0])

    maximizable_value = None
    optimized_f1 = 0
    optimized_r2 = 0
    optimized_mcc = 0
    optimized_weights = np.ones(sequence_length)

    if fit_mode == "MCC" or fit_mode == "mcc":
        for weights in chunk:
            specificity_matrix.apply_weights(weights)
            specificity_matrix.score_source_peptides(use_weighted=True)
            specificity_matrix.set_specificity_statistics(use_weighted=True, assign_f1=False, assign_mcc=True)

            if side == "upper":
                current_mcc = specificity_matrix.weighted_upper_mcc
            elif side == "lower":
                current_mcc = specificity_matrix.weighted_lower_mcc
            else:
                current_mcc = specificity_matrix.weighted_mean_mcc

            if current_mcc > optimized_mcc:
                optimized_mcc = current_mcc
                optimized_weights = weights

        specificity_matrix.apply_weights(optimized_weights)
        specificity_matrix.score_source_peptides(use_weighted=True)
        specificity_matrix.set_specificity_statistics(use_weighted=True, assign_f1=False, assign_mcc=True)
        optimized_r2 = specificity_matrix.weighted_linear_r2
        optimized_f1 = np.nan
        maximizable_value = optimized_mcc

    elif fit_mode == "f1":
        for weights in chunk:
            specificity_matrix.apply_weights(weights)
            specificity_matrix.score_source_peptides(use_weighted=True)
            specificity_matrix.set_specificity_statistics(use_weighted=True, assign_f1=True, assign_mcc=False)

            if side == "upper":
                current_f1 = specificity_matrix.weighted_upper_f1
            elif side == "lower":
                current_f1 = specificity_matrix.weighted_lower_f1
            else:
                current_f1 = specificity_matrix.weighted_mean_f1

            if current_f1 > optimized_f1:
                optimized_f1 = current_f1
                optimized_weights = weights

        specificity_matrix.apply_weights(optimized_weights)
        specificity_matrix.score_source_peptides(use_weighted=True)
        specificity_matrix.set_specificity_statistics(use_weighted=True, assign_f1=True, assign_mcc=False)
        optimized_r2 = specificity_matrix.weighted_linear_r2
        optimized_mcc = np.nan
        maximizable_value = optimized_f1

    elif fit_mode == "R2":
        for weights in chunk:
            specificity_matrix.apply_weights(weights)
            specificity_matrix.score_source_peptides(use_weighted=True)
            specificity_matrix.set_specificity_statistics(use_weighted=True, assign_f1=False, assign_mcc=False)

            current_r2 = specificity_matrix.weighted_linear_r2
            if current_r2 > optimized_r2:
                optimized_r2 = current_r2
                optimized_weights = weights

        specificity_matrix.apply_weights(optimized_weights)
        specificity_matrix.score_source_peptides(use_weighted=True)
        specificity_matrix.set_specificity_statistics(use_weighted=True, assign_f1=True, assign_mcc=False)
        optimized_f1 = specificity_matrix.weighted_mean_f1
        optimized_mcc = np.nan
        maximizable_value = optimized_r2

    else:
        raise ValueError(f"process_weights_chunk got fit_mode={fit_mode}, but must be `mcc`, `f1`, or `R2`")

    return (maximizable_value, optimized_mcc, optimized_f1, optimized_r2, optimized_weights)

def process_weights(weights_array_chunks, specificity_matrix, fit_mode = "f1", side = None):
    '''
    Upper helper function for parallelization of position weight optimization; processes weights by chunking

    Args:
        weights_array_chunks (list):            list of chunks as numpy arrays for feeding to process_weights_chunk
        specificity_matrix (SpecificityMatrix): the specificity matrix object
        fit_mode (str):                         name of value to maximize
        side (str|None):                        can be "upper", "lower", or None (default; takes mean of the two)

    Returns:
        results (tuple):                        (best_weights, best_mean_f1)
    '''

    # Calculate the initial value
    specificity_matrix.score_source_peptides(use_weighted=False)
    if fit_mode == "f1" or fit_mode == "R2":
        specificity_matrix.set_specificity_statistics(use_weighted=False, assign_f1=True, assign_mcc=False)
        initial_f1 = specificity_matrix.unweighted_mean_f1
        initial_r2 = specificity_matrix.unweighted_linear_r2
        print(f"Initial unweighted f1-score={initial_f1} and linear R2={initial_r2}")
    else:
        specificity_matrix.set_specificity_statistics(use_weighted=False, assign_f1=False, assign_mcc=True)
        initial_mcc = specificity_matrix.unweighted_mean_mcc
        initial_r2 = specificity_matrix.unweighted_linear_r2
        print(f"Initial unweighted MCC={initial_mcc} and linear R2={initial_r2}")

    # Set up the parallel processing operation
    pool = multiprocessing.Pool()
    process_partial = partial(process_weights_chunk, specificity_matrix = specificity_matrix,
                              fit_mode = fit_mode, side = side)

    best_maximizable_value = 0
    best_f1 = 0
    best_r2 = 0
    best_weights = None

    with trange(len(weights_array_chunks), desc="Processing specificity matrix weights") as pbar:
        for chunk_results in pool.imap_unordered(process_partial, weights_array_chunks):
            if chunk_results[0] > best_maximizable_value:
                best_maximizable_value, best_mcc, best_f1, best_r2, best_weights = chunk_results
                formatted_weights = ", ".join(best_weights.round(2).astype(str))
                print(f"\nNew record: {fit_mode}={best_maximizable_value} for weights: [{formatted_weights}]")

            pbar.update()

    pool.close()
    pool.join()

    return (best_weights, best_mcc, best_f1, best_r2)

def find_optimal_weights(specificity_matrix, motif_length, chunk_size = 5000, ignore_positions = None,
                         fit_mode = "f1", f1_side = None):
    '''
    Parent function for finding optimal position weights to generate an optimally weighted specificity matrix

    Args:
        specificity_matrix (SpecificityMatrix): the specificity matrix object
        motif_length (int):                     length of the motif being studied
        chunk_size (int):                       the number of position weights to process at a time
        ignore_positions (iterable):            positions to force to 0 for weights arrays
        fit_mode (str):                         which value to maximize
        f1_side (str|None):                     can be "upper", "lower", or None (default; takes mean of the two)

    Returns:
        specificity_matrix (SpecificityMatrix): the fitted SpecificityMatrix object containing matrices and scored data
    '''

    # Get the permuted weights and break into chunks for parallelization
    sample_size = 100000
    value_range = (0.0, 4.0)
    trial_weights = np.random.uniform(value_range[0], value_range[1], size=(sample_size, motif_length))
    if ignore_positions is not None:
        for position in ignore_positions:
            trial_weights[:,position] = 0

    all_zero_rows = np.all(trial_weights == 0, axis=1)
    trial_weights = trial_weights[~all_zero_rows]

    weights_array_chunks = [trial_weights[i:i + chunk_size] for i in range(0, len(trial_weights), chunk_size)]

    # Run the parallelized optimization process
    results = process_weights(weights_array_chunks, specificity_matrix, fit_mode, f1_side)
    best_weights, best_mcc, best_mean_f1, best_linear_r2 = results

    # Apply the final best weights onto the specificity matrix and score the source data
    specificity_matrix.apply_weights(best_weights)
    specificity_matrix.score_source_peptides(use_weighted = True)
    if fit_mode == "f1" or fit_mode == "R2":
        specificity_matrix.set_specificity_statistics(use_weighted=True, assign_f1=True)
    else:
        specificity_matrix.set_specificity_statistics(use_weighted=True, assign_f1=False, assign_mcc=True)

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

    # Save the unweighted results
    output_folder = specificity_params.get("output_folder")
    if save:
        specificity_matrix.save(output_folder)
        specificity_matrix.plot_regression(output_folder, use_weighted=False)

    # Save the unweighted SpecificityMatrix object
    specificity_matrix_path = os.path.join(output_folder, "specificity_matrix.pkl")
    with open(specificity_matrix_path, "wb") as f:
        pickle.dump(specificity_matrix, f)

    # Optionally optimize matrix weights; not necessary if predefined weights are given as this is automatic
    optimize_weights = specificity_params.get("optimize_weights")
    if optimize_weights:
        # Determine optimal weights by maximizing the R2 value against a large randomized array of weights arrays
        length = specificity_params["motif_length"]
        chunk_size = specificity_params["chunk_size"]
        ignore_positions = specificity_params["ignore_positions"]
        fit_mode = specificity_params["fit_mode"]

        print(f"---\nOptimizing specificity matrix weights for positive log2fc values: ")
        upper_specificity_matrix = deepcopy(specificity_matrix)
        upper_specificity_matrix = find_optimal_weights(upper_specificity_matrix, length, chunk_size,
                                                        ignore_positions, fit_mode)

        print(f"---\nOptimizing specificity matrix weights for negative log2fc values: ")
        lower_specificity_matrix = deepcopy(specificity_matrix)
        lower_specificity_matrix = find_optimal_weights(lower_specificity_matrix, length, chunk_size,
                                                        ignore_positions, fit_mode)

        # Save the weighted results
        if save:
            weighted_upper_folder = os.path.join(output_folder, "weighted_upper")
            if not os.path.exists(weighted_upper_folder):
                os.makedirs(weighted_upper_folder)
            upper_specificity_matrix.save(weighted_upper_folder)
            upper_specificity_matrix.plot_regression(weighted_upper_folder, use_weighted=True)

            weighted_lower_folder = os.path.join(output_folder, "weighted_lower")
            if not os.path.exists(weighted_lower_folder):
                os.makedirs(weighted_lower_folder)
            lower_specificity_matrix.save(weighted_lower_folder)
            lower_specificity_matrix.plot_regression(weighted_lower_folder, use_weighted=True)

        # Save the weighted SpecificityMatrix objects for reloading when scoring novel motifs
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        upper_path = os.path.join(output_folder, "weighted_upper_specificity_matrix.pkl")
        with open(upper_path, "wb") as f1:
            pickle.dump(upper_specificity_matrix, f1)

        lower_path = os.path.join(output_folder, "weighted_lower_specificity_matrix.pkl")
        with open(lower_path, "wb") as f2:
            pickle.dump(lower_specificity_matrix, f2)

        # Return unweighted and weighted matrices
        return (specificity_matrix, upper_specificity_matrix, lower_specificity_matrix)

    else:
        return (specificity_matrix)