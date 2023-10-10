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
        fit_mode (str):                         'accuracy', 'f1' (f1-score), 'R2' (linear R2), or 'mcc' (MCC)
        side (str|None):                        can be "upper", "lower", or None (default; takes mean of the two)

    Returns:
        chunk_results (tuple):                  (maximizable_value, accuracy, optimized_mcc, optimized_f1,
                                                optimized_r2, optimized_weights)
    '''

    sequence_length = len(chunk[0])

    optimized_f1 = 0
    optimized_r2 = 0
    optimized_mcc = 0
    optimized_accuracy = 0
    optimized_weights = np.ones(sequence_length)
    optimized_weights_cv = np.inf # coefficient of variation = SD/mean

    if fit_mode == "accuracy":
        for weights in chunk:
            current_weights_cv = weights.std() / weights.mean()

            specificity_matrix.apply_weights(weights)
            specificity_matrix.score_source_peptides(use_weighted=True)
            specificity_matrix.set_specificity_statistics(use_weighted=True, statistic_type="accuracy")

            current_accuracy = specificity_matrix.weighted_accuracy
            better_accuracy = current_accuracy > optimized_accuracy
            better_cv = current_accuracy == optimized_accuracy and current_weights_cv < optimized_weights_cv
            if better_accuracy or better_cv:
                optimized_accuracy = current_accuracy
                optimized_weights = weights
                optimized_weights_cv = current_weights_cv

        specificity_matrix.apply_weights(optimized_weights)
        specificity_matrix.score_source_peptides(use_weighted=True)
        specificity_matrix.set_specificity_statistics(use_weighted=True, statistic_type="f1")
        optimized_r2 = specificity_matrix.weighted_linear_r2
        optimized_f1 = specificity_matrix.weighted_mean_f1

        if side == "upper":
            optimized_accuracy = specificity_matrix.weighted_upper_accuracy
        elif side == "lower":
            optimized_accuracy = specificity_matrix.weighted_lower_accuracy
        else:
            optimized_accuracy = specificity_matrix.weighted_accuracy

        maximizable_value = optimized_accuracy

    elif fit_mode == "MCC" or fit_mode == "mcc":
        for weights in chunk:
            current_weights_cv = weights.std() / weights.mean()

            specificity_matrix.apply_weights(weights)
            specificity_matrix.score_source_peptides(use_weighted=True)
            specificity_matrix.set_specificity_statistics(use_weighted=True, statistic_type="mcc")

            if side == "upper":
                current_mcc = specificity_matrix.weighted_upper_mcc
            elif side == "lower":
                current_mcc = specificity_matrix.weighted_lower_mcc
            else:
                current_mcc = specificity_matrix.weighted_mean_mcc

            better_mcc = current_mcc > optimized_mcc
            better_cv = current_mcc == optimized_mcc and current_weights_cv < optimized_weights_cv
            if better_mcc or better_cv:
                optimized_mcc = current_mcc
                optimized_weights = weights
                optimized_weights_cv = current_weights_cv

        specificity_matrix.apply_weights(optimized_weights)
        specificity_matrix.score_source_peptides(use_weighted=True)
        specificity_matrix.set_specificity_statistics(use_weighted=True, statistic_type="mcc")
        optimized_r2 = specificity_matrix.weighted_linear_r2
        optimized_f1 = np.nan
        maximizable_value = optimized_mcc

        if side == "upper":
            optimized_accuracy = specificity_matrix.weighted_upper_accuracy
        elif side == "lower":
            optimized_accuracy = specificity_matrix.weighted_lower_accuracy
        else:
            optimized_accuracy = specificity_matrix.weighted_accuracy

    elif fit_mode == "f1":
        for weights in chunk:
            current_weights_cv = weights.std() / weights.mean()

            specificity_matrix.apply_weights(weights)
            specificity_matrix.score_source_peptides(use_weighted=True)
            specificity_matrix.set_specificity_statistics(use_weighted=True, statistic_type="f1")

            if side == "upper":
                current_f1 = specificity_matrix.weighted_upper_f1
            elif side == "lower":
                current_f1 = specificity_matrix.weighted_lower_f1
            else:
                current_f1 = specificity_matrix.weighted_mean_f1

            better_f1 = current_f1 > optimized_f1
            better_cv = current_f1 == optimized_f1 and current_weights_cv < optimized_weights_cv
            if better_f1 or better_cv:
                optimized_f1 = current_f1
                optimized_weights = weights
                optimized_weights_cv = current_weights_cv

        specificity_matrix.apply_weights(optimized_weights)
        specificity_matrix.score_source_peptides(use_weighted=True)
        specificity_matrix.set_specificity_statistics(use_weighted=True, statistic_type="f1")
        optimized_r2 = specificity_matrix.weighted_linear_r2
        optimized_mcc = np.nan
        maximizable_value = optimized_f1

        if side == "upper":
            optimized_accuracy = specificity_matrix.weighted_upper_accuracy
        elif side == "lower":
            optimized_accuracy = specificity_matrix.weighted_lower_accuracy
        else:
            optimized_accuracy = specificity_matrix.weighted_accuracy

    elif fit_mode == "R2":
        for weights in chunk:
            current_weights_cv = weights.std() / weights.mean()

            specificity_matrix.apply_weights(weights)
            specificity_matrix.score_source_peptides(use_weighted=True)
            specificity_matrix.set_specificity_statistics(use_weighted=True, statistic_type="f1")

            current_r2 = specificity_matrix.weighted_linear_r2
            better_r2 = current_r2 > optimized_r2
            better_cv = current_r2 == optimized_r2 and current_weights_cv < optimized_weights_cv
            if better_r2 or better_cv:
                optimized_r2 = current_r2
                optimized_weights = weights
                optimized_weights_cv = current_weights_cv

        specificity_matrix.apply_weights(optimized_weights)
        specificity_matrix.score_source_peptides(use_weighted=True)
        specificity_matrix.set_specificity_statistics(use_weighted=True, statistic_type="f1")
        optimized_f1 = specificity_matrix.weighted_mean_f1
        optimized_mcc = np.nan
        maximizable_value = optimized_r2

        if side == "upper":
            optimized_accuracy = specificity_matrix.weighted_upper_accuracy
        elif side == "lower":
            optimized_accuracy = specificity_matrix.weighted_lower_accuracy
        else:
            optimized_accuracy = specificity_matrix.weighted_accuracy

    else:
        raise ValueError(f"process_weights_chunk got fit_mode={fit_mode}, but must be `mcc`, `f1`, or `R2`")

    output = (maximizable_value,
              optimized_accuracy, optimized_mcc, optimized_f1, optimized_r2,
              optimized_weights, optimized_weights_cv)

    return output

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
    if fit_mode == "accuracy":
        specificity_matrix.set_specificity_statistics(use_weighted=False, statistic_type="accuracy")
        initial_accuracy = specificity_matrix.unweighted_accuracy
        print(f"Initial unweighted accuracy={initial_accuracy}")
    elif fit_mode == "f1" or fit_mode == "R2":
        specificity_matrix.set_specificity_statistics(use_weighted=False, statistic_type="f1")
        initial_f1 = specificity_matrix.unweighted_mean_f1
        initial_r2 = specificity_matrix.unweighted_linear_r2
        initial_accuracy = specificity_matrix.unweighted_accuracy
        print(f"Initial unweighted accuracy={initial_accuracy}, f1-score={initial_f1}, and linear R2={initial_r2}")
    else:
        specificity_matrix.set_specificity_statistics(use_weighted=False, statistic_type="mcc")
        initial_mcc = specificity_matrix.unweighted_mean_mcc
        initial_r2 = specificity_matrix.unweighted_linear_r2
        initial_accuracy = specificity_matrix.unweighted_accuracy
        print(f"Initial unweighted accuracy={initial_accuracy}, MCC={initial_mcc}, and linear R2={initial_r2}")

    # Set up the parallel processing operation
    pool = multiprocessing.Pool()
    process_partial = partial(process_weights_chunk, specificity_matrix = specificity_matrix,
                              fit_mode = fit_mode, side = side)

    best_maximizable_value = 0
    best_f1 = 0
    best_r2 = 0
    best_accuracy = 0
    best_weights = None
    best_weights_cv = np.inf

    with trange(len(weights_array_chunks), desc="Processing specificity matrix weights") as pbar:
        for chunk_results in pool.imap_unordered(process_partial, weights_array_chunks):
            better_value = chunk_results[0] > best_maximizable_value
            better_cv = chunk_results[0] == best_maximizable_value and chunk_results[6] < best_weights_cv
            if better_value or better_cv:
                best_maximizable_value, best_accuracy, best_mcc, best_f1, best_r2, best_weights, best_weights_cv = chunk_results
                weights_str = ", ".join(best_weights.round(2).astype(str))

                if side is not None:
                    max_str = f"{side} {fit_mode} = {best_maximizable_value}"
                else:
                    max_str = f"{fit_mode} = {best_maximizable_value}"

                if fit_mode == "accuracy":
                    print(f"\nNew record: {max_str} for weights: [{weights_str}] (CV = {best_weights_cv})")
                else:
                    acc_str = f"{side} accuracy = {best_accuracy}" if side is not None else f"accuracy = {best_accuracy}"
                    print(f"\nNew record: {max_str} ({acc_str}) for weights: [{weights_str}] (CV = {best_weights_cv})")

            pbar.update()

    pool.close()
    pool.join()

    return (best_weights, best_accuracy, best_mcc, best_f1, best_r2)

def find_optimal_weights(specificity_matrix, motif_length, chunk_size = 5000, ignore_positions = None,
                         fit_mode = "f1", side = None, plot_curves = False, plot_directory = None):
    '''
    Parent function for finding optimal position weights to generate an optimally weighted specificity matrix

    Args:
        specificity_matrix (SpecificityMatrix): the specificity matrix object
        motif_length (int):                     length of the motif being studied
        chunk_size (int):                       the number of position weights to process at a time
        ignore_positions (iterable):            positions to force to 0 for weights arrays
        fit_mode (str):                         which value to maximize
        side (str|None):                        can be "upper", "lower", or None (default; takes mean of the two)
        plot_curves (bool):                     whether to plot precision-recall curves; only applies to f1-score
        plot_directory (str):                   if plot_curves is True, this is the folder where plots will be saved

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
    print(f"---\nSpecificity weights optimization mode: maximize {fit_mode}")
    results = process_weights(weights_array_chunks, specificity_matrix, fit_mode, side)
    best_weights, best_accuracy, best_mcc, best_mean_f1, best_linear_r2 = results

    # Apply the final best weights onto the specificity matrix and score the source data
    specificity_matrix.apply_weights(best_weights)
    specificity_matrix.score_source_peptides(use_weighted=True)
    if fit_mode == "accuracy":
        specificity_matrix.set_specificity_statistics(use_weighted=True, statistic_type="accuracy")
    elif fit_mode == "f1" or fit_mode == "R2":
        use_weighted = True
        statistic_type = "f1"
        plot_upper_curve = side == "upper" and plot_curves
        plot_lower_curve = side == "lower" and plot_curves
        plot_file_name = f"{str(side)}-specific_precision_recall_curve.pdf"
        plot_path = os.path.join(plot_directory, plot_file_name)
        upper_plot_path = plot_path if plot_upper_curve else None
        lower_plot_path = plot_path if plot_lower_curve else None
        specificity_matrix.set_specificity_statistics(use_weighted, statistic_type, plot_upper_curve, plot_lower_curve,
                                                      upper_plot_path, lower_plot_path)
    else:
        specificity_matrix.set_specificity_statistics(use_weighted=True, statistic_type="mcc")

    return specificity_matrix

'''---------------------------------------------------------------------------------------------------------------------
                                      Define main functions and default parameters
   ------------------------------------------------------------------------------------------------------------------'''

def main(source_df, comparator_info = comparator_info, specificity_params = specificity_params, save = True,
         plot_sigmoid = False, plot_precision_recall = True):
    '''
    Main function for generating and assessing optimal specificity position-weighted matrices

    Args:
        source_df (pd.DataFrame):  dataframe containing sequences, pass/fail info, and log2fc values
        comparator_info (dict):    dict of info about comparators and data locations as described in config.py
        specificity_params (dict): dict of specificity matrix generation parameters as described in config.py
        save (bool):               whether to automatically save the results
        plot (bool):               whether to plot a sigmoid regression of log2fc (x) and scores (y)

    Returns:
        results (tuple):           results[0] --> output_df
                                   results[1] --> specificity_matrix
                                   results[2] --> upper_specificity_matrix (only if weights are optimized)
                                   results[3] --> lower_specificity_matrix (only if weights are optimized)
    '''

    # Construct the specificity matrix from source data
    specificity_matrix = SpecificityMatrix(source_df, standardize = False, comparator_info = comparator_info,
                                           specificity_params = specificity_params)
    optimize_weights = specificity_params.get("optimize_weights")

    # Save the unweighted results
    output_folder = specificity_params.get("output_folder")
    if save:
        save_df = not optimize_weights
        specificity_matrix.save(output_folder, save_df)
        specificity_matrix.plot_regression(output_folder, use_weighted=False) if plot_sigmoid else None

    # Save the unweighted SpecificityMatrix object
    specificity_matrix_path = os.path.join(output_folder, "specificity_matrix.pkl")
    with open(specificity_matrix_path, "wb") as f:
        pickle.dump(specificity_matrix, f)

    # Optionally optimize matrix weights; not necessary if predefined weights are given as this is automatic
    if optimize_weights:
        # Determine optimal weights by maximizing the R2 value against a large randomized array of weights arrays
        length = specificity_params["motif_length"]
        chunk_size = specificity_params["chunk_size"]
        ignore_positions = specificity_params["ignore_positions"]
        fit_mode = specificity_params["fit_mode"]

        # Standardize specificity matrix prior to applying weights
        specificity_matrix.standardize_matrix(standardize = True)

        print(f"---\nOptimizing specificity matrix weights for positive log2fc values: ")
        upper_specificity_matrix = deepcopy(specificity_matrix)
        curves_path = os.path.join(output_folder, "precision_recall_curves")
        if not os.path.exists(curves_path):
            os.makedirs(curves_path)
        upper_specificity_matrix = find_optimal_weights(upper_specificity_matrix, length, chunk_size,
                                                        ignore_positions, fit_mode, side="upper", plot_curves=True,
                                                        plot_directory=curves_path)

        print(f"---\nOptimizing specificity matrix weights for negative log2fc values: ")
        lower_specificity_matrix = deepcopy(specificity_matrix)
        lower_specificity_matrix = find_optimal_weights(lower_specificity_matrix, length, chunk_size,
                                                        ignore_positions, fit_mode, side="lower", plot_curves=True,
                                                        plot_directory=curves_path)

        # Save the weighted matrices
        if save:
            weighted_upper_folder = os.path.join(output_folder, "weighted_upper")
            if not os.path.exists(weighted_upper_folder):
                os.makedirs(weighted_upper_folder)
            upper_specificity_matrix.save(weighted_upper_folder, save_df=False)
            upper_specificity_matrix.plot_regression(weighted_upper_folder, use_weighted=True) if plot_sigmoid else None

            weighted_lower_folder = os.path.join(output_folder, "weighted_lower")
            if not os.path.exists(weighted_lower_folder):
                os.makedirs(weighted_lower_folder)
            lower_specificity_matrix.save(weighted_lower_folder, save_df=False)
            lower_specificity_matrix.plot_regression(weighted_lower_folder, use_weighted=True) if plot_sigmoid else None

            # Save merged scored dataframe of upper and lower matrices
            output_df = upper_specificity_matrix.scored_source_df.copy()
            output_df.rename(columns={"Weighted_Specificity_Score": "Plus_Weighted_Specificity_Score"}, inplace=True)
            minus_scores = lower_specificity_matrix.scored_source_df["Weighted_Specificity_Score"].values
            output_df["Minus_Weighted_Specificity_Score"] = minus_scores

            df_path = os.path.join(output_folder, "specificity_scored_data.csv")
            output_df.to_csv(df_path)

        # Save the weighted SpecificityMatrix objects for reloading when scoring novel motifs
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        upper_path = os.path.join(output_folder, "weighted_upper_specificity_matrix.pkl")
        lower_path = os.path.join(output_folder, "weighted_lower_specificity_matrix.pkl")
        with open(upper_path, "wb") as f1:
            pickle.dump(upper_specificity_matrix, f1)
        with open(lower_path, "wb") as f2:
            pickle.dump(lower_specificity_matrix, f2)

        # Return unweighted and weighted matrices
        return (output_df, specificity_matrix, upper_specificity_matrix, lower_specificity_matrix)

    else:
        return (output_df, specificity_matrix)