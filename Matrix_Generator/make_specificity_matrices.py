#This script constructs a singular weighted matrix to predict bait-bait specificity in SLiM sequences.

import numpy as np
import pandas as pd
import os
import multiprocessing
from tqdm import trange
from functools import partial
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from Matrix_Generator.config import comparator_info, specificity_params
from general_utils.general_utils import least_different, get_delimited_list
from general_utils.weights_utils import permute_weights
from general_utils.general_vars import amino_acids, amino_acids_phos
from general_utils.user_helper_functions import get_comparator_baits
from general_utils.matrix_utils import collapse_phospho

'''---------------------------------------------------------------------------------------------------------------------
                            Define functions for generating the specificity points matrix
   ------------------------------------------------------------------------------------------------------------------'''

def bias_ratio(source_df, least_different_values, thresholds = (1,-1), passes_col = "One_Passes", pass_str = "Yes"):
    '''
    Function for finding the ratio of entries in the dataframe specific to one bait set vs. the other bait set;
    necessary for statistical adjustment when the data is not evenly distributed between baits

    Args:
        source_df (pd.DataFrame):            dataframe containing sequences, log2fc data, and significance calls
        least_different_values (np.ndarray): output from least_different() representing log2fc values from the pair of
                                             least different baits between the active comparator sets
        thresholds (tuple):                  tuple of floats as (positive_thres, negative_thres)
        passes_col (str):                    col name in source_df containing pass/fail info (significance calls)
        pass_str (str):                      the string that indicates a pass in source_df[pass_col], e.g. "Yes"

    Returns:
        ratio (float): the ratio of entries above pos_thres to entries below neg_thres
    '''

    positive_thres, negative_thres = thresholds

    # Get boolean series for thresholds and pass/fail info
    above_thres = (least_different_values > positive_thres)
    below_neg_thres = (least_different_values < negative_thres)
    passes = source_df[passes_col] == pass_str

    # Count the number of qualifying entries that are above/below the relevant threshold and are marked as pass
    above_thres_count = (above_thres & passes).sum()
    below_neg_thres_count = (below_neg_thres & passes).sum()

    # Handle divide-by-zero instances by incrementing both values by 1
    if below_neg_thres_count == 0:
        below_neg_thres_count += 1
        above_thres_count += 1

    ratio = above_thres_count / below_neg_thres_count

    return ratio

def reorder_matrix(matrix_df, include_phospho = False):
    # Basic function to add missing amino acid rows if necessary and reorder matrix_df by aa_list
    if not include_phospho:
        collapse_phospho(matrix_df)

    aa_list = list(amino_acids) if not include_phospho else list(amino_acids_phos)

    missing_row_indices = []
    for aa in aa_list:
        if aa not in matrix_df.index:
            missing_row_indices.append(aa)

    if len(missing_row_indices) > 0:
        new_rows = pd.DataFrame(0, index = missing_row_indices, columns = matrix_df.columns)
        matrix_df = pd.concat([matrix_df, new_rows])

    # Reorder the matrix rows according to aa_list as the indices
    matrix_df = matrix_df.loc[aa_list]

    return matrix_df

def make_specificity_matrix(thresholds, matching_points = (2,1,-1,-2), bias_multiplier = 1, sequences = None,
                            log2fc_values = None, significance_array = None, source_df = None, sequence_col = None,
                            log2fc_col = None, significance_col = None, pass_str = "Yes", include_phospho = False):
    '''
    Function for generating a position-weighted matrix by assigning points based on sequences and their log2fc values

    Args:
        thresholds (tuple):              tuple of (upper_positive_thres, middle_positive_thres, middle_negative_thres, upper_negative_thres)
        matching_points (tuple):         tuple of (upper_positive_points, middle_positive_points, middle_negative_points, upper_negative_points)
        bias_multiplier (float):         ratio of data specific to the first set of comparators to those specific to the second set
        sequences (np.ndarray):          sequences can be given as a numpy array for performance improvement, otherwise source_df[sequence_col] is used
        log2fc_values (np.ndarray):      log2fc values can be given as a numpy array for performance improvement, otherwise source_df[log2fc_col] is used
        significance_array (np.ndarray): significance calls (either as bool or str) can be given for performance improvement, otherwise source_df[significance_col] is used
        source_df (pd.DataFrame):        source data; only required if sequences, log2fc_values, and significant_series are not given
        sequence_col (str):              sequence col name; only required if sequences is not given
        log2fc_col (str):                log2fc col name; only required if log2fc_values is not given
        significance_col (str):          significance col name; only required if significance_array is not given
        pass_str (str):                  the string value denoting significance, e.g. "Yes"; only required if significance_array dtype is not bool
        include_phospho (bool):          whether to include phospho-residues when applying points; if False, will pool with non-phospho counterparts

    Returns:
        matrix_df (pd.DataFrame): the unweighted specificity matrix
    '''

    # Check if thresholds are sorted
    if not np.all(np.array(thresholds)[:-1] >= np.array(thresholds)[1:]) or len(thresholds) != 4:
        raise ValueError("assign_points error: thresholds must be given in descending order as a tuple of (upper_positive, middle_positive, middle_negative, upper_negative)")

    # Get indices where the results are significant
    if significance_array is None:
        significance_array = source_df[significance_col].values
        significant_indices = np.where(significance_array == pass_str)[0]
    elif significance_array.dtype != 'bool':
        significant_indices = np.where(significance_array == pass_str)[0]
    else:
        significant_indices = np.where(significance_array)[0]

    # Extract the significant sequences and log2fc values as numpy arrays
    if sequences is None:
        sequences = source_df[sequence_col].values[significant_indices]
    else:
        sequences = sequences[significant_indices]

    if log2fc_values is None:
        log2fc_values = source_df[log2fc_col].values[significant_indices]
    else:
        log2fc_values = log2fc_values[significant_indices]

    # Check that all the sequences are the same length
    same_length = np.all(np.char.str_len(sequences.astype(str)) == np.char.str_len(sequences.astype(str)[0]))
    if not same_length:
        raise ValueError(f"source_df[{sequence_col}] sequences have varying length, but are required to be equal in length")
    sequence_length = len(sequences[0])

    # Find where log2fc values pass each threshold
    passes_upper_positive = np.where(log2fc_values >= thresholds[0])
    passes_middle_positive = np.where(np.logical_and(log2fc_values >= thresholds[1], log2fc_values < thresholds[0]))
    passes_middle_negative = np.where(np.logical_and(log2fc_values <= thresholds[2], log2fc_values > thresholds[3]))
    passes_upper_negative = np.where(log2fc_values <= thresholds[3])

    # Get an array of points values matching the sequences
    points_values = np.zeros_like(log2fc_values)
    points_values[passes_upper_positive] = matching_points[0]
    points_values[passes_middle_positive] = matching_points[1]
    points_values[passes_middle_negative] = matching_points[2] * bias_multiplier
    points_values[passes_upper_negative] = matching_points[3] * bias_multiplier

    # Convert sequences to array of arrays, and do the same for matching points
    sequences = sequences.astype("<U")
    sequences_unravelled = sequences.view("U1")
    sequences_2d = np.reshape(sequences_unravelled, (-1, sequence_length))
    points_2d = np.repeat(points_values[:, np.newaxis], sequence_length, axis = 1)

    # Make a new matrix and apply points to it
    matrix_indices = np.unique(sequences_2d)
    column_names = np.char.add("#", np.arange(1, sequence_length+1).astype(str))
    matrix_df = pd.DataFrame(index = matrix_indices, columns = column_names).fillna(0)

    for column_name, residues_column, points_column in zip(column_names, np.transpose(sequences_2d), np.transpose(points_2d)):
        unique_residues, counts = np.unique(residues_column, return_counts=True)
        indices = np.searchsorted(unique_residues, residues_column)
        sums = np.bincount(indices, weights=points_column)
        matrix_df.loc[unique_residues, column_name] = sums

    # Add missing amino acid rows if necessary and reorder matrix_df by aa_list
    matrix_df = reorder_matrix(matrix_df, include_phospho)

    # Standardize matrix by max column values
    max_values = matrix_df.max(axis=0)
    matrix_df = matrix_df / max_values

    return matrix_df

'''---------------------------------------------------------------------------------------------------------------------
                           Define function to apply/optimize weights and back-calculate scores 
   ------------------------------------------------------------------------------------------------------------------'''

def get_specificity_scores(sequences, log2fc_values, unweighted_matrix, position_weights = None,
                           abs_extrema_threshold = 0.5):
    '''
    Function to back-calculate specificity scores on peptide sequences based on the generated specificity matrix

    Args:
        sequences (np.ndarray):                     the sequences to score, containing strings as their values
        log2fc_values (np.ndarray):                 the log2fc values for the sequences
        unweighted_matrix (pd.DataFrame):           the unweighted specificity matrix dataframe for getting residue values
        position_weights (np.ndarray):              an array of weights for the columns in the matrix dataframe
        abs_extrema_threshold (float):              the threshold for calling extrema, used for calculating extrema R2

    Returns:
        score_values (np.ndarray):                  the specificity scores for the given sequences
        weighted_specificity_matrix (pd.DataFrame): the specificity matrix with weights applied
        r2 (float):                                 the r-squared value for the linear association between log2fc values and scores
    '''

    # Apply position weights if necessary
    if position_weights is None:
        weighted_specificity_matrix = unweighted_matrix
    else:
        weighted_specificity_matrix = pd.DataFrame(unweighted_matrix.values * position_weights, index=unweighted_matrix.index, columns=unweighted_matrix.columns)

    # Get the indices for the matrix for each amino acid at each position
    sequence_count = len(sequences)
    sequence_length = len(sequences[0])
    sequences = sequences.astype("<U")
    sequences_unravelled = sequences.view("U1")
    sequences_2d = np.reshape(sequences_unravelled, (-1, sequence_length))

    row_indices = weighted_specificity_matrix.index.get_indexer(sequences_2d.ravel()).reshape(sequences_2d.shape)
    column_indices = np.arange(sequence_length)[np.newaxis, :].repeat(sequence_count, axis=0)

    # Calculate the points
    matrix_array = weighted_specificity_matrix.values
    points_array = matrix_array[row_indices, column_indices]
    score_values = points_array.sum(axis=1)

    # Perform linear regression between log2fc values and scores
    model = LinearRegression()
    valid_indices = np.where(np.logical_and(~np.isnan(log2fc_values), ~np.isinf(log2fc_values)))
    x_actual = score_values[valid_indices].reshape(-1, 1)
    y_actual = log2fc_values[valid_indices].reshape(-1, 1)
    model.fit(x_actual, y_actual)
    coef = model.coef_[0][0]
    intercept = model.intercept_[0]
    y_pred = model.predict(x_actual)
    r2 = r2_score(y_actual, y_pred)
    intercept_str = str(intercept) if intercept < 0 else "+" + str(intercept)
    equation = "y=" + str(coef) + intercept_str

    # Perform linear regression also on only the extrema to get R2 without influence from middle values
    extrema_model = LinearRegression()
    extrema_bools = np.abs(y_actual) > abs_extrema_threshold
    x_actual_extrema = x_actual[extrema_bools].reshape(-1, 1)
    y_actual_extrema = y_actual[extrema_bools].reshape(-1, 1)
    extrema_model.fit(x_actual_extrema, y_actual_extrema)
    extrema_coef = extrema_model.coef_[0][0]
    extrema_intercept = extrema_model.intercept_[0]
    y_pred_extrema = extrema_model.predict(x_actual_extrema)
    r2_extrema = r2_score(y_actual_extrema, y_pred_extrema)
    extrema_intercept_str = str(extrema_intercept) if extrema_intercept < 0 else "+" + str(extrema_intercept)
    equation_extrema = "y=" + str(extrema_coef) + extrema_intercept_str

    return (score_values, weighted_specificity_matrix, equation, r2, equation_extrema, r2_extrema)

'''------------------------------------------------------------------------------------------------------------------
                     Define functions for parallelized optimization of specificity matrix weights
   ------------------------------------------------------------------------------------------------------------------'''

def process_weights_chunk(chunk, significant_sequences, significant_log2fc, unweighted_matrix, fit_mode = "extrema",
                          abs_extrema_threshold = 0.5):
    '''
    Lower helper function for parallelization of position weight optimization for the specificity matrix

    Args:
        chunk (np.ndarray):                 the chunk of permuted weights currently being processed
        significant_sequences (np.ndarray): 1D array of sequences of equal length that have passed significance testing
        significant_log2fc (np.ndarray):    1D array of matching log2fc values for each sequence
        unweighted_matrix (pd.DataFrame):   the unweighted specificity matrix onto which weights will be applied
        fit_mode (str):                     if "extrema", optimizes r2 with respect to data above abs_extrema_threshold;
                                            if "all", optimizes r2 with respect to all data points
        abs_extrema_threshold (float):      only required if fit_mode is "extrema"

    Returns:
        results_tuple (tuple):  best_weights is the array of optimal weights out of the current chunk
                                best_r2 is the matching R2 value for the linear regression of scores with log2fc values
    '''

    sequence_length = len(chunk[0])
    optimized_r2 = 0.0
    optimized_r2_extrema = 0.0
    optimized_weights = np.ones(sequence_length)

    if fit_mode == "extrema":
        for permuted_weights in chunk:
            _, _, _, current_r2, _, current_r2_extrema = get_specificity_scores(significant_sequences,
                                                                                significant_log2fc, unweighted_matrix,
                                                                                permuted_weights, abs_extrema_threshold)
            if current_r2 > optimized_r2:
                optimized_weights = permuted_weights
                optimized_r2 = current_r2
                optimized_r2_extrema = current_r2_extrema
    elif fit_mode == "all":
        for permuted_weights in chunk:
            _, _, _, current_r2, _, current_r2_extrema = get_specificity_scores(significant_sequences,
                                                                                significant_log2fc, unweighted_matrix,
                                                                                permuted_weights, abs_extrema_threshold)
            if current_r2_extrema > optimized_r2_extrema:
                optimized_weights = permuted_weights
                optimized_r2 = current_r2
                optimized_r2_extrema = current_r2_extrema
    else:
        raise Exception(f"process_weights_chunk fit_mode was set to {fit_mode}, but either `extrema` or `all` was expected")

    return (optimized_weights, optimized_r2, optimized_r2_extrema)

def process_weights(weights_array_chunks, significant_sequences, significant_log2fc, unweighted_matrix,
                    fit_mode = "extrema", abs_extrema_threshold = 0.5):
    '''
    Upper helper function for parallelization of position weight optimization; processes weights by chunking

    Args:
        weights_array_chunks (list): list of chunks as numpy arrays for feeding to process_weights_chunk
        significant_sequences (np.ndarray): 1D array of sequences of equal length that have passed significance testing
        significant_log2fc (np.ndarray):    1D array of matching log2fc values for each sequence
        unweighted_matrix (pd.DataFrame):   the unweighted specificity matrix onto which weights will be applied
        fit_mode (str):                     if "extrema", optimizes r2 with respect to data above abs_extrema_threshold;
                                            if "all", optimizes r2 with respect to all data points
        abs_extrema_threshold (float):      only required if fit_mode is "extrema"

    Returns:
        results_list (list):     the list of results sets for all the weights arrays
    '''

    pool = multiprocessing.Pool()
    process_partial = partial(process_weights_chunk, significant_sequences = significant_sequences,
                              significant_log2fc = significant_log2fc, unweighted_matrix = unweighted_matrix,
                              fit_mode = fit_mode, abs_extrema_threshold = abs_extrema_threshold)

    temp_optimization_val = 0

    best_weights = None
    best_r2 = 0
    best_r2_extrema = 0

    r2_optimization_index = 2 if fit_mode == "extrema" else 1

    with trange(len(weights_array_chunks), desc="Processing specificity matrix weights") as pbar:
        for chunk_results in pool.imap_unordered(process_partial, weights_array_chunks):
            if 1 >= chunk_results[r2_optimization_index] > temp_optimization_val:
                best_weights, best_r2, best_r2_extrema = chunk_results
                temp_optimization_val = chunk_results[r2_optimization_index]
                print(f"New record: R2={best_r2} (extrema R2={best_r2_extrema}) for weights: {best_weights}")

            pbar.update()

    pool.close()
    pool.join()

    return best_weights, best_r2, best_r2_extrema

def find_optimal_weights(significant_sequences, significant_log2fc, unweighted_matrix, slim_length,
                         output_folder, possible_weights = None, chunk_size = 1000, fit_mode = "extrema",
                         abs_extrema_threshold = 0.5):
    '''
    Parent function for finding optimal position weights to generate an optimally weighted specificity matrix

    Args:
        significant_sequences (np.ndarray): 1D array of sequences of equal length that have passed significance testing
        significant_log2fc (np.ndarray):    1D array of matching log2fc values for each sequence
        unweighted_matrix (pd.DataFrame):   the unweighted specificity matrix onto which weights will be applied
        slim_length (int):                  length of the motif being studied
        output_folder (str):                the path for saving the weighted specificity matrix
        possible_weights (list):            list of arrays of possible weights at each position of the motif
        chunk_size (int):                   the number of position weights to process at a time
        fit_mode (str):                     if "extrema", optimizes r2 with respect to data above abs_extrema_threshold;
                                            if "all", optimizes r2 with respect to all data points
        abs_extrema_threshold (float):      only required if fit_mode is "extrema"

    Returns:
        results_tuple (tuple):  (best_fdr, best_for, best_score_threshold, best_weights, best_weighted_matrices_dict, best_dens_df)
    '''

    # Get the permuted weights and break into chunks for parallelization
    permuted_weights = permute_weights(slim_length, possible_weights)
    weights_array_chunks = [permuted_weights[i:i + chunk_size] for i in range(0, len(permuted_weights), chunk_size)]

    # Run the parallelized optimization process
    best_weights, best_r2, best_r2_extrema = process_weights(weights_array_chunks, significant_sequences,
                                                             significant_log2fc, unweighted_matrix,
                                                             fit_mode, abs_extrema_threshold)
    results = get_specificity_scores(significant_sequences, significant_log2fc, unweighted_matrix, best_weights,
                                     abs_extrema_threshold)
    score_values, weighted_specificity_matrix, equation, r2, equation_extrema, r2_extrema = results

    # Save the weighted matrices and scored data
    matrix_output_path = os.path.join(output_folder, "specificity_weighted_matrix.csv")
    weighted_specificity_matrix.to_csv(matrix_output_path)

    return (best_weights, score_values, weighted_specificity_matrix, equation, r2, equation_extrema, r2_extrema)

'''---------------------------------------------------------------------------------------------------------------------
                                      Define main functions and default parameters
   ------------------------------------------------------------------------------------------------------------------'''

default_comparator_info = {"comparator_set_1": None,
                           "comparator_set_2": None,
                           "seq_col": "BJO_Sequence",
                           "bait_pass_col": "One_Passes",
                           "pass_str": "Yes"}

default_specificity_params = {"thresholds": None,
                              "matching_points": None,
                              "include_phospho": False,
                              "predefined_weights": None,
                              "optimize_weights": True,
                              "possible_weights": None,
                              "output_folder": None,
                              "chunk_size": 10000,
                              "fit_mode": "extrema",
                              "abs_extrema_threshold": 0.5}

def main(source_df, comparator_info = comparator_info, specificity_params = specificity_params):
    '''
    Main function for generating and assessing optimal specificity position-weighted matrices

    Args:
        source_df (pd.DataFrame):  dataframe containing sequences, pass/fail info, and log2fc values
        comparator_info (dict):    dict of info about comparators and data locations:
                                      --> comparator_set_1 (list): 1st set of pooled comparators (min length = 1)
                                      --> comparator_set_2 (list): 2nd set of pooled comparators (min length = 1)
                                      --> seq_col (str): source_df col with peptide sequences of equal length
                                      --> bait_pass_col (str): source_df col with pass/fail info on each peptide
                                      --> pass_str (str): string representing a pass, e.g. "Yes", for converting to bool
        specificity_params (dict): dict of parameters controlling how the specificity position-weighted matrix is made:
                                      --> thresholds (tuple): (upper_plus, mid_plus, mid_minus, upper_minus)
                                      --> matching_points (tuple): tuple of points corresponding to thresholds above
                                      --> include_phospho (bool): whether to include or merge phospho-residues in matrix
                                      --> predefined_weights (np.ndarray): set of weights to use if not optimize_weights
                                      --> optimize_weights (bool): whether to optimize weights from a permuted array
                                      --> possible_weights (list): list of arrays of possible weights at each position
                                      --> output_folder (str): the folder to save matrix and scored data into
                                      --> chunk_size (int): the number of weights arrays to process at a time during
                                          parallelized weights optimization
                                      --> fit_mode (str): refers to whether to focus on fitting extrema ("extrema") or
                                          to treat all the source data equally ("all")
                                      --> abs_extrema_threshold (float): the absolute threshold for a log2fc value to be
                                          considered part of the extrema; only required when fit_mode is "extrema"

    Returns:
        specificity_results (tuple):  (output_df, predefined_weights, score_values, weighted_matrix,
                                       equation, coef, intercept, r2)
    '''

    output_df = source_df.copy()

    # Calculate least different pair of baits and corresponding log2fc for each row in output_df
    comparator_set_1, comparator_set_2 = comparator_info.get("comparator_set_1"), comparator_info.get("comparator_set_2")
    if comparator_set_1 is None or comparator_set_2 is None:
        comparator_set_1, comparator_set_2 = get_comparator_baits()
    least_different_results = least_different(output_df, comparator_set_1, comparator_set_2,
                                              return_array = True, return_df = True, in_place = False)
    least_different_values, least_different_baits, output_df = least_different_results

    # Get the multiplier to adjust for asymmetric distribution of bait specificities in the data
    thresholds = specificity_params.get("thresholds")
    if thresholds is None:
        thresholds = tuple(get_delimited_list("Please enter a comma-delimited list of thresholds, in descending order, as floats:  ", 4))
    matching_points = specificity_params.get("matching_points")
    if matching_points is None:
        matching_points = tuple(get_delimited_list(f"Please enter a comma-delimited list of associated points for each of the given thresholds ({thresholds}), in order, as floats:  ", 4))

    extreme_thresholds = (thresholds[0], thresholds[3])
    passes_col, pass_str = comparator_info.get("bait_pass_col"), comparator_info.get("pass_str")
    bias_multiplier = bias_ratio(output_df, least_different_values, extreme_thresholds, passes_col, pass_str)

    # Generate the unweighted specificity matrix
    sequence_col = comparator_info.get("seq_col")
    sequences = source_df[sequence_col].to_numpy()
    significance_array = source_df[passes_col].values == pass_str
    include_phospho = specificity_params.get("include_phospho")
    unweighted_matrix = make_specificity_matrix(thresholds, matching_points, bias_multiplier, sequences,
                                                least_different_values, significance_array,
                                                include_phospho = include_phospho)

    optimize_weights = specificity_params.get("optimize_weights")
    abs_extrema_threshold = specificity_params.get("abs_extrema_threshold")
    significant_sequences = sequences[significance_array]
    significant_least_different = least_different_values[significance_array]
    sequence_length = len(sequences[0])
    output_folder = specificity_params.get("output_folder")
    if not output_folder:
        output_folder = input("Please enter a path where specificity matrix and scoring data should be saved:  ")

    if optimize_weights:
        # Determine optimal weights by maximizing the R2 value against a permuted array of weights arrays
        possible_weights = specificity_params.get("possible_weights")
        chunk_size = specificity_params.get("chunk_size")
        fit_mode = specificity_params.get("fit_mode")
        specificity_results = find_optimal_weights(significant_sequences, significant_least_different,
                                                   unweighted_matrix, sequence_length, output_folder, possible_weights,
                                                   chunk_size, fit_mode, abs_extrema_threshold)
    else:
        # Use predefined weights, checked for correct dimensions and type
        predefined_weights = specificity_params.get("predefined_weights")
        if isinstance(predefined_weights, list) or isinstance(predefined_weights, tuple):
            predefined_weights = np.array(predefined_weights)
        elif not isinstance(predefined_weights, np.ndarray):
            raise ValueError(f"predefined_weights type is {type(predefined_weights)}, but must be np.ndarray, list, or tuple, of equal length to sequence_length")

        if len(predefined_weights) != sequence_length:
            raise ValueError(f"predefined_weights length ({len(predefined_weights)}) does not match sequence_length ({sequence_length})")

        results = get_specificity_scores(significant_sequences, significant_least_different, unweighted_matrix,
                                         predefined_weights, abs_extrema_threshold)
        score_values, weighted_specificity_matrix, equation, r2, equation_extrema, r2_extrema = results
        specificity_results = (predefined_weights, score_values, weighted_specificity_matrix, equation, r2, equation_extrema, r2_extrema)

    best_weights, score_values, weighted_specificity_matrix = specificity_results[0:3]
    equation, r2, equation_extrema, r2_extrema = specificity_results[3:]
    first_word = "Optimal" if optimize_weights else "Predefined"
    print(f"{first_word} matrix weights, {best_weights}, gave the following linear model statistics: ",
          f"\n\tAll score/log2fc pairs: R2 = {r2}, {equation}",
          f"\n\tOnly extrema where |log2fc|>{abs_extrema_threshold}: R2 = {r2_extrema}, {equation_extrema}")

    # Apply score values to output dataframe
    score_values_all = np.full(sequences.shape, np.nan, dtype=float)
    score_values_all[significance_array] = score_values
    output_df["Specificity_Score"] = score_values_all

    specificity_results = (output_df,) + specificity_results

    return specificity_results