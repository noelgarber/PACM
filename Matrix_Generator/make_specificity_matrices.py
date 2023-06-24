#This script constructs a singular weighted matrix to predict bait-bait specificity in SLiM sequences.

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from general_utils.general_utils import permute_weights, least_different, get_delimited_list
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

    aa_list = amino_acids if not include_phospho else amino_acids_phos
    missing_row_indices = [residue for residue in aa_list if residue not in matrix_df.index]
    if missing_row_indices:
        new_rows = pd.DataFrame(0, columns = matrix_df.columns, index = missing_row_indices)
        matrix_df = pd.concat([matrix_df, new_rows])
    matrix_df = matrix_df.loc[aa_list]

    return matrix_df

def make_specificity_matrix(thresholds, matching_points = (2,1,-1,-2), bias_multiplier = 1, sequences = None,
                            log2fc_values = None, significance_array = None, source_df = None, sequence_col = None,
                            log2fc_col = None, significance_col = None, pass_str = "Yes", include_phospho = False):
    '''
    Function for generating a position-weigted matrix by assigning points based on sequences and their log2fc values

    Args:
        thresholds (tuple):              tuple of (upper_positive_thres, middle_positive_thres, middle_negative_thres, upper_negative_thres)
        matching_points (tuple):         tuple of (upper_positive_points, middle_positive_points, middle_negative_points, upper_negative_points)
        bias_multiplier (float):         ratio of data specific to the first set of comparators to those specific to the second set
        sequences (np.ndarray):          sequences can be given as a numpy array for performance improvement, otherwise source_df[sequence_col] is used
        log2fc_values (np.ndarray):      log2fc values can be given as a numpy array for performance improvement, otherwise source_df[log2fc_col] is used
        significance_array (np.ndarray): significance calls (either as bool or str) can be given for performance improvement, otherwise source_df[significance_col] is sued
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
    same_length = np.all(np.char.str_len(sequences) == np.char.str_len(sequences[0]))
    if not same_length:
        raise ValueError(f"source_df[{sequence_col}] sequences have varying length, but are required to be equal in length")
    sequence_length = len(sequences[0])

    # Find where log2fc values pass each threshold
    passes_upper_positive = np.where(log2fc_values >= thresholds[0])
    passes_middle_positive = np.where(log2fc_values >= thresholds[1] & log2fc_values < thresholds[0])
    passes_middle_negative = np.where(log2fc_values <= thresholds[2] & log2fc_values > thresholds[3])
    passes_upper_negative = np.where(log2fc_values <= thresholds[3])

    # Get an array of points values matching the sequences
    points_values = np.zeros_like(log2fc_values)
    points_values[passes_upper_positive] = matching_points[0]
    points_values[passes_middle_positive] = matching_points[1]
    points_values[passes_middle_negative] = matching_points[2] * bias_multiplier
    points_values[passes_upper_negative] = matching_points[3] * bias_multiplier

    # Convert sequences to array of arrays, and do the same for matching points
    sequences_2d = np.array(list(sequences.view('|U1'))).reshape(sequences.shape + (-1,))
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

    return matrix_df

'''---------------------------------------------------------------------------------------------------------------------
                           Define function to apply/optimize weights and back-calculate scores 
   ------------------------------------------------------------------------------------------------------------------'''

def get_specificity_scores(sequences, log2fc_values, unweighted_matrix, position_weights = None):
    '''
    Function to back-calculate specificity scores on peptide sequences based on the generated specificity matrix

    Args:
        sequences (np.ndarray):                     the sequences to score, containing strings as their values
        log2fc_values (np.ndarray):                 the log2fc values for the sequences
        unweighted_matrix (pd.DataFrame):           the unweighted specificity matrix dataframe for getting residue values
        position_weights (np.ndarray):              an array of weights for the columns in the matrix dataframe

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
    sequences_2d = np.array(list(sequences.view('|U1'))).reshape(sequences.shape + (-1,))

    row_indices = weighted_specificity_matrix.index.get_indexer(sequences_2d.ravel()).reshape(sequences_2d.shape)
    column_indices = np.arange(sequence_length)[np.newaxis, :].repeat(sequence_count, axis=0)

    # Calculate the points
    matrix_array = weighted_specificity_matrix.values
    points_array = matrix_array[row_indices, column_indices]
    score_values = points_array.sum(axis=1)

    # Perform linear regression between log2fc values and scores
    model = LinearRegression()
    x_actual = score_values.reshape(-1, 1)
    y_actual = log2fc_values.reshape(-1, 1)
    model.fit(x_actual, y_actual)
    coef = model.coef_
    intercept = model.intercept_
    y_pred = model.predict(x_actual)
    r2 = r2_score(y_actual, y_pred)

    return (score_values, weighted_specificity_matrix, coef, intercept, r2)

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
                              "position_copies": None}

def main(source_df, comparator_info = None, specificity_params = None):
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
                                      --> position_copies (dict): idx > copies for permuting weights; sum(vals)=len(seq)

    Returns:
        final_results (tuple):     (scores (arr), weighted_matrix (df), coef, intercept, r2)
    '''

    output_df = source_df.copy()
    comparator_info = comparator_info or default_comparator_info.copy()
    specificity_params = specificity_params or default_specificity_params.copy()

    # Calculate least different pair of baits and corresponding log2fc for each row in output_df
    comparator_set_1, comparator_set_2 = comparator_info.get("comparator_set_1"), comparator_info.get("comparator_set_2")
    if comparator_set_1 is None or comparator_set_2 is None:
        comparator_set_1, comparator_set_2 = get_comparator_baits()
    least_different_results = least_different(output_df, comparator_set_1, comparator_set_2,
                                              return_series = True, return_df = True, in_place = False)
    least_different_values, least_different_baits, output_df = least_different_results

    # Get the multiplier to adjust for asymmetric distribution of bait specificities in the data
    thresholds = specificity_params.get("thresholds")
    if thresholds is None:
        thresholds = tuple(get_delimited_list("Please enter a comma-delimited list of thresholds, in descending order, as floats:  ", 4))
    matching_points = specificity_params.get("matching_points")
    if matching_points is None:
        matching_points = tuple(get_delimited_list(f"Please entner a comma-delimited list of associated points for each of the given thresholds ({thresholds}), in order, as floats:  ", 4))

    extreme_thresholds = (thresholds[0], thresholds[3])
    passes_col, pass_str = comparator_info.get("bait_pass_col"), comparator_info.get("pass_str")
    bias_multiplier = bias_ratio(output_df, least_different_values, extreme_thresholds, passes_col, pass_str)

    # Generate the unweighted specificity matrix
    sequence_col = comparator_info.get("seq_col")
    include_phospho = specificity_params.get("include_phospho")
    unweighted_matrix = make_specificity_matrix(thresholds, matching_points, bias_multiplier, source_df = source_df,
                                                sequence_col = sequence_col, log2fc_col = "least_different_log2fc",
                                                significance_col = passes_col, pass_str = pass_str,
                                                include_phospho = include_phospho)

    optimize_weights = specificity_params.get("optimize_weights")
    sequences = output_df[sequence_col].values
    log2fc_values = output_df["least_different_log2fc"].values
    sequence_length = len(sequences[0])

    if optimize_weights:
        # Determine optimal weights by maximizing the R2 value against a permuted array of weights arrays
        position_copies = specificity_params.get("position_copies")
        permuted_weights_array = permute_weights(sequence_length, position_copies)

        best_r2 = 0.0
        best_weights = np.ones(sequence_length)
        for permuted_weights in permuted_weights_array:
            _, _, _, _, current_r2 = get_specificity_scores(sequences, log2fc_values, unweighted_matrix, permuted_weights)
            if current_r2 > best_r2:
                best_weights = permuted_weights
                best_r2 = current_r2

        final_results = get_specificity_scores(sequences, log2fc_values, best_weights)
        final_score_values, final_weighted_matrix, final_coef, final_intercept, final_r2 = final_results
        print(f"Optimal matrix weights, {best_weights}, gave an R2 value of {final_r2} for the equation y={final_coef}x+{final_intercept}")

    else:
        # Use predefined weights, checked for correct dimensions and type
        predefined_weights = specificity_params.get("predefined_weights")
        if isinstance(predefined_weights, list) or isinstance(predefined_weights, tuple):
            predefined_weights = np.array(predefined_weights)
        elif not isinstance(predefined_weights, np.ndarray):
            raise ValueError(f"predefined_weights type is {type(predefined_weights)}, but must be np.ndarray, list, or tuple, of equal length to sequence_length")

        if len(predefined_weights) != sequence_length:
            raise ValueError(f"predefined_weights length ({len(predefined_weights)}) does not match sequence_length ({sequence_length})")

        final_results = get_specificity_scores(sequences, log2fc_values, unweighted_matrix, predefined_weights)
        final_score_values, final_weighted_matrix, final_coef, final_intercept, final_r2 = final_results
        print(f"The given matrix weights, {predefined_weights}, gave an R2 value of {final_r2} for the equation y={final_coef}x+{final_intercept}")

    return final_results