# This script conducts residue-residue pairwise analysis to generate context-aware SLiM matrices and back-calculated scores.

import numpy as np
import pandas as pd
import os
from Matrix_Generator.ConditionalMatrix import ConditionalMatrices, default_data_params, default_matrix_params
from Matrix_Generator.conditional_scoring import apply_motif_scores
from general_utils.general_utils import save_dataframe, save_weighted_matrices
from general_utils.matrix_utils import add_matrix_weights
from general_utils.general_vars import aa_charac_dict
from general_utils.user_helper_functions import get_position_weights
from general_utils.statistics import apply_threshold

def apply_predefined_weights(input_df, position_weights, matrices_dict, slim_length, sequence_col, significance_col,
                             truth_val, score_col, matrix_output_folder, output_folder, make_calls):
    '''
    Function that applies and assesses a given set of weights against matrices and source data, for when weights will
    not be optimized automatically.

    Args:
        input_df (pd.DataFrame): 	the dataframe containing densitometry values for the peptides being analyzed
        position_weights (list):    list of position weights to use; length must be equal to slim_score
        matrices_dict (dict):       the dictionary of position-type rules --> unweighted matrices
        slim_length (int): 			the length of the motif being studied
        sequence_col (str):         the column in the dataframe that contains peptide sequences
        significance_col (str): 	the column in the dataframe that contains significance calls (Yes/No)
        truth_val (str):            the value to test against input_df[significance_col]
        score_col (str):            the name of the column where motif scores are found
        matrix_output_folder (str): the path to the folder where final matrices should be saved
        output_folder (str): 		the path to the folder where the output data should be saved
        make_calls (bool): 			whether to prompt the user to set a threshold for making positive/negative calls

    Returns:
        output_df (pd.DataFrame): 			the modified dataframe with scores applied
        predictive_value_df (pd.DataFrame): a dataframe containing sensitivity/specificity/PPV/NPV values for different
                                            score thresholds
    '''

    output_df = input_df.copy()

    # Get weights for positions along the motif sequence
    if position_weights is None:
        position_weights = get_position_weights(slim_length)

    # Apply the weights to the matrices
    weighted_matrices_dict = add_matrix_weights(np.array(position_weights), matrices_dict = matrices_dict)

    # Apply the motif scoring algorithm back onto the peptide sequences
    output_df = apply_motif_scores(output_df, weighted_matrices_dict, slim_length, sequence_col, score_col,
                                   add_residue_cols = True, in_place = False)

    # Use thresholding to declare true/false positives/negatives in the peptide sequences
    if make_calls:
        results = apply_threshold(output_df, sig_col = significance_col, score_col = score_col, truth_value = truth_val)
        output_df, selected_threshold, predictive_value_df = results
    else:
        selected_threshold = None
        predictive_value_df = apply_threshold(output_df, sig_col = significance_col, score_col = score_col,
                                              truth_value = truth_val, return_predictive_only = True)

    # Save the weighted matrices and scored data
    save_weighted_matrices(weighted_matrices_dict, matrix_output_folder, save_pickled_dict = True)
    scored_data_filename = "pairwise_scored_data_thres" + str(selected_threshold) + ".csv"
    save_dataframe(output_df, output_folder, scored_data_filename)

    return output_df, weighted_matrices_dict, predictive_value_df


default_general_params = {"percentiles_dict": None,
                          "motif_length": None,
                          "always_allowed_dict": None,
                          "position_weights": None,
                          "output_folder": None,
                          "make_calls": True,
                          "optimize_weights": False,
                          "possible_weights": None,
                          "aa_charac_dict": aa_charac_dict,
                          "convert_phospho": True}

def main(input_df, general_params = None, data_params = None, matrix_params = None, verbose = True):
    '''
    Main function for making pairwise position-weighted matrices

    Args:
        input_df (pd.DataFrame): 	the dataframe containing densitometry values for the peptides being analyzed
        general_params (dict):      dictionary of general parameters:
                                            --> percentiles_dict (dict): input data percentile --> mean signal value
                                            --> motif_length (int): the length of the motif being studied
                                            --> always_allowed_dict (dict): position --> always-permitted residues
                                            --> position_weights (list): predefined weight values, if not optimizing
                                            --> output_folder (str): path where the output data should be saved
                                            --> make_calls (bool): whether to set thresholds and making +/- calls
                                            --> optimize_weights (bool): whether to optimize weights to maximize FDR/FOR
                                            --> possible_weights (list): list of arrays of possible weights at each position
                                            --> aa_charac_dict (dict): dictionary of chemical_characteristic --> [AAs]
                                            --> convert_phospho (bool): whether to convert phospho-residues to unphospho
        data_params (dict):         dictionary of parameters describing the source_dataframe structure, used in matrix-building:
                                            --> bait (str): the bait to use for matrix generation; defaults to best if left blank
                                            --> bait_signal_col_marker (str): keyword that marks columns in source_dataframe that
                                                contain signal values; required only if bait is given
                                            --> best_signal_col (str): column name with best signal values; used if bait is None
                                            --> bait_pass_col (str): column name with pass/fail information
                                            --> pass_str (str): the string representing a pass in bait_pass_col, e.g. "Yes"
                                            --> seq_col (str): column name containing peptide sequences as strings
        matrix_params (dict):       dictionary of parameters that affect matrix-building behaviour, used in matrix-building:
                                            --> thresholds_points_dict (dict): dictionary where threshold_value --> points_value
                                            --> included_residues (list): the residues included for the current type-position rule
                                            --> amino_acids (tuple): the alphabet of amino acids to use when constructing the matrix
                                            --> min_members (int): the minimum number of peptides that must follow the current
                                                type-position rule for the matrix to be built
                                            --> position_for_filtering (int): the position for the type-position rule being assessed
                                            --> clear_filtering_column (bool): whether to set values in the filtering column to zero
        verbose (bool): 			whether to display user feedback and debugging information

    Returns:
        output_df (pd.DataFrame): 			the modified dataframe with scores applied
        best_weights (np.ndarray):          only returned if optimize_weights is set to True; it is the best detected
                                            weights leading to optimal FDR/FOR
        predictive_value_df (pd.DataFrame): only returned if optimize_weights is set to False; it is a dataframe 
                                            containing sensitivity/specificity/PPV/NPV values for different score thres
    '''

    if general_params is None:
        general_params = default_general_params.copy()
    if data_params is None:
        data_params = default_data_params.copy()
    if matrix_params is None:
        matrix_params = default_matrix_params.copy()

    # Declare the output folder for saving pairwise weighted matrices
    output_folder = general_params.get("output_folder")
    if output_folder is None:
        output_folder = os.getcwd()
    matrix_output_folder = os.path.join(output_folder, "Pairwise_Matrices")

    # Obtain the dictionary of matrices that have not yet been weighted
    percentiles_dict = general_params.get("percentiles_dict")
    motif_length = general_params.get("motif_length")
    aa_charac_dict = general_params.get("aa_charac_dict")
    conditional_matrices = ConditionalMatrices(motif_length, input_df, percentiles_dict, aa_charac_dict, data_params, matrix_params)

    # Apply weights to the generated matrices, or find optimal weights
    optimize_weights = general_params.get("optimize_weights")
    sequence_col = data_params.get("seq_col")
    significance_col = data_params.get("bait_pass_col")
    significant_str = data_params.get("pass_str")
    score_col = data_params.get("dest_score_col")
    output_statistics = {}

    if optimize_weights:
        # Find the optimal weights that produce the lowest FDR/FOR pair
        convert_phospho = general_params.get("convert_phospho")
        possible_weights = general_params.get("possible_weights")
        results_tuple = find_optimal_weights(input_df, motif_length, conditional_matrices, sequence_col,
                                             significance_col, significant_str, score_col, matrix_output_folder,
                                             output_folder, possible_weights, aa_charac_dict, convert_phospho,
                                             save_pickled_matrix_dict = True)
        best_fdr, best_for, best_score_threshold, best_weights, weighted_matrices_dict, scored_df = results_tuple
        position_weights = best_weights
        output_statistics["FDR"] = best_fdr
        output_statistics["FOR"] = best_for
        output_statistics["cutoff_threshold"] = best_score_threshold
        output_statistics["position_weights"] = best_weights

    else:
        # Apply predefined weights and calculate predictive values
        position_weights = general_params.get("position_weights")
        make_calls = general_params.get("make_calls")
        results = apply_predefined_weights(input_df, position_weights, matrices_dict, motif_length, sequence_col,
                                           significance_col, significant_str, score_col, matrix_output_folder,
                                           output_folder, make_calls)
        scored_df, weighted_matrices_dict, predictive_value_df = results
        output_statistics["predictive_value_df"] = predictive_value_df

    return (scored_df, position_weights, weighted_matrices_dict, output_statistics)
