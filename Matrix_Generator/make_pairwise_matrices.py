# This script conducts residue-residue pairwise analysis to generate context-aware SLiM matrices and back-calculated scores.

import numpy as np
import pandas as pd
import os
from Matrix_Generator.config import general_params, data_params, matrix_params
from Matrix_Generator.ConditionalMatrix import ConditionalMatrices
from conditional_thresholds_optimization import find_optimal_thresholds

def main(input_df, general_params = general_params, data_params = data_params, matrix_params = matrix_params):
    '''
    Main function for making pairwise position-weighted matrices

    Args:
        input_df (pd.DataFrame): 	the dataframe containing densitometry values for the peptides being analyzed
        general_params (dict):      dictionary of general parameters:
                                            --> percentiles_dict (dict): input data percentile --> mean signal value
                                            --> motif_length (int): the length of the motif being studied
                                            --> output_folder (str): path where the output data should be saved
                                            --> make_calls (bool): whether to set thresholds and making +/- calls
                                            --> threshold_steps: number of steps between 0 and 1 for optimizing aa thres
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

    Returns:
        output_df (pd.DataFrame): 			the modified dataframe with scores applied
        best_weights (np.ndarray):          only returned if optimize_weights is set to True; it is the best detected
                                            weights leading to optimal FDR/FOR
        predictive_value_df (pd.DataFrame): only returned if optimize_weights is set to False; it is a dataframe 
                                            containing sensitivity/specificity/PPV/NPV values for different score thres
    '''

    # Declare the output folder for saving pairwise weighted matrices
    output_folder = general_params.get("output_folder")
    if output_folder is None:
        output_folder = os.getcwd()
    matrix_output_folder = os.path.join(general_params.get("output_folder"), "Pairwise_Matrices")

    # Obtain the dictionary of matrices that have not yet been weighted
    percentiles_dict = general_params.get("percentiles_dict")
    motif_length = general_params.get("motif_length")
    aa_charac_dict = general_params.get("aa_charac_dict")
    conditional_matrices = ConditionalMatrices(motif_length, input_df, percentiles_dict, aa_charac_dict, data_params, matrix_params)

    # Apply weights to the generated matrices, or find optimal weights
    sequence_col = data_params.get("seq_col")
    significance_col = data_params.get("bait_pass_col")
    significant_str = data_params.get("pass_str")
    score_col = data_params.get("dest_score_col")

    # Find the optimal residue points thresholds that produce the lowest FDR/FOR pair
    convert_phospho = general_params.get("convert_phospho")
    position_thresholds = general_params.get("position_thresholds")
    results = find_optimal_thresholds(input_df, motif_length, conditional_matrices, sequence_col, significance_col,
                                      significant_str, score_col, matrix_output_folder, output_folder,
                                      position_thresholds, convert_phospho, save_pickled_matrix_dict = True)

    return results