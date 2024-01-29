# This script conducts residue-residue pairwise analysis to generate context-aware SLiM matrices and back-calculated scores.

import numpy as np
import pandas as pd
import os
import pickle
from Matrix_Generator.ConditionalMatrix import ConditionalMatrices
from sklearn.model_selection import train_test_split
try:
    from Matrix_Generator.config_local import general_params, data_params, matrix_params, aa_equivalence_dict
except:
    from Matrix_Generator.config import general_params, data_params, matrix_params, aa_equivalence_dict

def main(input_df, general_params = general_params, data_params = data_params, matrix_params = matrix_params):
    '''
    Main function for making pairwise position-weighted matrices

    Args:
        input_df (pd.DataFrame): 	the dataframe containing densitometry values for the peptides being analyzed
        general_params (dict):      dictionary of general conditional matrix params described in config.py
        data_params (dict):         dictionary of data-specific params described in config.py
        matrix_params (dict):       dictionary of matrix-specific params described in config.py

    Returns:
        best_fdr (float):             final false discovery rate
        best_for (float):             final false omission rate
        best_score_threshold (float): optimal threshold for peptide scoring
        scored_df (pd.DataFrame):     scored version of the input dataframe
    '''

    # Declare the output folder for saving pairwise weighted matrices
    output_folder = general_params.get("output_folder")
    if output_folder is None:
        output_folder = os.getcwd()

    # Check if there is a cached version - if there is, use this rather than generating conditional matrices again
    cached_path = os.path.join(output_folder, "cached_conditional_matrices.pkl")
    if os.path.isfile(cached_path):
        with open(cached_path, "rb") as f:
            print("Found cached conditional matrices! Loading...")
            conditional_matrices, scored_result, output_df = pickle.load(f)
    else:
        # Generate the conditional position-weighted matrices
        percentiles_dict = general_params.get("percentiles_dict")
        motif_length = general_params.get("motif_length")
        aa_charac_dict = general_params.get("aa_charac_dict")

        split_data = matrix_params["train_test_split"]
        retrain_with_all = matrix_params["retrain_with_all"]

        if split_data:
            train_df, test_df = train_test_split(input_df, test_size=0.1)
            train_df = train_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)
            conditional_matrices = ConditionalMatrices(motif_length, train_df, percentiles_dict, aa_charac_dict,
                                                       output_folder, data_params, matrix_params, test_df=test_df)
            if retrain_with_all:
                conditional_matrices = ConditionalMatrices(motif_length, input_df, percentiles_dict, aa_charac_dict,
                                                           output_folder, data_params, matrix_params)
        else:
            conditional_matrices = ConditionalMatrices(motif_length, input_df, percentiles_dict, aa_charac_dict,
                                                       output_folder, data_params, matrix_params)

        scored_result = conditional_matrices.scored_result
        output_df = conditional_matrices.output_df

        conditional_matrices.save(output_folder, save_weighted = True)

        # Cache the data
        with open(cached_path, "wb") as f:
            print(f"Cached conditional matrices were dumped to {cached_path}")
            pickle.dump((conditional_matrices, scored_result, output_df), f)

    # Save ConditionalMatrices object for later use in motif_predictor
    conditional_matrices_path = os.path.join(output_folder, "conditional_matrices.pkl")
    with open(conditional_matrices_path, "wb") as f:
        pickle.dump(conditional_matrices, f)

    return (output_df, scored_result)