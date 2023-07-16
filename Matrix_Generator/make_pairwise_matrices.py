# This script conducts residue-residue pairwise analysis to generate context-aware SLiM matrices and back-calculated scores.

import numpy as np
import pandas as pd
import os
from Matrix_Generator.config import general_params, data_params, matrix_params
from Matrix_Generator.ConditionalMatrix import ConditionalMatrices
from Matrix_Generator.ForbiddenMatrix import ForbiddenMatrix
from Matrix_Generator.conditional_scoring import apply_motif_scores

def find_optimal_rates(motif_scores, passes_bools, contains_forbidden):
    # Helper function to find the optimal score threshold where FDR and FOR are approximately equal

    # Ignore peptides with forbidden residues, as they will be rejected later anyway
    non_forbidden_passes = passes_bools[~contains_forbidden]
    non_forbidden_scores = motif_scores[~contains_forbidden]

    # Iterate over the range of scores to find the optimal threshold
    score_range = np.linspace(non_forbidden_scores.min(), non_forbidden_scores.max(), 500)
    best_score_threshold = None
    best_rate_delta = 1
    for threshold in score_range:
        above_threshold = non_forbidden_passes >= threshold

        false_positives = np.sum(above_threshold & ~passes_bools)
        positive_calls = np.sum(above_threshold)

        false_negatives = np.sum(~above_threshold & passes_bools)
        negative_calls = np.sum(~above_threshold)

        if positive_calls > 0 and negative_calls > 0:
            fdr_val = false_positives / positive_calls
            for_val = false_negatives / negative_calls
            rate_delta = abs(fdr_val - for_val)
            if rate_delta < best_rate_delta:
                best_score_threshold = threshold
                best_rate_delta = rate_delta

    if best_score_threshold is None:
        raise Exception("make_pairwise_matrices error: failed to find optimal threshold with defined FDR and FOR")

    return best_score_threshold

def apply_final_rates(scored_df, motif_scores, best_score_threshold, contains_forbidden, passes_bools):
    # Helper function to calculate the final FDR and FOR

    above_best_threshold = motif_scores >= best_score_threshold
    predicted_positive = np.logical_and(above_best_threshold, ~contains_forbidden)

    true_positives = np.logical_and(predicted_positive, passes_bools)
    false_positives = np.logical_and(predicted_positive, ~passes_bools)
    FP_count = np.sum(false_positives)
    positive_calls = np.sum(predicted_positive)
    best_fdr = FP_count / positive_calls if positive_calls > 0 else np.inf

    true_negatives = np.logical_and(~predicted_positive, ~passes_bools)
    false_negatives = np.logical_and(~predicted_positive, passes_bools)
    FN_count = np.sum(false_negatives)
    negative_calls = np.sum(~predicted_positive)
    best_for = FN_count / negative_calls if negative_calls > 0 else np.inf

    # Assign binary classifications
    classifications = np.full(shape=len(scored_df), fill_value="-", dtype="<U10")
    classifications[true_positives] = "TP"
    classifications[false_positives] = "FP"
    classifications[true_negatives] = "TN"
    classifications[false_negatives] = "FN"
    scored_df["Classification"] = classifications

    return scored_df, best_fdr, best_for

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

    # Generate the conditional position-weighted matrices
    percentiles_dict = general_params.get("percentiles_dict")
    motif_length = general_params.get("motif_length")
    aa_charac_dict = general_params.get("aa_charac_dict")
    conditional_matrices = ConditionalMatrices(motif_length, input_df, percentiles_dict, aa_charac_dict,
                                               data_params, matrix_params)
    conditional_matrices.save(output_folder)

    # Apply motif scores
    seq_col = data_params.get("seq_col")
    score_col = data_params.get("dest_score_col")
    convert_phospho = not matrix_params.get("include_phospho")
    scored_df, motif_scores = apply_motif_scores(input_df, motif_length, conditional_matrices, seq_col = seq_col,
                                                 score_col = score_col, convert_phospho = convert_phospho,
                                                 add_residue_cols = True, use_weighted = True, return_array = True,
                                                 return_df = True)

    # Generate forbidden residues matrix and use it to check sequences
    forbidden_matrix = ForbiddenMatrix(motif_length, input_df, data_params, matrix_params)
    sequences = scored_df[seq_col].to_numpy()
    contains_forbidden = forbidden_matrix.predict_seqs(sequences)
    forbidden_residues_col = np.full(shape=len(scored_df), fill_value="-", dtype="<U10")
    forbidden_residues_col[contains_forbidden] = "Yes"
    scored_df["Forbidden_Residues"] = forbidden_residues_col
    forbidden_matrix.save(output_folder)

    # Find the optimal score threshold where FDR and FOR are approximately equal
    bait_pass_col, pass_str = data_params.get("bait_pass_col"), data_params.get("pass_str")
    passes_bools = scored_df[bait_pass_col] == pass_str
    best_score_threshold = find_optimal_rates(motif_scores, passes_bools, contains_forbidden)

    # Calculate and apply the final FDR and FOR
    scored_df, best_fdr, best_for = apply_final_rates(scored_df, motif_scores, best_score_threshold,
                                                      contains_forbidden, passes_bools)

    results = (best_fdr, best_for, best_score_threshold, scored_df)

    return results