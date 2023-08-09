# This script conducts residue-residue pairwise analysis to generate context-aware SLiM matrices and back-calculated scores.

import numpy as np
import pandas as pd
import os
import pickle
from Matrix_Generator.ConditionalMatrix import ConditionalMatrices
from Matrix_Generator.ForbiddenMatrix import ForbiddenMatrix
from Matrix_Generator.conditional_scoring import apply_motif_scores
from Matrix_Generator.conditional_weight_optimization import optimize_conditional_weights
try:
    from Matrix_Generator.config_local import general_params, data_params, matrix_params, aa_equivalence_dict
except:
    from Matrix_Generator.config import general_params, data_params, matrix_params, aa_equivalence_dict

def find_optimal_threshold(motif_scores, passes_bools, contains_forbidden):
    # Helper function to find the optimal score threshold where FDR and FOR are approximately equal

    # Ignore peptides with forbidden residues, as they will be rejected later anyway
    non_forbidden_passes = passes_bools[~contains_forbidden]
    non_forbidden_scores = motif_scores[~contains_forbidden]

    # Iterate over the range of scores to find the optimal threshold
    min_score = non_forbidden_scores.min()
    max_score = non_forbidden_scores.max()
    score_range = np.linspace(start = min_score, stop = max_score, num = 500)
    best_score_threshold = None
    best_accuracy = 0
    for threshold in score_range:
        above_threshold = non_forbidden_scores >= threshold

        true_positives = np.sum(above_threshold & non_forbidden_passes)
        false_positives = np.sum(above_threshold & ~non_forbidden_passes)
        true_negatives = np.sum(~above_threshold & ~non_forbidden_passes)
        false_negatives = np.sum(~above_threshold & non_forbidden_passes)

        right_calls = true_positives + true_negatives
        wrong_calls =  false_positives + false_negatives
        accuracy = right_calls / (right_calls + wrong_calls)

        if accuracy > best_accuracy:
            best_score_threshold = threshold
            best_accuracy = accuracy

    if best_score_threshold is None:
        raise Exception("make_pairwise_matrices error: failed to find optimal threshold")

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

def main(input_df, general_params = general_params, data_params = data_params, matrix_params = matrix_params,
         aa_equivalence_dict = aa_equivalence_dict):
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
                                               data_params, matrix_params, aa_equivalence_dict)

    # Optionally optimize weights
    if matrix_params.get("optimize_weights"):
        possible_weights = matrix_params["possible_weights"]
        sequence_col = data_params["seq_col"]
        significance_col = data_params["bait_pass_col"]
        significant_str = data_params["pass_str"]
        convert_phospho = general_params["convert_phospho"]
        chunk_size = matrix_params["chunk_size"]
        fit_mode = matrix_params["fit_mode"]
        conditional_matrices = optimize_conditional_weights(input_df, motif_length, conditional_matrices, sequence_col,
                                                            significance_col, significant_str, possible_weights,
                                                            convert_phospho, chunk_size, fit_mode)
        conditional_matrices.save(output_folder)
    else:
        conditional_matrices.save(output_folder, save_weighted = False)

    if matrix_params.get("use_sigmoid"):
        conditional_matrices.save_sigmoid_plot(output_folder)

    # Apply motif scores
    seq_col = data_params.get("seq_col")
    score_col = data_params.get("dest_score_col")
    convert_phospho = not matrix_params.get("include_phospho")
    scored_df, motif_scores = apply_motif_scores(input_df, motif_length, conditional_matrices, seq_col = seq_col,
                                                 score_col = score_col, convert_phospho = convert_phospho,
                                                 add_residue_cols = True, use_weighted = True, return_array = True,
                                                 return_df = True)

    # Generate forbidden residues matrix and use it to check sequences
    sequences = scored_df[seq_col].to_numpy()
    bait_pass_col, pass_str = data_params.get("bait_pass_col"), data_params.get("pass_str")
    passes_bools = scored_df[bait_pass_col].to_numpy() == pass_str
    forbidden_matrix = ForbiddenMatrix(motif_length, sequences, passes_bools, aa_equivalence_dict, matrix_params,
                                       verbose = True)

    contains_forbidden = forbidden_matrix.predict_seqs(sequences)
    forbidden_residues_col = np.full(shape=len(scored_df), fill_value="-", dtype="<U10")
    forbidden_residues_col[contains_forbidden] = "Yes"
    scored_df["Forbidden_Residues"] = forbidden_residues_col
    forbidden_matrix.save(output_folder)

    # Find the optimal score threshold where FDR and FOR are approximately equal
    best_score_threshold = find_optimal_threshold(motif_scores, passes_bools, contains_forbidden)

    # Calculate and apply the final FDR and FOR
    scored_df, best_fdr, best_for = apply_final_rates(scored_df, motif_scores, best_score_threshold,
                                                      contains_forbidden, passes_bools)
    print(f"Conditional matrices metrics: FDR={best_fdr} | FOR={best_for} | threshold={best_score_threshold}")

    # Save ConditionalMatrices object for later use in motif_predictor
    conditional_matrices_path = os.path.join(output_folder, "conditional_matrices.pkl")
    with open(conditional_matrices_path, "wb") as f:
        pickle.dump(conditional_matrices, f)

    results = (best_fdr, best_for, best_score_threshold, scored_df)

    return results