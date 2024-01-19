# This script conducts residue-residue pairwise analysis to generate context-aware SLiM matrices and back-calculated scores.

import numpy as np
import pandas as pd
import os
import pickle
from general_utils.general_utils import unravel_seqs
from Matrix_Generator.ConditionalMatrix import ConditionalMatrices
from Matrix_Generator.ScoredPeptideResult import ScoredPeptideResult
try:
    from Matrix_Generator.config_local import general_params, data_params, matrix_params, aa_equivalence_dict
except:
    from Matrix_Generator.config import general_params, data_params, matrix_params, aa_equivalence_dict

def apply_motif_scores(input_df, conditional_matrices, slice_scores_subsets = None, actual_truths = None,
                       signal_values = None, seq_col = None, convert_phospho = True, add_residue_cols = False,
                       in_place = False, seqs_2d = None, precision_recall_path = None, coefficients_path = None):
    '''
    Function to apply the score_seqs() function to all sequences in the source df and add residue cols for sorting

    Args:
        input_df (pd.DataFrame):                    df containing motif sequences to back-apply motif scores onto
        conditional_matrices (ConditionalMatrices): conditional weighted matrices for scoring peptides
        slice_scores_subsets (np.ndarray):          array of frame lengths for stratifying 2D score arrays
        actual_truths (np.ndarray):                 array of experimentally confirmed truth values of input peptides
        signal_values (np.ndarray):                 array of binding signal values for peptides against protein bait(s)
        seq_col (str): 			                    col in input_df with peptide seqs to score
        convert_phospho (bool):                     whether to convert phospho-residues to non-phospho before lookups
        add_residue_cols (bool):                    whether to add columns containing individual residue letters
        in_place (bool):                            whether to apply operations in-place; add_residue_cols not supported
        seqs_2d (np.ndarray):                  unravelled peptide sequences; optionally provide this upfront for
                                                    performance improvement in loops
        precision_recall_path (str):                output file path for saving the precision/recall graph
        coefficients_path (str):                    output file path for saving score standardization coefficients into

    Returns:
        result (ScoredPeptideResult):               result object containing signal, suboptimal, and forbidden scores
        output_df (pd.DataFrame):                   input dataframe with scores and calls added
    '''

    # Get sequences only if needed; if seqs_2d is already provided, then sequences is not necessary
    if seqs_2d is None:
        seqs = input_df[seq_col].values.astype("<U")
        motif_length = len(seqs[0]) # assumes all sequences are the same length
        seqs_2d = unravel_seqs(seqs, motif_length, convert_phospho)
    else:
        motif_length = seqs_2d.shape[1]
        
    # Score the input data; the result is an instance of ScoredPeptideResult
    result = conditional_matrices.optimize_scoring_weights(seqs_2d, actual_truths, signal_values, slice_scores_subsets,
                                                           precision_recall_path, coefficients_path)

    # Construct the output dataframe
    output_df = input_df if in_place else input_df.copy()
    output_df = pd.concat([output_df, result.scored_df], axis=1)

    # Optionally add position residue columns
    if add_residue_cols and not in_place:
        # Define the index where residue columns should be inserted
        current_cols = list(output_df.columns)
        insert_index = current_cols.index(seq_col) + 1

        # Assign residue columns
        residue_cols = ["#" + str(i) for i in np.arange(1, motif_length + 1)]
        residues_df = pd.DataFrame(seqs_2d, columns=residue_cols)
        output_df = pd.concat([output_df, residues_df], axis=1)

        # Define list of columns in the desired order
        final_columns = current_cols[0:insert_index]
        final_columns.extend(residue_cols)
        final_columns.extend(current_cols[insert_index:])

        # Reassign the output df with the ordered columns
        output_df = output_df[final_columns]

    elif add_residue_cols and in_place:
        raise Exception("apply_motif_scores error: in_place cannot be set to True when add_residue_cols is True")

    return (result, output_df, conditional_matrices)

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
        conditional_matrices = ConditionalMatrices(motif_length, input_df, percentiles_dict, aa_charac_dict,
                                                   data_params, matrix_params)

        # Score the input data
        seq_col = data_params.get("seq_col")
        convert_phospho = not matrix_params.get("include_phospho")
        slice_scores_subsets = matrix_params.get("slice_scores_subsets")
        pass_str = data_params.get("pass_str")
        pass_col = data_params.get("bait_pass_col")
        pass_values = input_df[pass_col].to_numpy()
        actual_truths = np.equal(pass_values, pass_str)

        bait_signal_cols = list(set([col for cols in data_params["bait_cols_dict"].values() for col in cols]))
        bait_signal_values = input_df[bait_signal_cols].to_numpy()
        mean_signal_values = bait_signal_values.mean(axis=1)

        precision_recall_path = os.path.join(output_folder, "precision_recall_graph.pdf")
        scoring_output_tuple = apply_motif_scores(input_df, conditional_matrices, slice_scores_subsets, actual_truths,
                                                  mean_signal_values, seq_col, convert_phospho, add_residue_cols = True,
                                                  in_place = False, precision_recall_path = precision_recall_path,
                                                  coefficients_path = output_folder)
        scored_result, output_df, conditional_matrices = scoring_output_tuple

        conditional_matrices.save(output_folder, save_weighted = True)

        # Cache the data
        with open(cached_path, "wb") as f:
            print(f"Cached conditional matrices were dumped to {cached_path}")
            pickle.dump((conditional_matrices, scored_result, output_df), f)

    # Save ConditionalMatrices object for later use in motif_predictor
    conditional_matrices_path = os.path.join(output_folder, "conditional_matrices.pkl")
    with open(conditional_matrices_path, "wb") as f:
        pickle.dump(conditional_matrices, f)

    # Save weights for later use in motif_predictor
    weights_path = os.path.join(output_folder, "conditional_weights_tuple.pkl")
    weights_tuple = (scored_result.positives_weights,
                     scored_result.suboptimals_weights,
                     scored_result.forbiddens_weights)
    with open(weights_path, "wb") as f:
        pickle.dump(weights_tuple, f)

    return (output_df, scored_result)