# This script conducts residue-residue pairwise analysis to generate context-aware SLiM matrices and back-calculated scores.

import numpy as np
import pandas as pd
import os
import pickle
from general_utils.general_utils import unravel_seqs
from Matrix_Generator.ConditionalMatrix import ConditionalMatrices
from Matrix_Generator.ScoredPeptideResult import ScoredPeptideResult
from Matrix_Generator.train_score_nn import train_score_model
from Matrix_Generator.train_locally_connected import train_model as train_lcnn
try:
    from Matrix_Generator.config_local import general_params, data_params, matrix_params, aa_equivalence_dict
except:
    from Matrix_Generator.config import general_params, data_params, matrix_params, aa_equivalence_dict

def apply_motif_scores(input_df, conditional_matrices, slice_scores_subsets, seq_col = None,
                       convert_phospho = True, add_residue_cols = False, in_place = False, sequences_2d = None):
    '''
    Function to apply the score_seqs() function to all sequences in the source df and add residue cols for sorting

    Args:
        input_df (pd.DataFrame):                    df containing motif sequences to back-apply motif scores onto
        conditional_matrices (ConditionalMatrices): conditional weighted matrices for scoring peptides
        slice_scores_subsets (np.ndarray):          array of frame lengths for stratifying 2D score arrays
        seq_col (str): 			                    col in input_df with peptide seqs to score
        convert_phospho (bool):                     whether to convert phospho-residues to non-phospho before lookups
        add_residue_cols (bool):                    whether to add columns containing individual residue letters
        in_place (bool):                            whether to apply operations in-place; add_residue_cols not supported
        sequences_2d (np.ndarray):                  unravelled peptide sequences; optionally provide this upfront for
                                                    performance improvement in loops

    Returns:
        result (ScoredPeptideResult):               result object containing signal, suboptimal, and forbidden scores
        output_df (pd.DataFrame):                   input dataframe with scores and calls added
    '''

    # Get sequences only if needed; if sequences_2d is already provided, then sequences is not necessary
    if sequences_2d is None:
        seqs = input_df[seq_col].values.astype("<U")
        motif_length = len(seqs[0]) # assumes all sequences are the same length
        sequences_2d = unravel_seqs(seqs, motif_length, convert_phospho)
    else:
        motif_length = sequences_2d.shape[1]
        
    # Score the input data; the result is an instance of ScoredPeptideResult
    weights_exist = True if matrix_params.get("position_weights") is not None else False
    scored_result = conditional_matrices.score_peptides(sequences_2d, conditional_matrices, slice_scores_subsets,
                                                        use_weighted = weights_exist)

    # Construct the output dataframe
    output_df = input_df if in_place else input_df.copy()
    output_df = pd.concat([output_df, scored_result.scored_df], axis=1)

    # Optionally add position residue columns
    if add_residue_cols and not in_place:
        # Define the index where residue columns should be inserted
        current_cols = list(output_df.columns)
        insert_index = current_cols.index(seq_col) + 1

        # Assign residue columns
        residue_cols = ["#" + str(i) for i in np.arange(1, motif_length + 1)]
        residues_df = pd.DataFrame(sequences_2d, columns=residue_cols)
        output_df = pd.concat([output_df, residues_df], axis=1)

        # Define list of columns in the desired order
        final_columns = current_cols[0:insert_index]
        final_columns.extend(residue_cols)
        final_columns.extend(current_cols[insert_index:])

        # Reassign the output df with the ordered columns
        output_df = output_df[final_columns]

    elif add_residue_cols and in_place:
        raise Exception("apply_motif_scores error: in_place cannot be set to True when add_residue_cols is True")

    return scored_result, output_df

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
        weights_exist = True if matrix_params.get("position_weights") is not None else False
        conditional_matrices.save(output_folder, save_weighted = weights_exist)

        # Score the input data
        seq_col = data_params.get("seq_col")
        convert_phospho = not matrix_params.get("include_phospho")
        slice_scores_subsets = matrix_params.get("slice_scores_subsets")
        scored_result, output_df = apply_motif_scores(input_df, conditional_matrices, slice_scores_subsets, seq_col,
                                                      convert_phospho, add_residue_cols = True, in_place = False)

        # Cache the data
        with open(cached_path, "wb") as f:
            print(f"Cached conditional matrices were dumpted to {cached_path}")
            pickle.dump((conditional_matrices, scored_result, output_df), f)

    # Train a simple dense neural network based on the scoring results
    bait_pass_col = data_params["bait_pass_col"]
    pass_str = data_params["pass_str"]
    passes_strs = input_df[bait_pass_col].to_numpy()
    passes_bools = np.equal(passes_strs, pass_str)
    score_model, score_stats, score_preds = train_score_model(scored_result, passes_bools, save_path=output_folder)
    output_df["Score_Model_Predictions"] = score_preds
    print(f"Statistics for score interpretation dense neural network: ")
    for label, stat in score_stats.items():
        print(f"{label}: {stat:.4f}")

    # Also train a locally connected network based on chemical characteristics of residues for comparison
    lcnn_model, lcnn_stats, lcnn_preds = train_lcnn(scored_result.sequences_2d, passes_bools, save_path=output_folder)
    output_df["LCNN_Model_Predictions"] = lcnn_preds
    print(f"Statistics for locally connected neural network: ")
    for label, stat in lcnn_stats.items():
        print(f"{label}: {stat:.4f}")

    # Save ConditionalMatrices object for later use in motif_predictor
    conditional_matrices_path = os.path.join(output_folder, "conditional_matrices.pkl")
    with open(conditional_matrices_path, "wb") as f:
        pickle.dump(conditional_matrices, f)

    return (output_df, scored_result, score_model, lcnn_model)