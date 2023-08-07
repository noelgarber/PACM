#This script takes protein sequences and computes their motif scores based on the results of make_pairwise_matrices.py

import numpy as np
import pandas as pd
import pickle
from functools import partial
from Motif_Predictor.predictor_config import predictor_params
from Matrix_Generator.conditional_scoring import apply_motif_scores
from general_utils.general_utils import finite_sorted_indices, add_number_suffix

if predictor_params["compare_classical_method"]:
    from Motif_Predictor.classical_method import classical_method

def scan_protein_seq(protein_seq, conditional_matrices, predictor_params = predictor_params):
    '''

    Args:
        protein_seq (str):                          full length protein sequence to score
        conditional_matrices (ConditionalMatrices): object containing conditional weighted matrices
        predictor_params (dict):                    dictionary of user-defined parameters from predictor_config.py

    Returns:
        best_score, best_motif, second_best_score, second_best_motif
    '''

    # Get necessary arguments
    motif_length = predictor_params["motif_length"]
    use_weighted = predictor_params["use_weighted"]
    convert_phospho = predictor_params["convert_phospho"]
    return_count = predictor_params["return_count"]

    # Get N-term and C-term trailing residue info
    leading_glycines = np.repeat("G", predictor_params["leading_glycines"])
    trailing_glycines = np.repeat("G", predictor_params["trailing_glycines"])

    # Determine whether protein seq is valid
    valid_seq = True
    if not isinstance(protein_seq, str):
        valid_seq = False
    elif len(protein_seq) < motif_length:
        valid_seq = False
    elif "*" in protein_seq[:-1]:
        valid_seq = False

    # Extract protein sequence into overlapping motif-sized segments with step size of 1
    if valid_seq:
        # Remove stop asterisk if present
        if protein_seq[-1] == "*":
            protein_seq = protein_seq[:-1]

        seq_array = np.array(list(protein_seq))

        seq_array = np.concatenate([leading_glycines, seq_array, trailing_glycines])
        slice_indices = np.arange(len(seq_array) - motif_length + 1)[:, np.newaxis] + np.arange(motif_length)
        sliced_seqs_2d = seq_array[slice_indices]

        # Enforce position rules
        enforced_position_rules = predictor_params.get("enforced_position_rules")
        if enforced_position_rules is not None:
            for position_index, allowed_residues in enforced_position_rules.items():
                column_residues = sliced_seqs_2d[:,position_index]
                residues_allowed = np.isin(column_residues, allowed_residues)
                sliced_seqs_2d = sliced_seqs_2d[residues_allowed]

        # After rules have been enforced, replace uncertain residues (X) with G (no side chain) for purposes of scoring
        missing_residues = "X" in sliced_seqs_2d
        cleaned_sliced_2d = sliced_seqs_2d.copy()
        if missing_residues:
            cleaned_sliced_2d[cleaned_sliced_2d == "X"] = "G"

        # Optionally replace selenocysteine with cysteine for scoring purposes
        replace_selenocysteine = predictor_params["replace_selenocysteine"]
        if replace_selenocysteine and "U" in sliced_seqs_2d:
            cleaned_sliced_2d[cleaned_sliced_2d == "U"] = "C"

        # Apply motif scores
        motif_scores = apply_motif_scores(None, motif_length, conditional_matrices, cleaned_sliced_2d, convert_phospho,
                                          use_weighted, return_array = True, return_2d = False, return_df = False)
        sorted_score_indices = finite_sorted_indices(motif_scores)
        sorted_motifs = []
        sorted_score_values = []
        for i in np.arange(return_count):
            if i < len(motif_scores):
                next_best_idx = sorted_score_indices[i]
                next_best_score = motif_scores[next_best_idx]
                next_best_motif = "".join(sliced_seqs_2d[next_best_idx])
                sorted_motifs.append(next_best_motif)
                sorted_score_values.append(next_best_score)
            else:
                sorted_motifs.append(" " * motif_length)
                sorted_score_values.append(np.nan)

    else:
        sorted_motifs = np.repeat("", return_count)
        sorted_score_values = np.repeat(np.nan, return_count)

    sorted_motifs = np.array(sorted_motifs)
    sorted_score_values = np.array(sorted_score_values)

    return sorted_motifs, sorted_score_values

def score_protein_seqs(protein_seqs_df, predictor_params = predictor_params):
    '''
    Top level function to score protein sequences based on conditional matrices

    Args:
        protein_seqs_df (pd.DataFrame): protein sequences dataframe
        predictor_params (dict):        dictionary of user-defined parameters from predictor_config.py

    Returns:
        protein_seqs_df (pd.DataFrame): dataframe with protein sequences and found motifs
        motif_col_names (list):         list of column names containing motif sequences for each protein
    '''

    # Get protein sequences to score
    protein_seqs_list = protein_seqs_df["Sequence"].to_list()

    # Load ConditionalMatrices object to be used in scoring
    conditional_matrices_path = predictor_params["conditional_matrices_path"]
    with open(conditional_matrices_path, "rb") as f:
        conditional_matrices = pickle.load(f)

    # Generate columns for the number of motifs that will be returned per protein
    return_count = predictor_params["return_count"]
    compare_classical_method = predictor_params["compare_classical_method"]

    ordered_motifs_cols = []
    ordered_scores_cols = []
    motif_col_names = []
    score_col_names = []
    if compare_classical_method:
        classical_motifs_cols = []
        classical_scores_cols = []

    for i in np.arange(return_count):
        ordered_motifs_cols.append([])
        ordered_scores_cols.append([])
        if compare_classical_method:
            classical_motifs_cols.append([])
            classical_scores_cols.append([])
        suffix_number = add_number_suffix(i+1)
        motif_col_names.append(suffix_number+"_motif")
        score_col_names.append(suffix_number+"_motif_score")

    # Loop over the protein sequences to score them
    scan_seq_partial = partial(scan_protein_seq, conditional_matrices = conditional_matrices,
                               predictor_params = predictor_params)

    for i, protein_seq in enumerate(protein_seqs_list):
        print(f"Current entry: #{i}")
        # Score the protein sequence using conditional matrices
        sorted_motifs, sorted_score_values = scan_seq_partial(protein_seq)
        for j, (motif, score) in enumerate(zip(sorted_motifs, sorted_score_values)):
            ordered_motifs_cols[j].append(motif)
            ordered_scores_cols[j].append(score)
        #print(f"\tNovel method found motifs {sorted_motifs} with scores {sorted_score_values}")

        # Optionally score the sequence using a classical method for comparison
        if compare_classical_method:
            classical_motifs, classical_scores = classical_method(protein_seq, predictor_params)
            for j, (classical_motif, classical_score) in enumerate(zip(classical_motifs, classical_scores)):
                classical_motifs_cols[j].append(classical_motif)
                classical_scores_cols[j].append(classical_score)
            #print(f"\tClassical method found motifs {classical_motifs} with scores {classical_scores}")

    # Apply motifs and scores as columns to the dataframe
    zipped_cols = zip(ordered_motifs_cols, motif_col_names, ordered_scores_cols, score_col_names)
    for ordered_motifs_col, motif_col_name, ordered_scores_col, score_col_name in zipped_cols:
        if compare_classical_method:
            motif_col_name = "Novel_" + motif_col_name
            score_col_name = "Novel_" + score_col_name
        protein_seqs_df[motif_col_name] = ordered_motifs_col
        protein_seqs_df[score_col_name] = ordered_scores_col

    # Optionally apply classical motifs and scores as columns if they were generated
    if compare_classical_method:
        zipped_classical_cols = zip(classical_motifs_cols, motif_col_names, classical_scores_cols, score_col_names)
        for classical_motif_col, motif_col_name, classical_scores_col, score_col_name in zipped_classical_cols:
            motif_col_name = "Classical_" + motif_col_name
            score_col_name = "Classical_" + score_col_name
            protein_seqs_df[motif_col_name] = classical_motif_col
            protein_seqs_df[score_col_name] = classical_scores_col

    return protein_seqs_df, motif_col_names