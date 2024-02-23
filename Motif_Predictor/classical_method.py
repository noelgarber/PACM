
import os
import numpy as np
import pandas as pd
from Motif_Predictor.predictor_config import predictor_params

''' ---------------------------------------------------------------------------------------------------------------- 
            It is sometimes useful to compare the predictions of the PACM workflow against known algorithms. 
            The classical_method() function is a container for such known algorithms if you wish to 
            perform that analysis in parallel. It is completely OPTIONAL.
    ---------------------------------------------------------------------------------------------------------------- '''

try:
    classical_matrix = pd.read_csv("Motif_Predictor/classical_matrix.csv", index_col = 0)
except:
    raise Exception("Failed to load classical_matrix.csv, which is required when using classical_method()")

motif_length = 15

def classical_motif_method(motif_seqs, classical_matrix = classical_matrix):
    '''
    Example of a parallel classical scoring method to employ alongside the new one; replace with your method as needed

    Args:
        motif_seqs (list|np.ndarray): motif sequences of equal length; if given as a 2D array, each row is a motif

    Returns:
        total_points_motifs (np.ndarray): array of corresponding classical motif scores for the input sequences
    '''

    if not isinstance(motif_seqs, np.ndarray):
        motif_seqs = np.array(motif_seqs)

    if motif_seqs.ndim == 1:
        correct_lengths = np.array([len(seq) == motif_length for seq in motif_seqs])
        valid_indices = np.where(correct_lengths)[0]
        valid_motif_seqs = motif_seqs[valid_indices]
        valid_motif_seqs_2d = np.array([list(seq) for seq in valid_motif_seqs])
    elif motif_seqs.ndim == 2:
        valid_motif_seqs_2d = motif_seqs
        if motif_seqs.shape[1] != motif_length:
            raise ValueError(f"motif_seqs axis=1 shape is {motif_seqs.shape[1]}, but should be {motif_length}")
        valid_indices = np.arange(len(valid_motif_seqs_2d))
    else:
        raise Exception(f"motif_seqs ndim={motif_seqs.ndim}, but ndim must be either 1 or 2")

    tract_seqs = valid_motif_seqs_2d[:,0:6]
    core_seqs = valid_motif_seqs_2d[:,6:13]

    # Calculate charges on tract sequences
    tract_residue_charges = np.zeros(shape=tract_seqs.shape, dtype=float)
    tract_residue_charges[tract_seqs == "D"] = -1
    tract_residue_charges[tract_seqs == "E"] = -1
    tract_residue_charges[tract_seqs == "S"] = -0.5
    tract_residue_charges[tract_seqs == "T"] = -0.5
    tract_residue_charges[tract_seqs == "R"] = 1
    tract_residue_charges[tract_seqs == "K"] = 1

    tract_charges = tract_residue_charges.sum(axis=1)

    # Translate tract charges into points values
    tract_points = np.full(shape=tract_charges.shape, fill_value=np.inf, dtype=float)
    tract_points[tract_charges <= -4] = 0
    tract_points[np.logical_and(tract_charges <= -3, tract_charges > -4)] = 0.5
    tract_points[np.logical_and(tract_charges <= -2, tract_charges > -3)] = 1
    tract_points[tract_charges > -2] = 1.5

    # Apply matrix to core sequences
    core_points = np.zeros(shape=core_seqs.shape, dtype=float)
    for col_index in np.arange(core_seqs.shape[1]):
        col_residues = core_seqs[:,col_index]
        col_points = np.full(shape=len(col_residues), fill_value=np.inf, dtype=float)

        matrix_row_indices = classical_matrix.index.get_indexer_for(col_residues)
        matrix_column = classical_matrix.values[:,col_index]
        col_points[matrix_row_indices != -1] = matrix_column[matrix_row_indices[matrix_row_indices != -1]]
        core_points[:,col_index] = col_points

    core_points_sums = core_points.sum(axis=1)

    # Get the total points for all the possible motifs
    valid_total_points = tract_points + core_points_sums

    if motif_seqs.ndim == 1:
        total_points_motifs = np.full(len(motif_seqs), fill_value=np.nan, dtype=float)
        total_points_motifs[valid_indices] = valid_total_points
    else:
        total_points_motifs = valid_total_points

    return total_points_motifs

def classical_protein_method(sequence, predictor_params = predictor_params, classical_matrix = classical_matrix):
    '''
    Example of a parallel classical scoring method to employ alongside the new one; replace with your method as needed

    Args:
        sequence (str):       protein sequence to check for motifs

    Returns:
        sorted_motifs (np.ndarray): sorted motifs (count = return_count)
        sorted_scores (np.ndarray): corresponding classical motif scores
    '''

    return_count = predictor_params["return_count"]

    if isinstance(sequence, str):
        leading_glycines = np.repeat("G", predictor_params["leading_glycines"])
        trailing_glycines = np.repeat("G", predictor_params["trailing_glycines"])
        seq_array = np.array(list(sequence))
        seq_array = np.concatenate([leading_glycines, seq_array, trailing_glycines])
        slice_indices = np.arange(len(seq_array) - motif_length + 1)[:, np.newaxis] + np.arange(motif_length)
        sliced_seqs_2d = seq_array[slice_indices]

        tract_seqs = sliced_seqs_2d[:,0:6]
        core_seqs = sliced_seqs_2d[:,6:13]

        # Calculate charges on tract sequences
        tract_residue_charges = np.zeros(shape=tract_seqs.shape, dtype=float)
        tract_residue_charges[tract_seqs == "D"] = -1
        tract_residue_charges[tract_seqs == "E"] = -1
        tract_residue_charges[tract_seqs == "S"] = -0.5
        tract_residue_charges[tract_seqs == "T"] = -0.5
        tract_residue_charges[tract_seqs == "R"] = 1
        tract_residue_charges[tract_seqs == "K"] = 1

        tract_charges = tract_residue_charges.sum(axis=1)

        # Translate tract charges into points values
        tract_points = np.full(shape=tract_charges.shape, fill_value=np.inf, dtype=float)
        tract_points[tract_charges <= -4] = 0
        tract_points[np.logical_and(tract_charges <= -3, tract_charges > -4)] = 0.5
        tract_points[np.logical_and(tract_charges <= -2, tract_charges > -3)] = 1
        tract_points[tract_charges > -2] = 1.5

        # Apply matrix to core sequences
        core_points = np.zeros(shape=core_seqs.shape, dtype=float)
        for col_index in np.arange(core_seqs.shape[1]):
            col_residues = core_seqs[:,col_index]
            col_points = np.full(shape=len(col_residues), fill_value=np.inf, dtype=float)

            matrix_row_indices = classical_matrix.index.get_indexer_for(col_residues)
            matrix_column = classical_matrix.values[:,col_index]
            col_points[matrix_row_indices != -1] = matrix_column[matrix_row_indices[matrix_row_indices != -1]]
            core_points[:,col_index] = col_points

        core_points_sums = core_points.sum(axis=1)

        # Get the total points for all the possible motifs
        total_points_motifs = tract_points + core_points_sums
        motifs_count = len(total_points_motifs)

        # Generate columns for the number of motifs that will be returned per protein
        sorted_motifs = []
        sorted_scores = []
        sorted_score_indices = np.argsort(total_points_motifs)
        for i in np.arange(return_count):
            if i < motifs_count:
                next_best_idx = sorted_score_indices[i]
                next_best_score = total_points_motifs[next_best_idx]
                next_best_motif = "".join(sliced_seqs_2d[next_best_idx])
                sorted_scores.append(next_best_score)
                sorted_motifs.append(next_best_motif)
            else:
                sorted_scores.append(np.inf)
                sorted_motifs.append("")
    else:
        sorted_motifs = np.repeat("", return_count)
        sorted_scores = np.repeat(np.nan, return_count)

    return sorted_motifs, sorted_scores