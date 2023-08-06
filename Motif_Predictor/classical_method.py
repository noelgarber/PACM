
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

def classical_method(sequence, predictor_params = predictor_params, classical_matrix = classical_matrix):
    '''
    Example of a parallel classical scoring method to employ alongside the new one; replace with your method as needed

    Args:
        sequence (str): protein sequence to check for motifs

    Returns:
        best_motif_seq, best_motif_score, second_best_seq, second_best_score
    '''

    seq_array = np.array(list("GGGGGG" + sequence + "GG"), dtype="U")
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

    # Generate columns for the number of motifs that will be returned per protein
    return_count = predictor_params["return_count"]
    sorted_motifs = []
    sorted_scores = []
    sorted_score_indices = np.argsort(total_points_motifs)
    for i in np.arange(return_count):
        next_best_idx = sorted_score_indices[i]
        next_best_score = total_points_motifs[next_best_idx]
        next_best_motif = "".join(sliced_seqs_2d[next_best_idx])
        sorted_scores.append(next_best_score)
        sorted_motifs.append(next_best_motif)

    return sorted_motifs, sorted_scores