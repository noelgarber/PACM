# Defines the functions for scoring peptides of equal length using conditional matrices

import numpy as np
import pandas as pd
from general_utils.general_utils import unravel_seqs

def score_seqs(sequences, slim_length, conditional_matrices, sequences_2d = None, convert_phospho = True,
               use_weighted = False, return_2d = True):
    '''
    Vectorized function to score amino acid sequences based on the dictionary of context-aware weighted matrices

    Args:
        sequences (np.ndarray):                     peptide sequences of equal length, as a 1D numpy array
        slim_length (int): 		                    the length of the motif being studied
        conditional_matrices (ConditionalMatrices): conditional weighted matrices for scoring peptides
        sequences_2d (np.ndarray):                  unravelled peptide sequences; optionally provide this upfront for
                                                    performance improvement in loops
        convert_phospho (bool):                     whether to convert phospho-residues to non-phospho before lookups
        use_weighted (bool):                        whether to use conditional_matrices.stacked_weighted_matrices
        return_2d (bool):                           whether to return an array of arrays of points values at each
                                                    position of each peptide in addition to the 1d array

    Returns:
        final_points_array (np.ndarray):           the total motif scores for the input sequences
    '''

    # Unravel sequences to a 2D array if not already provided
    if sequences_2d is None:
        sequences_2d = unravel_seqs(sequences, slim_length, convert_phospho)

    # Get row indices for unique residues
    unique_residues = np.unique(sequences_2d)
    unique_residue_indices = conditional_matrices.index.get_indexer_for(unique_residues)

    if (unique_residue_indices == -1).any():
        failed_residues = unique_residues[unique_residue_indices == -1]
        raise Exception(f"score_seqs error: the following residues were not found by the matrix indexer: {failed_residues}")

    # Get the matrix row indices for all the residues
    aa_row_indices_2d = np.ones(shape=sequences_2d.shape, dtype=int) * -1
    for unique_residue, row_index in zip(unique_residues, unique_residue_indices):
        aa_row_indices_2d[sequences_2d == unique_residue] = row_index

    # Define residues flanking either side of the residues of interest; for out-of-bounds cases, use opposite side twice
    flanking_left_2d = np.concatenate((sequences_2d[:, 0:1], sequences_2d[:, 0:-1]), axis=1)
    flanking_right_2d = np.concatenate((sequences_2d[:, 1:], sequences_2d[:, -1:]), axis=1)

    # Get integer-encoded chemical classes for each residue
    left_encoded_classes_2d = np.zeros(flanking_left_2d.shape, dtype=int)
    right_encoded_classes_2d = np.zeros(flanking_right_2d.shape, dtype=int)
    for member_aa, encoded_class in conditional_matrices.encoded_chemical_classes.items():
        left_encoded_classes_2d[flanking_left_2d == member_aa] = encoded_class
        right_encoded_classes_2d[flanking_right_2d == member_aa] = encoded_class

    # Find the matrix identifier number (first dimension of weighted_matrix_of_matrices) for each encoded class, depending on sequence position
    encoded_positions = np.arange(slim_length) * conditional_matrices.chemical_class_count
    left_encoded_matrix_refs = left_encoded_classes_2d + encoded_positions
    right_encoded_matrix_refs = right_encoded_classes_2d + encoded_positions

    # Flatten the encoded matrix refs, which serve as the first dimension referring to weighted_matrix_of_matrices
    left_encoded_matrix_refs_flattened = left_encoded_matrix_refs.flatten()
    right_encoded_matrix_refs_flattened = right_encoded_matrix_refs.flatten()

    # Flatten the amino acid row indices into a matching array serving as the second dimension
    aa_row_indices_flattened = aa_row_indices_2d.flatten()

    # Tile the column indices into a matching array serving as the third dimension
    column_indices = np.arange(slim_length)
    column_indices_tiled = np.tile(column_indices, len(sequences_2d))

    # Extract values from weighted_matrix_of_matrices
    if use_weighted:
        weighted_matrix_of_matrices = conditional_matrices.stacked_weighted_matrices
    else:
        weighted_matrix_of_matrices = conditional_matrices.stacked_matrices
    left_matrix_values_flattened = weighted_matrix_of_matrices[left_encoded_matrix_refs_flattened, aa_row_indices_flattened, column_indices_tiled]
    right_matrix_values_flattened = weighted_matrix_of_matrices[right_encoded_matrix_refs_flattened, aa_row_indices_flattened, column_indices_tiled]

    # Reshape the extracted values to match sequences_2d
    shape_2d = sequences_2d.shape
    left_matrix_values_2d = left_matrix_values_flattened.reshape(shape_2d)
    right_matrix_values_2d = right_matrix_values_flattened.reshape(shape_2d)

    # Get the final scores by summing values of each row
    final_points_2d = left_matrix_values_2d + right_matrix_values_2d
    final_points_array = final_points_2d.sum(axis=1)

    if return_2d:
        return final_points_array, final_points_2d
    else:
        return final_points_array

def apply_motif_scores(input_df, slim_length, conditional_matrices, sequences_2d = None, seq_col = "No_Phos_Sequence",
                       score_col = "SLiM_Score", convert_phospho = True, add_residue_cols = False, in_place = False,
                       use_weighted = True, return_array = True, return_2d = False, return_df = True,
                       residue_calls_2d = None):
    '''
    Function to apply the score_seqs() function to all sequences in the source df and add residue cols for sorting

    Args:
        input_df (pd.DataFrame):                   df containing motif sequences to back-apply motif scores onto
        slim_length (int): 		                   the length of the motif being studied
        conditional_matrices (ConditionalMatrices): conditional weighted matrices for scoring peptides
        sequences_2d (np.ndarray):                 unravelled peptide sequences; optionally provide this upfront for 
                                                   performance improvement in loops
        seq_col (str): 			                   col in input_df with peptide seqs to score
        score_col (str): 		                   col in input_df that will contain the score values
        convert_phospho (bool):                    whether to convert phospho-residues to non-phospho before lookups
        add_residue_cols (bool):                   whether to add columns containing individual residue letters
        in_place (bool):                           whether to apply operations in-place; add_residue_cols not supported
        use_weighted (bool):                       whether to use conditional_matrices.stacked_weighted_matrices
        return_array (bool):                       whether to only return the array of points values
        return_2d (bool):                          whether to return an array of arrays of points values at each
                                                   position of each peptide in addition to the 1d array
        residue_calls_2d (np.ndarray):             2D array of residue calls as bools

    Returns:
        output_df (pd.DataFrame): dens_df with scores added
    '''

    # Get sequences only if needed; if sequences_2d is already provided, then sequences is not necessary
    if sequences_2d is None:
        seqs = input_df[seq_col].values.astype("<U")
        sequences_2d = unravel_seqs(seqs, slim_length, convert_phospho)
    else:
        seqs = None

    # Get the motif scores for the peptide sequences
    if return_2d and return_array and not return_df:
        scores, scores_2d = score_seqs(seqs, slim_length, conditional_matrices, sequences_2d, convert_phospho,
                                       use_weighted, return_2d = True)
        return (scores, scores_2d)
    elif not return_2d and return_array and not return_df:
        scores = score_seqs(seqs, slim_length, conditional_matrices, sequences_2d, convert_phospho,
                            use_weighted, return_2d = False)
        return scores
    else:
        scores, scores_2d = score_seqs(seqs, slim_length, conditional_matrices, sequences_2d, convert_phospho,
                                       use_weighted, return_2d = True)

    # Assign scores to dataframe
    output_df = input_df if in_place else input_df.copy()
    if return_2d:
        score_res_cols = ["#" + str(position) + "_score" for position in np.arange(1, slim_length + 1)]
        output_df[score_res_cols] = scores_2d
    output_df[score_col] = scores

    # If residue_calls_2d was given, assign to dataframe
    if residue_calls_2d is not None:
        score_call_cols = ["#" + str(position) + "_call" for position in np.arange(1, slim_length + 1)]
        output_df[score_call_cols] = residue_calls_2d.astype("U")
        overall_call_count = residue_calls_2d.sum(axis=1)
        output_df["Residue_Pass_Count"] = overall_call_count

    if add_residue_cols and not in_place:
        # Define the index where residue columns should be inserted
        current_cols = list(output_df.columns)
        insert_index = current_cols.index(seq_col) + 1

        # Assign residue columns
        residue_cols = ["#" + str(i) for i in np.arange(1, slim_length + 1)]
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

    if return_df and return_array and return_2d:
        return (output_df, scores, scores_2d)
    elif return_array and return_df:
        return (output_df, scores)
    elif return_2d and return_df:
        return (output_df, scores_2d)
    elif return_df:
        return output_df