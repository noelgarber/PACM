# This is a resource script containing functions that aid in the construction of position-weighted matrices

import numpy as np
import pandas as pd
import warnings

def make_empty_matrix(position_count, amino_acid_list, dtype=np.int64):
    '''
    Function to make an empty position-weighted matrix with the list of amino acids as rows and the columns as positions

    Args:
        position_count (int):   the number of positions for making columns
        amino_acid_list (list): the list of amino acids in the desired order for rows
        dtype (type):           matrix value type; default is np.int64

    Returns:
        empty_matrix_df (pd.DataFrame): an empty position-weighted matrix filled with zeros
    '''
    # Create a dataframe where the index is the list of amino acids and the columns are the positions (e.g. #1)
    numbered_positions = []
    for i in range(1, int(position_count) + 1):
        numbered_positions.append("#" + str(i))

    empty_matrix_df = pd.DataFrame(index = amino_acid_list, columns = numbered_positions)
    empty_matrix_df = empty_matrix_df.fillna(0)

    empty_matrix_df = empty_matrix_df.astype(dtype)

    return empty_matrix_df

def get_peptide_points(signal_values, sequence_count, sorted_thresholds, enforced_points = None, verbose = False):
    # Helper function to apply sorted thresholds, as tuples of (threshold, points), to an array of peptide sequences

    if enforced_points is None:
        points_values = np.zeros(sequence_count, dtype=float)
        for thres_val, points_val in sorted_thresholds:
            points_values[signal_values >= thres_val] = points_val
        mean_points = points_values.mean()
        print(f"Assigned an average of {mean_points} points to {sequence_count} positive peptides") if verbose else None
    else:
        points_values = np.repeat(enforced_points, sequence_count)
        mean_points = enforced_points
        print(f"Assigned {enforced_points} to {sequence_count} negative peptides") if verbose else None

    return points_values, mean_points

def increment_matrix(sequences, matrix_df, sequences_2d = None, sorted_thresholds = None, signal_values = None,
                     enforced_points = None, return_mean_points = False, verbose = False):
    '''
    Function to increment a position-weighted matrix based on sequences and their associated points values

    Args:
        sequences (array-like):     the sequences as a list of strings
        matrix_df (pd.DataFrame):   the matrix to increment
        sequences_2d (np.ndarray):  optionally work from a 2D unravelled array of sequences where each row is an
                                    array of amino acids constituting a peptide
        sorted_thresholds (list):   list of tuples of signal thresholds and associated points values
        signal_values (array-like): the signal values associated with the sequences
        enforced_points (float):    if not using thresholding, this is the enforced points value used for incrementing;
                                    negative values result in decrementing
        return_mean_points (bool):  whether to also return the mean points value for all positive peptides
        verbose (bool):             whether to display debugging info

    Returns:
        matrix_df (pd.DataFrame):   the updated matrix with incremented values
    '''

    # Get sequence count; if 0, return matrix_df as-is, since there is nothing more to do
    sequence_count = len(sequences) if sequences_2d is None else len(sequences_2d)
    if sequence_count == 0:
        warnings.warn("increment_matrix warning: 0 sequences were passed, so matrix_df was returned as-is", category = RuntimeWarning)
        results = matrix_df if not return_mean_points else (matrix_df, 0)
        return results

    # Get the relevant points values for each sequence
    points_values, mean_points = get_peptide_points(signal_values, sequence_count, sorted_thresholds,
                                                    enforced_points, verbose)

    # Get the indices for incrementing matrix_df
    if sequences_2d is None:
        rows_2d = np.array([matrix_df.index.get_indexer_for(list(seq)) for seq in sequences])
    else:
        rows_2d = np.array([matrix_df.index.get_indexer_for(seq_array) for seq_array in sequences_2d])

    # Increment the matrix
    matrix_array = matrix_df.values.astype(float)
    if (rows_2d==-1).any():
        # Some residues were not found, so masking is required
        for i, row_indices in enumerate(rows_2d):
            valid_col_indices = np.where(row_indices != -1)[0]
            valid_row_indices = row_indices[valid_col_indices]
            points = points_values[i]
            matrix_array[valid_row_indices, valid_col_indices] += points
    else:
        # All columns are valid because all residues at all positions received valid row indices from get_indexer_for()
        for i, row_indices in enumerate(rows_2d):
            points = points_values[i]
            col_indices = np.arange(rows_2d.shape[1])
            matrix_array[row_indices, col_indices] += points

    # Reconstruct the matrix dataframe based on the newly incremented array
    matrix_df = pd.DataFrame(matrix_array, index=matrix_df.index, columns=matrix_df.columns)

    results = matrix_df if not return_mean_points else (matrix_df, mean_points)

    return results

def collapse_phospho(matrix_df, in_place = True):
    '''
    Function to collapse the matrix rows for B,J,O (pSer, pThr, pTyr) into S,T,Y respectively, since phosphorylation
    status is not usually known when scoring a de novo sequence.

    Args:
        matrix_df (pd.DataFrame): the matrix containing rows for B, J, O
        in_place (bool):          whether to perform the operation in-place or return a modified copy

    Returns:
        matrices_dict (dict): updated dictionary with rows collapsed per the above description
    '''

    output_df = matrix_df if in_place else matrix_df.copy()

    phospho_residues = ["B","J","O"]
    counterparts = ["S","T","Y"]
    drop_indices = []
    for phospho_residue, counterpart in zip(phospho_residues, counterparts):
        if output_df.index.__contains__(phospho_residue):
            output_df.loc[counterpart] = output_df.loc[counterpart] + output_df.loc[phospho_residue]
            drop_indices.append(phospho_residue)

    output_df.drop(labels=drop_indices, axis=0, inplace=True)

    return output_df

def apply_always_allowed(matrix_df, always_allowed_dict, in_place = True):
    '''
    Function to apply the override always-allowed residues specified by the user for the matrix

    Args:
        matrix_df (pd.DataFrame):   the matrix being operated on
        always_allowed_dict (dict): dictionary of position (e.g. #1) --> list of residues always allowed at that position
        in_place (bool):            whether to perform the operation in-place or return a modified copy

    Returns:
        output_df (pd.DataFrame): the modified matrix_df
    '''

    output_df = matrix_df if in_place else matrix_df.copy()
    position_cols = output_df.columns

    # Check if hash signs are consistently used
    df_contains_hash = output_df.columns.str.contains('#').all()
    keys_contain_hash = np.char.count(list(always_allowed_dict.keys()),"#").all()
    if not df_contains_hash and keys_contain_hash:
        raise Exception("apply_always_allowed error: matrix_df columns are not in the format \"#n\", where n is a number, but this was expected for the given always_allowed_dict")

    for position_col in position_cols:
        always_allowed_residues = always_allowed_dict.get(position_col)
        for residue in always_allowed_residues:
            output_df.at[residue, position_col] = 1

    return output_df

def add_matrix_weights(position_weights, matrix = None, matrices_dict = None):
    '''
    Function to apply the matrix weights by position to the generated matrices

    Args:
        position_weights (np.ndarray):        list of position weights; length is equal to slim_length
        matrix (pd.DataFrame or np.ndarray):  if given, it is a single matrix to which weights will be applied
        matrices_dict (dict):                 if given, it is a dict of matrices (either as pd.DataFrame or np.ndarray),
                                              where weights are applied to each matrix

    Returns:
        weighted_matrix (pd.DataFrame or np.ndarray): if a single matrix was given, this is the modified matrix of the
                                                      same object type as the original (either df or ndarray)
        weighted_matrices_dict (dict):                if a dict of matrices was given, this is the modified dict with
                                                      weights applied, in the same object type as original (df or arr)
    '''

    if matrices_dict is not None:
        weighted_matrices_dict = {}
        for key, matrix in matrices_dict.items():
            weighted_matrix = matrix * position_weights
            weighted_matrices_dict[key] = weighted_matrix
        return weighted_matrices_dict

    elif matrix is not None:
        weighted_matrix = matrix * position_weights
        return weighted_matrix

    else:
        raise ValueError("add_matrix_weights error: either matrix_df (pd.DataFrame) or matrices_dict (dict of pd.DataFrames) must be given, but both were set to None")