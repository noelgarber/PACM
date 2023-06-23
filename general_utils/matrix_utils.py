# This is a resource script containing functions that aid in the construction of position-weighted matrices

import numpy as np
import pandas as pd

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

def increment_matrix(sequences, sorted_thresholds, signal_values, matrix_df):
    '''
    Function to increment a position-weighted matrix based on sequences and their associated points values

    Args:
        sequences (array-like):     the sequences as a list of strings
        sorted_thresholds (list):   list of tuples of signal thresholds and associated points values
        signal_values (array-like): the signal values associated with the sequences
        matrix_df (pd.DataFrame):   the matrix to increment

    Returns:
        None; operations are performed in-place
    '''

    # Get the relevant points values for each sequence
    points_values = np.zeros(len(sequences))
    for thres_val, points_val in sorted_thresholds:
        points_values[signal_values >= thres_val] = points_val

    # Get lists of indices for use in incrementing the matrix
    row_indices_list = [matrix_df.index.get_indexer_for(list(seq)) for seq in sequences]
    valid_indices_list = [np.where(np.array(row_indices) != -1) for row_indices in row_indices_list]
    valid_row_indices_list = [row_indices[valid_indices] for row_indices, valid_indices in zip(row_indices_list, valid_indices_list)]

    # Increment the matrix
    for points, valid_row_indices, valid_indices in zip(points_values, valid_indices_list, valid_row_indices_list):
        matrix_df.values[valid_row_indices, valid_indices] += points

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

def add_matrix_weights(position_weights, matrix_df = None, matrices_dict = None):
    '''
    Function to apply the matrix weights by position to the generated matrices

    Args:
        position_weights (np.ndarray):  list of position weights; length is equal to slim_length
        matrix_df (pd.DataFrame):       if given, it is a single matrix to which weights will be applied
        matrices_dict (dict):           if given, it is a dict of matrices, where weights are applied to each matrix

    Returns:
        output_df (pd.DataFrame):       the modified matrix_df
    '''

    if matrices_dict is not None:
        weighted_matrices_dict = {}
        for key, matrix in matrices_dict.items():
            weighted_matrix = matrix * position_weights
            weighted_matrices_dict[key] = weighted_matrix
        return weighted_matrices_dict

    elif matrix_df is not None:
        weighted_matrix = matrix_df * position_weights
        return weighted_matrix

    else:
        raise ValueError("add_matrix_weights error: either matrix_df (pd.DataFrame) or matrices_dict (dict of pd.DataFrames) must be given, but both were set to None")