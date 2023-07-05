# This script conducts residue-residue pairwise analysis to generate context-aware SLiM matrices and back-calculated scores.

import numpy as np
import pandas as pd
import os
import time
import multiprocessing
from tqdm import trange
from functools import partial
from general_utils.general_utils import input_number, save_dataframe, save_weighted_matrices, unravel_seqs, check_seq_lengths
from general_utils.weights_utils import permute_weights
from general_utils.matrix_utils import increment_matrix, make_empty_matrix, collapse_phospho, apply_always_allowed, add_matrix_weights
from general_utils.general_vars import aa_charac_dict, amino_acids, amino_acids_phos
from general_utils.user_helper_functions import get_min_members, get_thresholds, get_always_allowed, get_position_weights
from general_utils.statistics import optimize_threshold_fdr, apply_threshold

def qualifying_entries_count(source_dataframe, seq_col, position_for_filtering, residues_included_at_filter_position):
    '''
    Simple function for getting the number of peptides in the source dataframe that fulfill the filter position rule

    Args:
        source_dataframe (pd.DataFrame):				the dataframe containing the peptide binding data
        seq_col (str): 								    the name of the column containing peptide sequences
        position_for_filtering (int):					the position to omit during matrix generation; indexed from 1
        residues_included_at_filter_position (list):	list of residues in the chemical class being assessed

    Returns:
        num_qualifying_entries (int):   the number of peptides that fulfill the filter position rule
    '''
    index_for_filtering = position_for_filtering - 1

    # Default to no filtering if the number of members is below the minimum
    num_qualifying_entries = 0
    for i in np.arange(len(source_dataframe)):
        seq = source_dataframe.at[i, seq_col]
        if position_for_filtering > len(seq):
            raise IndexError(f"Position {position_for_filtering} (index {index_for_filtering}) is out of bounds for sequence {seq}")
        else:
            aa_at_filter_index = seq[index_for_filtering]
            if aa_at_filter_index in residues_included_at_filter_position:
                num_qualifying_entries += 1

    return num_qualifying_entries

'''---------------------------------------------------------------------------------------------------------------------
                                    Define generalized conditional matrix function 
   ------------------------------------------------------------------------------------------------------------------'''

default_data_params = {"bait": None,
                       "bait_signal_col_marker": "Background-Adjusted_Standardized_Signal",
                       "best_signal_col": "Max_Bait_Background-Adjusted_Mean",
                       "bait_pass_col": "One_Passes",
                       "pass_str": "Yes",
                       "seq_col": "BJO_Sequence",
                       "dest_score_col": "SLiM_Score"}

default_matrix_params = {"thresholds_points_dict": None,
                         "points_assignment_mode": "continuous",
                         "included_residues": amino_acids_phos,
                         "amino_acids": amino_acids_phos,
                         "include_phospho": False,
                         "min_members": None,
                         "position_for_filtering": None,
                         "clear_filtering_column": False,
                         "penalize_negatives": True}


def get_signal_cols(source_df, data_params):
    # Helper function to extract signal value containing columns from the source dataframe

    bait = data_params.get("bait")
    if bait is None:
        signal_cols = [data_params.get("best_signal_col")]
    else:
        signal_col_marker = data_params.get("bait_signal_col_marker")
        signal_cols = []
        for col in source_df.columns:
            if bait in col and signal_col_marker in col:
                signal_cols.append(col)

    return signal_cols

def decrement_negatives(penalize_negatives, matrix_df, pass_calls, qualifying_member_calls, sequences_2d,
                        masked_sequences_2d, mean_positive_points):
    # Helper function for conditional_matrix() that decrements the matrix based on negative peptides if indicated

    if penalize_negatives:
        inverse_logical_mask = np.logical_and(~pass_calls, qualifying_member_calls)
        inverse_masked_sequences_2d = sequences_2d[inverse_logical_mask]

        # Decrement the matrix based on inverse masked sequences
        if len(inverse_masked_sequences_2d) > 0:
            # Adjust for inequality in number of negative hits compared to positive
            positive_count = len(masked_sequences_2d)
            negative_count = len(inverse_masked_sequences_2d)
            positive_negative_ratio = positive_count / negative_count
            negative_points = -1 * mean_positive_points * positive_negative_ratio

            # Only use AAs that are never found in passing peptides, i.e. forbidden AAs, to decrement the matrix
            for col_number in np.arange(inverse_masked_sequences_2d.shape[1]):
                positive_masked_col = masked_sequences_2d[:, col_number]
                negative_masked_col = inverse_masked_sequences_2d[:, col_number]
                for aa in np.unique(negative_masked_col):
                    if np.isin(aa, positive_masked_col):
                        # Set non-forbidden residues to X such that they will not be used to decrement the matrix
                        negative_masked_col[negative_masked_col == aa] = "X"
                inverse_masked_sequences_2d[:, col_number] = negative_masked_col

            matrix_df = increment_matrix(None, matrix_df, inverse_masked_sequences_2d,
                                         enforced_points = negative_points, points_mode = "enforced_points")

        else:
            print("No failing sequences found for the current type-position rule, so negatives were not penalized.")

    return matrix_df

def conditional_matrix(motif_length, source_dataframe, data_params = None, matrix_params = None):
    '''
    Function for generating unadjusted conditional matrices from source peptide data,
    based on type-position rules (e.g. #1=Acidic)

    Args:
        motif_length (int):                the length of the motif being assessed
        source_dataframe (pd.DataFrame):   dataframe containing peptide-protein binding data
        data_params (dict):                dictionary of parameters describing the source_dataframe structure:
                                             --> bait (str): the bait to use for matrix generation; defaults to best if left blank
                                             --> bait_signal_col_marker (str): keyword that marks columns in source_dataframe that
                                                 contain signal values; required only if bait is given
                                             --> best_signal_col (str): column name with best signal values; used if bait is None
                                             --> bait_pass_col (str): column name with pass/fail information
                                             --> pass_str (str): the string representing a pass in bait_pass_col, e.g. "Yes"
                                             --> seq_col (str): column name containing peptide sequences as strings
        matrix_params (dict):              dictionary of parameters that affect matrix-building behaviour:
                                             --> thresholds_points_dict (dict): dictionary where threshold_value --> points_value
                                             --> included_residues (list): the residues included for the current type-position rule
                                             --> amino_acids (tuple): the alphabet of amino acids to use when constructing the matrix
                                             --> min_members (int): the minimum number of peptides that must follow the current
                                                 type-position rule for the matrix to be built
                                             --> position_for_filtering (int): the position for the type-position rule being assessed
                                             --> clear_filtering_column (bool): whether to set values in the filtering column to zero
                                             --> penalize_negatives (bool): whether to decrement the matrix based on negative peptides

    Returns:
        matrix_df (pd.DataFrame):		   standardized matrix for the given type-position rule
    '''

    # Get default params if any of them are set to None
    data_params = default_data_params.copy() if data_params is None else data_params
    matrix_params = default_matrix_params.copy() if matrix_params is None else matrix_params

    # Extract and unravel sequences and filter position
    position_for_filtering = matrix_params.get("position_for_filtering")
    index_for_filtering = position_for_filtering - 1
    seq_col = data_params.get("seq_col")
    sequences = source_dataframe[seq_col].to_numpy()
    check_seq_lengths(source_dataframe[seq_col], motif_length)  # check that all seqs in seq_col are the same length
    sequences_2d = unravel_seqs(sequences, motif_length, convert_phospho=False)

    # Create a dataframe where the index is the list of amino acids and the columns are the positions (e.g. #1)
    amino_acids = matrix_params.get("amino_acids")
    matrix_df = make_empty_matrix(motif_length, amino_acids)

    # Get the number of entries that qualify under the current filter position rule
    min_members = matrix_params.get("min_members")
    included_residues = matrix_params.get("included_residues")
    num_qualifying_entries = qualifying_entries_count(source_dataframe, seq_col, position_for_filtering, included_residues)
    included_residues = matrix_params.get("included_residues") if num_qualifying_entries >= min_members else amino_acids

    # Get signal values and pass/fail information
    signal_cols = get_signal_cols(source_dataframe, data_params)
    pass_col = data_params.get("bait_pass_col")
    mean_signal_values = source_dataframe[signal_cols].mean(axis=1).to_numpy()
    pass_values = source_dataframe[pass_col].to_numpy()

    # Get boolean calls for whether each peptide passes and is a qualifying member of the matrix's filtering rule
    pass_str = data_params.get("pass_str")
    pass_calls = pass_values == pass_str
    get_nth_position = np.vectorize(lambda x: x[index_for_filtering] if len(x) >= index_for_filtering+1 else "")
    filter_position_chars = get_nth_position(sequences)
    qualifying_member_calls = np.isin(filter_position_chars, included_residues)

    # Get the sequences and signal values to use for incrementing the matrix
    logical_mask = np.logical_and(pass_calls, qualifying_member_calls)
    masked_sequences_2d = sequences_2d[logical_mask]
    masked_signal_values = mean_signal_values[logical_mask]

    # Assign points for positive peptides and increment the matrix
    points_assignment_mode = matrix_params.get("points_assignment_mode")
    if points_assignment_mode == "continuous":
        matrix_df, mean_points = increment_matrix(None, matrix_df, masked_sequences_2d,
                                                  signal_values = masked_signal_values, points_mode = "continuous",
                                                  return_mean_points = True)
    elif points_assignment_mode == "thresholds":
        thresholds_points_dict = matrix_params.get("thresholds_points_dict")
        sorted_thresholds = sorted(thresholds_points_dict.items(), reverse=True)
        matrix_df, mean_points = increment_matrix(None, matrix_df, masked_sequences_2d, sorted_thresholds,
                                                  masked_signal_values, return_mean_points = True)
    else:
        raise ValueError(f"conditional_matrix error: matrix_params[\"points_assignment_mode\"] is set to {points_assignment_mode}, which is not an accepted mode for positive peptides")

    # If penalizing negatives, decrement the matrix appropriately for negative peptides
    penalize_negatives = matrix_params.get("penalize_negatives") # boolean on whether to penalize negative peptides
    matrix_df = decrement_negatives(penalize_negatives, matrix_df, pass_calls, qualifying_member_calls, sequences_2d,
                                    masked_sequences_2d, mean_points)

    # Optionally combine phospho and non-phospho residue rows in the matrix
    include_phospho = matrix_params.get("include_phospho")
    if not include_phospho:
        collapse_phospho(matrix_df, in_place = True)

    # Optionally set the filtering position to zero
    clear_filtering_column = matrix_params.get("clear_filtering_column")
    if clear_filtering_column:
        matrix_df["#"+str(position_for_filtering)] = 0

    # Convert matrix to floating point values
    matrix_df = matrix_df.astype("float32")

    return matrix_df

'''------------------------------------------------------------------------------------------------------------------
                      Define the main function to generate type-position rule dicts of matrices
   ------------------------------------------------------------------------------------------------------------------'''

def make_unweighted_matrices(source_df, percentiles_dict, slim_length, residue_charac_dict = None, data_params = None,
                             matrix_params = None, return_3d_matrix = False):
    '''
    Function for generating weighted matrices corresponding to each type/position rule (e.g. position #1 = Acidic)

    Args:
        source_df (pd.DataFrame):    dataframe containing peptide-protein binding data
        percentiles_dict (dict): 	 dictionary of percentile --> mean signal value, for use in thesholding
        slim_length (int):           the length of the motif being assessed
        residue_charac_dict (dict):  the dictionary of amino acid characteristics and their constituent amino acids
        data_params (dict):          same as data_params in conditional_matrix()
        matrix_params (dict):        same as matrix_params in conditional_matrix()
        return_3d_matrix (bool):     whether to return a 3D compiled matrix-of-matrices along with the mapping

    Returns:
        dictionary_of_matrices (dict): a dictionary of standardized matrices
    '''

    # Get default params if any of them are set to None
    if data_params is None:
        data_params = default_data_params.copy()
    if matrix_params is None:
        matrix_params = default_matrix_params.copy()

    if matrix_params.get("min_members") is None:
        matrix_params["min_members"] = get_min_members()
    if matrix_params.get("points_assignment_mode") == "thresholds" and not isinstance(matrix_params.get("thresholds_points_dict"), dict):
        matrix_params["thresholds_points_dict"] = get_thresholds(percentiles_dict, use_percentiles = True, show_guidance = True)

    # Get default params if any of them are set to None
    if residue_charac_dict is None:
        residue_charac_dict = aa_charac_dict.copy()
    if data_params is None:
        data_params = default_data_params.copy()
    if matrix_params is None:
        matrix_params = default_matrix_params.copy()

    # Declare dict where keys are position-type rules (e.g. "#1=Acidic") and values are corresponding weighted matrices
    dictionary_of_matrices = {}

    # If a 3D matrix is being returned, create an empty list to hold the matrices to stack, and the mapping
    chemical_class_count = len(residue_charac_dict.keys())
    encoded_chemical_classes = {}
    chemical_class_decoder = {}
    matrices_list = []

    # Iterate over dict of chemical characteristic --> list of member amino acids (e.g. "Acidic" --> ["D","E"]
    for i, (chemical_characteristic, member_list) in enumerate(residue_charac_dict.items()):
        if return_3d_matrix:
            for aa in member_list:
                encoded_chemical_classes[aa] = i
            chemical_class_decoder[i] = chemical_characteristic
        # Iterate over columns for the weighted matrix (position numbers)
        for filter_position in np.arange(1, slim_length + 1):
            # Assign parameters for the current type-position rule
            current_matrix_params = matrix_params.copy()
            current_matrix_params["included_residues"] = member_list
            current_matrix_params["position_for_filtering"] = filter_position

            # Generate the weighted matrix
            print(f"Generating conditional_matrix for rule: #{filter_position}={chemical_characteristic}")
            current_matrix = conditional_matrix(slim_length, source_df, data_params, current_matrix_params)

            # Standardize the weighted matrix so that the max value is 1
            max_values = np.max(current_matrix.values, axis=0)
            max_values = np.maximum(max_values, 1) # prevents divide-by-zero errors
            current_matrix /= max_values

            # Assign the weighted matrix to the dictionary
            dict_key_name = "#" + str(filter_position) + "=" + chemical_characteristic
            dictionary_of_matrices[dict_key_name] = current_matrix
            if return_3d_matrix:
                matrices_list.append(current_matrix)

    if return_3d_matrix:
        '''The order of the 2D matrices in the 3D matrix is one chemical characteristic x all the positions, 
        followed by the next chemical characteristic x all the positions, etc.'''
        matrix_of_matrices = np.stack(matrices_list)
        matrix_index = matrices_list[0].index
        return (dictionary_of_matrices, matrix_of_matrices, matrix_index, encoded_chemical_classes,
                chemical_class_decoder, chemical_class_count)
    else:
        return dictionary_of_matrices

'''------------------------------------------------------------------------------------------------------------------
                    Define functions for scoring source peptide sequences based on generated matrices
   ------------------------------------------------------------------------------------------------------------------'''

def score_seqs(sequences, slim_length, weighted_matrix_of_matrices, matrix_index, encoded_chemical_classes,
               encoded_class_count, sequences_2d = None, convert_phospho = True):
    '''
    Vectorized function to score amino acid sequences based on the dictionary of context-aware weighted matrices

    Args:
        sequences (np.ndarray):                    peptide sequences of equal length, as a 1D numpy array
        slim_length (int): 		                   the length of the motif being studied
        weighted_matrix_of_matrices (np.ndarray):  3D array of shape (matrix_count, matrix_length, matrix_width)
        matrix_index (pd.Index):                   indexer of matrix rows along matrix_length axis
        encoded_chemical_classes (dict):           dict of amino_acid --> integer-encoded chemical classification
        encoded_class_count (int):                 the number of potential integer_encoded chemical classifications
        sequences_2d (np.ndarray):                 unravelled peptide sequences; optionally provide this upfront for 
                                                   performance improvement in loops
        convert_phospho (bool):                    whether to convert phospho-residues to non-phospho before lookups

    Returns:
        final_points_array (np.ndarray):           the total motif scores for the input sequences
    '''

    # Unravel sequences to a 2D array if not already provided
    if sequences_2d is None:
        sequences_2d = unravel_seqs(sequences, slim_length, convert_phospho)

    # Get row indices for unique residues
    unique_residues = np.unique(sequences_2d)
    unique_residue_indices = matrix_index.get_indexer_for(unique_residues)

    if (unique_residue_indices==-1).any():
        failed_residues = unique_residues[unique_residue_indices==-1]
        raise Exception(f"score_seqs error: the following residues were not found by the matrix indexer: {failed_residues}")
    
    # Get the matrix row indices for all the residues
    aa_row_indices_2d = np.ones(shape=sequences_2d.shape, dtype=int) * -1
    for unique_residue, row_index in zip(unique_residues, unique_residue_indices):
        aa_row_indices_2d[sequences_2d == unique_residue] = row_index

    # Define residues flanking either side of the residues of interest; for out-of-bounds cases, use opposite side twice
    flanking_left_2d = np.concatenate((sequences_2d[:,0:1], sequences_2d[:,0:-1]), axis=1)
    flanking_right_2d = np.concatenate((sequences_2d[:,1:], sequences_2d[:,-1:]), axis=1)

    # Get integer-encoded chemical classes for each residue
    left_encoded_classes_2d = np.zeros(flanking_left_2d.shape, dtype=int)
    right_encoded_classes_2d = np.zeros(flanking_right_2d.shape, dtype=int)
    for member_aa, encoded_class in encoded_chemical_classes.items():
        left_encoded_classes_2d[flanking_left_2d==member_aa] = encoded_class
        right_encoded_classes_2d[flanking_right_2d==member_aa] = encoded_class

    # Find the matrix identifier number (first dimension of weighted_matrix_of_matrices) for each encoded class, depending on sequence position
    encoded_positions = np.arange(slim_length) * encoded_class_count
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
    left_matrix_values_flattened = weighted_matrix_of_matrices[left_encoded_matrix_refs_flattened, aa_row_indices_flattened, column_indices_tiled]
    right_matrix_values_flattened = weighted_matrix_of_matrices[right_encoded_matrix_refs_flattened, aa_row_indices_flattened, column_indices_tiled]
    
    # Reshape the extracted values to match sequences_2d
    shape_2d = sequences_2d.shape
    left_matrix_values_2d = left_matrix_values_flattened.reshape(shape_2d)
    right_matrix_values_2d = right_matrix_values_flattened.reshape(shape_2d)
    
    # Get the final scores by summing values of each row
    final_points_array = left_matrix_values_2d.sum(axis=1) + right_matrix_values_2d.sum(axis=1)

    return final_points_array

def apply_motif_scores(input_df, slim_length, weighted_matrix_of_matrices, matrix_index, encoded_chemical_classes,
                       encoded_class_count, sequences_2d = None, seq_col = "No_Phos_Sequence", score_col = "SLiM_Score",
                       convert_phospho = True, add_residue_cols = False, in_place = False, return_array = True):
    '''
    Function to apply the score_seqs() function to all sequences in the source df and add residue cols for sorting

    Args:
        input_df (pd.DataFrame):                   df containing motif sequences to back-apply motif scores onto
        slim_length (int): 		                   the length of the motif being studied
        weighted_matrix_of_matrices (np.ndarray):  3D array of shape (matrix_count, matrix_length, matrix_width)
        matrix_index (pd.Index):                   indexer of matrix rows along matrix_length axis
        encoded_chemical_classes (dict):           dict of amino_acid --> integer-encoded chemical classification
        encoded_class_count (int):                 the number of potential integer_encoded chemical classifications
        sequences_2d (np.ndarray):                 unravelled peptide sequences; optionally provide this upfront for 
                                                   performance improvement in loops
        seq_col (str): 			                   col in input_df with peptide seqs to score
        score_col (str): 		                   col in input_df that will contain the score values
        convert_phospho (bool):                    whether to convert phospho-residues to non-phospho before lookups
        add_residue_cols (bool):                   whether to add columns containing individual residue letters
        in_place (bool):                           whether to apply operations in-place; add_residue_cols not supported

    Returns:
        output_df (pd.DataFrame): dens_df with scores added
    '''

    # Dataframe handling
    if not in_place and not return_array:
        output_df = input_df
    elif in_place and not return_array:
        output_df = input_df.copy()
    else:
        output_df = None

    # Get sequences only if needed; if sequences_2d is already provided, then sequences is not necessary
    sequences = input_df[seq_col].values.astype("<U") if sequences_2d is None else None

    # Get the motif scores for the peptide sequences
    scores = score_seqs(sequences, slim_length, weighted_matrix_of_matrices, matrix_index, encoded_chemical_classes,
                        encoded_class_count, sequences_2d, convert_phospho)

    if return_array:
        return scores

    output_df[score_col] = scores

    if add_residue_cols and not in_place:
        # Assign residue columns
        residues_df = input_df[seq_col].apply(list).apply(pd.Series)
        residue_cols = ["Residue_" + str(i) for i in np.arange(1, slim_length + 1)]
        residues_df.columns = residue_cols
        output_df = pd.concat([output_df, residues_df])

        # Define list of columns in order
        current_cols = list(output_df.columns)
        insert_index = current_cols.index(seq_col) + 1
        final_columns = current_cols[0:insert_index]
        final_columns.extend(residue_cols)
        final_columns.extend(current_cols[insert_index:])

        # Reassign the output df with the ordered columns
        output_df = output_df[final_columns]

    elif add_residue_cols and in_place:
        raise Exception("apply_motif_scores error: in_place cannot be set to True when add_residue_cols is True")

    return output_df

'''------------------------------------------------------------------------------------------------------------------
                        Define functions for parallelized optimization of matrix weights
   ------------------------------------------------------------------------------------------------------------------'''

def process_weights_chunk(chunk, matrix_arrays_dict, matrix_index, source_df, slim_length, sequence_col,
                          significance_col, score_col, significant_str = "Yes", dict_of_aa_characs = None,
                          convert_phospho = True, verbose = False):
    '''
    Lower helper function for parallelization of position weight optimization; processes chunks of permuted weights

    Note that matrix_arrays_dict, unlike matrices_dict which appears elsewhere, contains numpy arrays instead of
    pandas dataframes. This greatly improves performance, but requires the matching pd.Index to be passed separately.

    Args:
        chunk (np.ndarray):         the chunk of permuted weights currently being processed
        matrix_arrays_dict (dict):  the dictionary of position-type rules --> unweighted matrices as np.ndarray
        matrix_index (pd.Index):    the original index for the matrix arrays, as pd.Index, which may be queried using
                                    get_indexer_for() without needing to query each matrix df
        source_df (pd.DataFrame):   the dataframe containing source peptide-protein binding data
        slim_length (int):          the length of the motif being studied
        sequence_col (str):         the name of the column where sequences are stored
        significance_col (str):     the name of the column where significance information is found
        score_col (str):            the name of the column where motif scores are found
        significant_str (str):      the string representing a pass in source_df[significance_col]
        dict_of_aa_characs (dict):  the dictionary of chemical_class --> [amino acid members]
        chemical_class_dict (dict): an inverted dict of amino_acid --> chemical_characteristic; auto-generated if None
        convert_phospho (bool):     whether to convert phospho-residues to non-phospho before doing lookups

    Returns:
        results_tuple (tuple):  (chunk_best_fdr, chunk_best_for, chunk_best_score_threshold, chunk_best_weights,
                                 chunk_best_weighted_matrices_dict, chunk_best_source_df)
    '''

    output_df = source_df.copy()

    passes_bools = source_df[significance_col].values == significant_str

    start_time = time.time()
    # Convert the dictionary of matrices into a 3D numpy array with a dedicated pandas indexer
    ordered_keys = list(matrix_arrays_dict.keys())
    ordered_arrays = list(matrix_arrays_dict.values())
    matrix_of_matrices = np.stack(ordered_arrays) # shape = (array_count, row_count, column_count)

    # Convert keys to encoded chemical classes
    classes_from_keys = []
    for key in ordered_keys:
        key = key.split("#")[-1]
        elements = key.split("=")
        chemical_class = elements[1]
        classes_from_keys.append(chemical_class)
    chemical_classes = np.unique(classes_from_keys)
    chemical_class_encoder = dict(zip(chemical_classes, np.arange(len(chemical_classes))))

    # Get the encoded chemical class dict
    encoded_chemical_classes = {}
    if not isinstance(dict_of_aa_characs, dict):
        dict_of_aa_characs = aa_charac_dict.copy()
    for chemical_class, members in dict_of_aa_characs.items():
        encoded_chemical_class = chemical_class_encoder.get(chemical_class)
        for aa in members:
            encoded_chemical_classes[aa] = encoded_chemical_class
    encoded_class_count = len(dict_of_aa_characs)

    end_time = time.time()
    print(f"Time elapsed while building encoded chemical class dict: {round(end_time-start_time, 3)} s") if verbose else None

    # Convert sequences to a 2D array upfront for performance improvement
    sequences = source_df[sequence_col].to_numpy()
    sequences_2d = unravel_seqs(sequences, slim_length, convert_phospho)

    # Initialize what will become a list of tuples of (best_score, best_fdr, best_for) matching the indices of chunk
    optimal_values = []

    for weights_array in chunk:
        t0 = time.time()
        # Apply the current weights_array to the dict of matrices with numpy; takes almost no time
        current_matrix_of_matrices = matrix_of_matrices * weights_array
        t1 = time.time() # uses 0.1% of the time

        # Get the array of scores for peptide entries in source_df using the current set of weighted matrices
        scores_array = apply_motif_scores(output_df, slim_length, current_matrix_of_matrices, matrix_index,
                                          encoded_chemical_classes, encoded_class_count, sequences_2d,
                                          convert_phospho = convert_phospho, return_array = True)
        t2 = time.time() # uses around 7% of the total time

        # Determine the optimal threshold score that gives balanced FDR/FOR values, which are inversely correlated
        score_range_series = np.linspace(scores_array.min(), scores_array.max(), num=100)
        best_results = optimize_threshold_fdr(None, score_range_series, passes_bools = passes_bools,
                                              scores_array = scores_array, verbose = verbose)
        current_best_score, current_best_fdr, current_best_for = best_results
        t3 = time.time() # uses around 93% of the total time

        total_elapsed = t3 - t0

        print(f"Total time for current loop: {round(total_elapsed, 3)} s",
              f"\n\tCheckpoint #1: {round((t1-t0)/total_elapsed, 3)}",
              f"\n\tCheckpoint #2: {round((t2-t1)/total_elapsed, 2)}",
              f"\n\tCheckpoint #3 {round((t3-t2)/total_elapsed, 2)}") if verbose else None

        optimal_values.append((current_best_score, current_best_fdr, current_best_for))

    # Find the chunk index for the weights array that produces the lowest optimal FDR value
    optimal_values_array = np.array(optimal_values)
    optimal_values_array[np.isnan(optimal_values_array)] = np.inf
    best_index = optimal_values_array[:,1].argmin()

    chunk_best_score_threshold, chunk_best_fdr, chunk_best_for = optimal_values_array[best_index]
    chunk_best_weights = chunk[best_index]

    # Get the matching dict of weighted matrices and use it to apply final scores to output_df
    best_weighted_matrices_dict = add_matrix_weights(chunk_best_weights, matrices_dict = matrix_arrays_dict)
    weighted_matrix_of_matrices = matrix_of_matrices * chunk_best_weights
    chunk_best_source_df = apply_motif_scores(output_df, slim_length, weighted_matrix_of_matrices, matrix_index,
                                              encoded_chemical_classes, encoded_class_count, sequences_2d, score_col,
                                              convert_phospho = convert_phospho, add_residue_cols = True,
                                              in_place = False, return_array = True)

    results_tuple = (chunk_best_fdr, chunk_best_for, chunk_best_score_threshold,
                     chunk_best_weights, best_weighted_matrices_dict, chunk_best_source_df)

    return results_tuple

def process_weights(weights_array_chunks, matrix_arrays_dict, matrix_index, slim_length, source_df,
                    sequence_col, significance_col, significant_str, score_col, dict_of_aa_characs = None,
                    convert_phospho = True):
    '''
    Upper helper function for parallelization of position weight optimization; processes weights by chunking

    Args:
        weights_array_chunks (list): list of chunks as numpy arrays for feeding to process_weights_chunk
        matrix_arrays_dict (dict):   the dictionary of position-type rules --> unweighted matrices as np.ndarray
        matrix_index (pd.Index):     the original index for the matrix arrays, as pd.Index, which may be queried using
                                     get_indexer_for() without needing to query each matrix df
        slim_length (int):           the length of the motif being studied
        source_df (pd.DataFrame):    the dataframe containing source peptide-protein binding data
        sequence_col (str):          the name of the column in chunk where sequences are stored
        significance_col (str):      the name of the column in chunk where significance information is found
        significant_str (str):       the value in significance_col that denotes significance, e.g. "Yes"
        score_col (str):             the name of the column where motif scores are found
        dict_of_aa_characs (dict):   the dictionary of chemical_class --> [amino acid members]
        convert_phospho (bool):      whether to convert phospho-residues to non-phospho before doing lookups

    Returns:
        results_list (list):     the list of results sets for all the weights arrays
    '''

    pool = multiprocessing.Pool()
    if not isinstance(dict_of_aa_characs, dict):
        dict_of_aa_characs = aa_charac_dict.copy()
    process_partial = partial(process_weights_chunk, matrix_arrays_dict = matrix_arrays_dict,
                              matrix_index = matrix_index, source_df = source_df, slim_length = slim_length,
                              sequence_col = sequence_col, significance_col = significance_col, score_col = score_col,
                              significant_str = significant_str, dict_of_aa_characs = dict_of_aa_characs,
                              convert_phospho = convert_phospho)

    results = None
    best_fdr = 9999

    with trange(len(weights_array_chunks), desc="Processing weights") as pbar:
        for chunk_results in pool.imap_unordered(process_partial, weights_array_chunks):
            if chunk_results[0] < best_fdr:
                results = chunk_results

            pbar.update()

    pool.close()
    pool.join()

    return results

def find_optimal_weights(input_df, slim_length, position_copies, matrix_dataframes_dict, sequence_col, significance_col,
                         significant_str, score_col, matrix_output_folder, output_folder,
                         dict_of_aa_characs = aa_charac_dict, convert_phospho = True, chunk_size = 1000,
                         save_pickled_matrix_dict = True):
    '''
    Parent function for finding optimal position weights to generate optimally weighted matrices

    Args:
        input_df (pd.DataFrame):         input dataframe containing all of the sequences, values, and significances
        slim_length (int):               length of the motif being studied
        position_copies (dict):          integer-keyed dictionary where values must be integers whose sum is equal to slim_length
        matrix_dataframes_dict (dict):   the dictionary of position-type rules --> unweighted matrices as pd.DataFrame
        sequence_col (str):              the name of the column in chunk where sequences are stored
        significance_col (str):          the name of the column in chunk where significance information is found
        significant_str (str):           the value in significance_col that denotes significance, e.g. "Yes"
        score_col (str):                 the name of the column where motif scores are found
        dict_of_aa_characs (dict):       the dictionary of chemical_class --> [amino acid members]
        matrix_output_folder (str):      the path for saving weighted matrices
        output_folder (str):             the path for saving the scored data
        convert_phospho (bool):          whether to convert phospho-residues to non-phospho before doing lookups
        chunk_size (int):                the number of position weights to process at a time
        save_pickled_matrix_dict (bool): whether to save a pickled version of the dict of matrices

    Returns:
        results_tuple (tuple):  (best_fdr, best_for, best_score_threshold, best_weights, best_weighted_matrices_dict, best_dens_df)
    '''

    output_df = input_df.copy()

    # Get the permuted weights and break into chunks for parallelization
    expanded_weights_array = permute_weights(slim_length, position_copies)
    weights_array_chunks = [expanded_weights_array[i:i + chunk_size] for i in range(0, len(expanded_weights_array), chunk_size)]

    # Make a dictionary of matrices as np.ndarray instead of dataframes, for performance improvement
    arbitrary_matrix = list(matrix_dataframes_dict.values())[0]
    matrix_index = arbitrary_matrix.index
    matrix_arrays_dict = {}
    for key, matrix_df in matrix_dataframes_dict.items():
        matrix_arrays_dict[key] = matrix_df.to_numpy()

    # Run the parallelized optimization process
    results = process_weights(weights_array_chunks, matrix_arrays_dict, matrix_index, slim_length, output_df,
                              sequence_col, significance_col, significant_str, score_col, dict_of_aa_characs,
                              convert_phospho)
    best_fdr, best_for, best_score_threshold, best_weights, best_weighted_matrices_dict, best_scored_df = results
    print("\t---\n", f"\tOptimal weights of {best_weights} gave FDR = {best_fdr} and FOR = {best_for} at a SLiM score threshold > {best_score_threshold}")

    # Rebuild best_weighted_matrices_dict into a dict of dataframes rather than numpy arrays
    for key, arr in best_weighted_matrices_dict.items():
        df = pd.DataFrame(arr, index = arbitrary_matrix.index, columns = arbitrary_matrix.columns)
        best_weighted_matrices_dict[key] = df

    # Save the weighted matrices and scored data
    save_weighted_matrices(best_weighted_matrices_dict, matrix_output_folder, save_pickled_matrix_dict)
    scored_data_filename = "pairwise_scored_data_thres" + str(best_score_threshold) + ".csv"
    save_dataframe(best_scored_df, output_folder, scored_data_filename)
    print(f"Saved weighted matrices and scored data to {output_folder}")

    return results

'''------------------------------------------------------------------------------------------------------------------
                        Define alternative method for applying predefined, unoptimized weights
   ------------------------------------------------------------------------------------------------------------------'''

def apply_predefined_weights(input_df, position_weights, matrices_dict, slim_length, sequence_col, significance_col,
                             truth_val, score_col, matrix_output_folder, output_folder, make_calls):
    '''
    Function that applies and assesses a given set of weights against matrices and source data

    Args:
        input_df (pd.DataFrame): 	the dataframe containing densitometry values for the peptides being analyzed
        position_weights (list):    list of position weights to use; length must be equal to slim_score
        matrices_dict (dict):       the dictionary of position-type rules --> unweighted matrices
        slim_length (int): 			the length of the motif being studied
        sequence_col (str):         the column in the dataframe that contains peptide sequences
        significance_col (str): 	the column in the dataframe that contains significance calls (Yes/No)
        truth_val (str):          the value to test against input_df[significance_col]
        score_col (str):            the name of the column where motif scores are found
        matrix_output_folder (str): the path to the folder where final matrices should be saved
        output_folder (str): 		the path to the folder where the output data should be saved
        make_calls (bool): 			whether to prompt the user to set a threshold for making positive/negative calls

    Returns:
        output_df (pd.DataFrame): 			the modified dataframe with scores applied
        predictive_value_df (pd.DataFrame): a dataframe containing sensitivity/specificity/PPV/NPV values for different
                                            score thresholds
    '''

    output_df = input_df.copy()

    # Get weights for positions along the motif sequence
    if position_weights is None:
        position_weights = get_position_weights(slim_length)

    # Apply the weights to the matrices
    weighted_matrices_dict = add_matrix_weights(np.array(position_weights), matrices_dict = matrices_dict)

    # Apply the motif scoring algorithm back onto the peptide sequences
    output_df = apply_motif_scores(output_df, weighted_matrices_dict, slim_length, sequence_col, score_col,
                                   add_residue_cols = True, in_place = False)

    # Use thresholding to declare true/false positives/negatives in the peptide sequences
    if make_calls:
        results = apply_threshold(output_df, sig_col = significance_col, score_col = score_col, truth_value = truth_val)
        output_df, selected_threshold, predictive_value_df = results
    else:
        selected_threshold = None
        predictive_value_df = apply_threshold(output_df, sig_col = significance_col, score_col = score_col,
                                              truth_value = truth_val, return_predictive_only = True)

    # Save the weighted matrices and scored data
    save_weighted_matrices(weighted_matrices_dict, matrix_output_folder, save_pickled_dict = True)
    scored_data_filename = "pairwise_scored_data_thres" + str(selected_threshold) + ".csv"
    save_dataframe(output_df, output_folder, scored_data_filename)

    return output_df, weighted_matrices_dict, predictive_value_df

default_general_params = {"percentiles_dict": None,
                          "slim_length": None,
                          "always_allowed_dict": None,
                          "position_weights": None,
                          "output_folder": None,
                          "make_calls": True,
                          "optimize_weights": False,
                          "position_copies": None,
                          "aa_charac_dict": aa_charac_dict,
                          "convert_phospho": True}

def main(input_df, general_params = None, data_params = None, matrix_params = None, verbose = True):
    '''
    Main function for making pairwise position-weighted matrices

    Args:
        input_df (pd.DataFrame): 	the dataframe containing densitometry values for the peptides being analyzed
        general_params (dict):      dictionary of general parameters:
                                            --> percentiles_dict (dict): input data percentile --> mean signal value
                                            --> slim_length (int): the length of the motif being studied
                                            --> always_allowed_dict (dict): position --> always-permitted residues
                                            --> position_weights (list): predefined weight values, if not optimizing
                                            --> output_folder (str): path where the output data should be saved
                                            --> make_calls (bool): whether to set thresholds and making +/- calls
                                            --> optimize_weights (bool): whether to optimize weights to maximize FDR/FOR
                                            --> position_copies (dict): permuted_weight_index --> copy_number;
                                                sum of copy_number values must be equal to slim_length;
                                                required only if optimize_weights is True
                                            --> aa_charac_dict (dict): dictionary of chemical_characteristic --> [AAs]
                                            --> convert_phospho (bool): whether to convert phospho-residues to unphospho
        data_params (dict):         dictionary of parameters describing the source_dataframe structure, used in matrix-building:
                                            --> bait (str): the bait to use for matrix generation; defaults to best if left blank
                                            --> bait_signal_col_marker (str): keyword that marks columns in source_dataframe that
                                                contain signal values; required only if bait is given
                                            --> best_signal_col (str): column name with best signal values; used if bait is None
                                            --> bait_pass_col (str): column name with pass/fail information
                                            --> pass_str (str): the string representing a pass in bait_pass_col, e.g. "Yes"
                                            --> seq_col (str): column name containing peptide sequences as strings
        matrix_params (dict):       dictionary of parameters that affect matrix-building behaviour, used in matrix-building:
                                            --> thresholds_points_dict (dict): dictionary where threshold_value --> points_value
                                            --> included_residues (list): the residues included for the current type-position rule
                                            --> amino_acids (tuple): the alphabet of amino acids to use when constructing the matrix
                                            --> min_members (int): the minimum number of peptides that must follow the current
                                                type-position rule for the matrix to be built
                                            --> position_for_filtering (int): the position for the type-position rule being assessed
                                            --> clear_filtering_column (bool): whether to set values in the filtering column to zero
        verbose (bool): 			whether to display user feedback and debugging information

    Returns:
        output_df (pd.DataFrame): 			the modified dataframe with scores applied
        best_weights (np.ndarray):          only returned if optimize_weights is set to True; it is the best detected
                                            weights leading to optimal FDR/FOR
        predictive_value_df (pd.DataFrame): only returned if optimize_weights is set to False; it is a dataframe 
                                            containing sensitivity/specificity/PPV/NPV values for different score thres
    '''

    if general_params is None:
        general_params = default_general_params.copy()
    if data_params is None:
        data_params = default_data_params.copy()
    if matrix_params is None:
        matrix_params = default_matrix_params.copy()

    # Declare the output folder for saving pairwise weighted matrices
    output_folder = general_params.get("output_folder")
    if output_folder is None:
        output_folder = os.getcwd()
    matrix_output_folder = os.path.join(output_folder, "Pairwise_Matrices")

    # Obtain the dictionary of matrices that have not yet been weighted
    percentiles_dict = general_params.get("percentiles_dict")
    slim_length = general_params.get("slim_length")
    aa_charac_dict = general_params.get("aa_charac_dict")
    results = make_unweighted_matrices(input_df, percentiles_dict, slim_length, aa_charac_dict, data_params,
                                       matrix_params, return_3d_matrix = True)
    matrices_dict, matrix_of_matrices, matrix_index, encoded_chemical_classes, chemical_class_decoder, chemical_class_count = results

    # Apply weights to the generated matrices, or find optimal weights
    optimize_weights = general_params.get("optimize_weights")
    convert_phospho = general_params.get("convert_phospho")
    sequence_col = data_params.get("seq_col")
    significance_col = data_params.get("bait_pass_col")
    significant_str = data_params.get("pass_str")
    score_col = data_params.get("dest_score_col")
    output_statistics = {}

    if optimize_weights:
        # Find the optimal weights that produce the lowest FDR/FOR pair
        position_copies = general_params.get("position_copies")
        results_tuple = find_optimal_weights(input_df, slim_length, position_copies, matrices_dict, sequence_col,
                                             significance_col, significant_str, score_col, matrix_output_folder,
                                             output_folder, aa_charac_dict, convert_phospho, chunk_size = 1000,
                                             save_pickled_matrix_dict = True)
        best_fdr, best_for, best_score_threshold, best_weights, weighted_matrices_dict, scored_df = results_tuple
        position_weights = best_weights
        output_statistics["FDR"] = best_fdr
        output_statistics["FOR"] = best_for
        output_statistics["cutoff_threshold"] = best_score_threshold
        output_statistics["position_weights"] = best_weights

    else:
        # Apply predefined weights and calculate predictive values
        position_weights = general_params.get("position_weights")
        make_calls = general_params.get("make_calls")
        results = apply_predefined_weights(input_df, position_weights, matrices_dict, slim_length, sequence_col,
                                           significance_col, significant_str, score_col, matrix_output_folder,
                                           output_folder, make_calls)
        scored_df, weighted_matrices_dict, predictive_value_df = results
        output_statistics["predictive_value_df"] = predictive_value_df

    return (scored_df, position_weights, weighted_matrices_dict, output_statistics)
