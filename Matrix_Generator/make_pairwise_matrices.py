# This script conducts residue-residue pairwise analysis to generate context-aware SLiM matrices and back-calculated scores.

import numpy as np
import pandas as pd
import os
import pickle
import multiprocessing
from tqdm import trange
from functools import partial
from general_utils.general_utils import input_number, save_dataframe, save_weighted_matrices
from general_utils.weights_utils import permute_weights
from general_utils.matrix_utils import increment_matrix, make_empty_matrix, collapse_phospho, apply_always_allowed, add_matrix_weights
from general_utils.general_vars import aa_charac_dict, amino_acids, amino_acids_phos
from general_utils.user_helper_functions import get_min_members, get_thresholds, get_always_allowed, get_position_weights
from general_utils.statistics import apply_threshold

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
                         "included_residues": amino_acids_phos,
                         "amino_acids": amino_acids_phos,
                         "min_members": None,
                         "position_for_filtering": None,
                         "clear_filtering_column": False}

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

    Returns:
        matrix_df (pd.DataFrame):		   standardized matrix for the given type-position rule
    '''

    # Get default params if any of them are set to None, by getting the first truthy value with 'or'
    data_params = data_params or default_data_params.copy()
    matrix_params = matrix_params or default_matrix_params.copy()

    # Define the position for filtering, for the conditional matrix being generated
    position_for_filtering = matrix_params.get("position_for_filtering")
    index_for_filtering = position_for_filtering - 1

    # Get the dictionary of signal thresholds and associated points for peptides
    thresholds_points_dict = matrix_params.get("thresholds_points_dict")

    # Check if the sequences are all the same length and equal to motif_length, which is required
    seq_col = data_params.get("seq_col")
    sequence_lengths = source_dataframe[seq_col].str.len()
    if sequence_lengths.nunique() > 1:
        raise ValueError(f"weighted_matrix error: source_dataframe sequences in \"{seq_col}\" vary in length from {sequence_lengths.min()} to {sequence_lengths.max()}, but must be equal.")
    elif sequence_lengths[0] != motif_length:
        raise ValueError(f"weighted_matrix error: source_dataframe sequences are {sequence_lengths[0]} amino acids long, but motif_length is set to {motif_length}")

    # Sort the thresholds
    sorted_thresholds = sorted(thresholds_points_dict.items(), reverse = True)

    # Create a dataframe where the index is the list of amino acids and the columns are the positions (e.g. #1)
    amino_acids = matrix_params.get("amino_acids")
    matrix_df = make_empty_matrix(motif_length, amino_acids)

    # Get the number of entries that qualify under the current filter position rule
    min_members = matrix_params.get("min_members")
    included_residues = matrix_params.get("included_residues")
    num_qualifying_entries = qualifying_entries_count(source_dataframe, seq_col, position_for_filtering, included_residues)
    if num_qualifying_entries < min_members:
        # Default to no filtering, which means an unfiltered position-weighted matrix will be returned
        included_residues = amino_acids

    # Get the bait name and data column information
    bait = data_params.get("bait")
    pass_col = data_params.get("bait_pass_col")
    if bait is None:
        signal_cols = [data_params.get("best_signal_col")]
    else:
        signal_col_marker = data_params.get("bait_signal_col_marker")
        signal_cols = []
        for col in source_dataframe.columns:
            if bait in col and signal_col_marker in col:
                signal_cols.append(col)

    mean_signal_values = source_dataframe[signal_cols].mean(axis=1).values
    pass_values = source_dataframe[pass_col].values
    sequences = source_dataframe[seq_col].values

    # Get boolean calls for whether each peptide passes and is a qualifying member of the matrix's filtering rule
    pass_str = data_params.get("pass_str")
    pass_calls = pass_values == pass_str
    get_nth_position = np.vectorize(lambda x: x[index_for_filtering] if len(x) >= index_for_filtering+1 else "")
    filter_position_chars = get_nth_position(sequences)
    qualifying_member_calls = np.isin(filter_position_chars, included_residues)

    # Get the sequences and signal values to use for incrementing the matrix
    logical_mask = np.logical_and(pass_calls, qualifying_member_calls)
    masked_sequences = sequences[logical_mask]
    masked_signal_values = mean_signal_values[logical_mask]

    # Conditionally increment the matrix by the number of points associated with its mean signal value
    matrix_df = increment_matrix(masked_sequences, sorted_thresholds, masked_signal_values, matrix_df)

    # Optionally set the filtering position to zero
    clear_filtering_column = matrix_params.get("clear_filtering_column")
    if clear_filtering_column:
        matrix_df["#"+str(position_for_filtering)] = 0

    # Convert matrix to floating point values
    matrix_df = matrix_df.astype("float32")

    return matrix_df

'''------------------------------------------------------------------------------------------------------------------
                       Define the function to generate type-position rule dicts of matrices
   ------------------------------------------------------------------------------------------------------------------'''

def make_conditional_matrices(slim_length, source_df, residue_charac_dict = None, data_params = None, matrix_params = None):
    '''
    Function for generating weighted matrices corresponding to each type/position rule (e.g. position #1 = Acidic)

    Args:
        slim_length (int):           the length of the motif being assessed
        source_df (pd.DataFrame):    dataframe containing peptide-protein binding data
        residue_charac_dict (dict):  the dictionary of amino acid characteristics and their constituent amino acids
        data_params (dict):          same as data_params in conditional_matrix()
        matrix_params (dict):        same as matrix_params in conditional_matrix()

    Returns:
        dictionary_of_matrices (dict): a dictionary of standardized matrices
    '''

    # Get default params if any of them are set to None, by getting the first truthy value with 'or'
    residue_charac_dict = residue_charac_dict or aa_charac_dict.copy()
    data_params = data_params or default_data_params.copy()
    matrix_params = matrix_params or default_matrix_params.copy()

    # Declare dict where keys are position-type rules (e.g. "#1=Acidic") and values are corresponding weighted matrices
    dictionary_of_matrices = {}

    # Iterate over columns for the weighted matrix (position numbers)
    for filter_position in np.arange(1, slim_length + 1):
        # Iterate over dict of chemical characteristic --> list of member amino acids (e.g. "Acidic" --> ["D","E"]
        for chemical_characteristic, member_list in residue_charac_dict.items():
            # Assign parameters for the current type-position rule
            current_matrix_params = matrix_params.copy()
            current_matrix_params["included_residues"] = member_list
            current_matrix_params["position_for_filtering"] = filter_position

            # Generate the weighted matrix
            current_matrix = conditional_matrix(slim_length, source_df, data_params, current_matrix_params)

            # Standardize the weighted matrix so that the max value is 1
            max_values = np.max(current_matrix.values, axis=0)
            max_values = np.maximum(max_values, 1) # prevents divide-by-zero errors
            current_matrix /= max_values

            # Assign the weighted matrix to the dictionary
            dict_key_name = "#" + str(filter_position) + "=" + chemical_characteristic
            dictionary_of_matrices[dict_key_name] = current_matrix

    return dictionary_of_matrices

'''------------------------------------------------------------------------------------------------------------------
                    Define functions for scoring source peptide sequences based on generated matrices
   ------------------------------------------------------------------------------------------------------------------'''

def score_seqs(sequences, slim_length, weighted_matrices, matrix_type = "df", matrix_index = None,
               convert_phospho = True, dict_of_aa_characs = aa_charac_dict, chemical_class_dict = None):
    '''
    Vectorized function to score amino acid sequences based on the dictionary of context-aware weighted matrices

    Args:
        sequences (np.ndarray):     peptide sequences of equal length (matching weighted matrices), as a 1D numpy array
        slim_length (int): 		    the length of the motif being studied
        weighted_matrices (dict):   dictionary of type-position rule --> position-weighted matrix
        matrix_type (str):          denotes whether each matrix is "df" (pd.DataFrame) or "numpy" (np.ndarray)
        matrix_index (pd.Index):    only required if matrix_type is "numpy"; it is the index that matches the matrices'
                                    dataframe counterparts, and is used for the get_indexer_for() method
        convert_phospho (bool):     whether to convert phospho-residues to non-phospho before doing lookups
        dict_of_aa_characs (dict):  the dictionary of chemical_class --> [amino acid members]
        chemical_class_dict (dict): an inverted dictionary of amino_acid --> chemical_class; auto-generated if not given

    Returns:
        final_points_array (np.ndarray): the total motif scores for the input sequences
    '''

    # Check that all the sequences are an equal, correct length, and then unravel them into a 2D array of residues
    if not np.all(np.vectorize(len)(sequences)==len(sequences[0])):
        raise ValueError(f"score_seqs error: sequences have variable length, but must all be of an equal length")
    elif slim_length != len(sequences[0]):
        raise ValueError(f"score_seqs error: len(sequences[0])={len(sequences[0])}, but slim_length={slim_length}")

    sequences_unravelled = sequences.view("U1")
    sequences_2d = np.reshape(sequences_unravelled, (-1, slim_length))
    if convert_phospho:
        sequences_2d[sequences_2d=="B"] = "S"
        sequences_2d[sequences_2d=="J"] = "T"
        sequences_2d[sequences_2d=="O"] = "Y"

    # Get row indices for unique residues; assume that all the dataframes in weighted_matrices have the same shape
    unique_residues = np.unique(sequences_2d)
    arbitrary_matrix = list(weighted_matrices.values())[0] # obtains a random matrix to use for indexing, assuming they are all indexed the same way
    if matrix_type == "df":
        unique_residue_indices = arbitrary_matrix.index.get_indexer_for(unique_residues)
    elif matrix_type == "numpy" and matrix_index is not None:
        unique_residue_indices = matrix_index.get_indexer_for(unique_residues)
    elif matrix_type == "numpy":
        raise ValueError(f"score_seqs error: matrix_type = \"numpy\", but matrix_index was not given")
    else:
        raise ValueError(f"score_seqs error: invalid matrix_type, got {matrix_type} but expected either \"df\" or \"numpy\"")

    if (unique_residue_indices==-1).any(): # Get weighted matrices from keys
        failed_residues = unique_residues[unique_residue_indices==-1]
        raise Exception(f"score_seqs error: the following residues were not found in by the matrix indexer: {failed_residues}")

    # Define residues flanking either side of the residues of interest; for out-of-bounds cases, use only the other side
    flanking_left_2d = np.concatenate((sequences_2d[:,0:1], sequences_2d[:,0:-1]), axis=1)
    flanking_right_2d = np.concatenate((sequences_2d[:,1:], sequences_2d[:,-1:]), axis=1)

    # Convert chemical classes dict from class_name --> [member list] to member --> class_name
    if not chemical_class_dict:
        chemical_class_dict = {}
        for characteristic, member_list in dict_of_aa_characs.items():
            for member_aa in member_list:
                chemical_class_dict[member_aa] = characteristic

    # Get chemical classes for flanking residues
    left_classes_2d = np.empty(flanking_left_2d.shape, dtype = "<U100") # hard-coded to accept max string length of 100
    right_classes_2d = np.empty(flanking_right_2d.shape, dtype = "<U100")
    for member_aa, characteristic in chemical_class_dict.items():
        left_classes_2d[flanking_left_2d==member_aa] = characteristic
        right_classes_2d[flanking_right_2d==member_aa] = characteristic

    # Get positions, indexed from 1, for residues of interest and the flanking residues on either side
    positions = np.arange(1, slim_length + 1)
    left_positions = positions - 1
    right_positions = positions + 1

    # Start/end residues don't have neighbours on both sides; in these cases, use the existing neighbour twice
    left_positions[0] = right_positions[0]
    right_positions[-1] = left_positions[-1]

    # Get keys for weighted matrices
    left_keys_prefixes = np.char.add(np.char.add("#", left_positions.astype(str)), "=")
    right_keys_prefixes = np.char.add(np.char.add("#", right_positions.astype(str)), "=")

    # Tile prefixes to match shape of sequences_2d
    repeat_count = len(sequences)
    left_prefixes_2d = np.tile(left_keys_prefixes, repeat_count).reshape((repeat_count,len(left_keys_prefixes)))
    right_prefixes_2d = np.tile(right_keys_prefixes, repeat_count).reshape((repeat_count,len(right_keys_prefixes)))

    # Concatenate final keys in 2D
    left_keys_2d = np.char.add(left_prefixes_2d, left_classes_2d)
    right_keys_2d = np.char.add(right_prefixes_2d, right_classes_2d)

    # Convert matrices_dict to a dict of numpy arrays (rather than dataframes) for subsequent concatenation
    if matrix_type != "numpy":
        matrices_arrays_dict = {}
        for key, matrix in weighted_matrices.items():
            matrices_arrays_dict[key] = matrix.to_numpy()
    else:
        matrices_arrays_dict = weighted_matrices

    # Get 3D array of shape (number_of_sequences, sequence_length, matrix_col_slice)
    left_positions_matrices_3d = np.empty((sequences_2d.shape[0], sequences_2d.shape[1], arbitrary_matrix.shape[0]))
    right_positions_matrices_3d = np.empty((sequences_2d.shape[0], sequences_2d.shape[1], arbitrary_matrix.shape[0]))

    for unique_key in np.unique(np.concatenate([left_keys_2d, right_keys_2d])):
        unique_matrix = matrices_arrays_dict.get(unique_key)

        # Iterate over columns (positions) in the unique matrix
        for position_index, matrix_col_slice in enumerate(unique_matrix.T):
            left_column_mask = np.where(left_keys_2d[:,position_index] == unique_key)
            left_positions_matrices_3d[left_column_mask,position_index] = matrix_col_slice

            right_column_mask = np.where(right_keys_2d[:,position_index] == unique_key)
            right_positions_matrices_3d[right_column_mask, position_index] = matrix_col_slice

    # Get matrix row indices
    left_matrix_row_indices = np.empty_like(sequences_2d, dtype=int)
    right_matrix_row_indices = np.empty_like(sequences_2d, dtype=int)
    for unique_residue, row_index in zip(unique_residues, unique_residue_indices):
        left_matrix_row_indices[sequences_2d==unique_residue] = row_index
        right_matrix_row_indices[sequences_2d==unique_residue] = row_index

    # Collapse the 3D arrays of matrix values to a single value at each 2D index for the residue matching it
    left_collapsed_array = left_positions_matrices_3d[np.arange(sequences_2d.shape[0])[:, None], np.arange(sequences_2d.shape[1]), left_matrix_row_indices]
    right_collapsed_array = right_positions_matrices_3d[np.arange(sequences_2d.shape[0])[:, None], np.arange(sequences_2d.shape[1]), right_matrix_row_indices]

    # Get a 1D array of summed points values from the
    left_summed_points = left_collapsed_array.sum(axis=1)
    right_summed_points = right_collapsed_array.sum(axis=1)
    final_points_array = left_summed_points + right_summed_points

    return final_points_array

def apply_motif_scores(input_df, slim_length, weighted_matrices, matrix_type = "df", matrix_index = None,
                       seq_col = "No_Phos_Sequence", score_col = "SLiM_Score", convert_phospho = True,
                       dict_of_aa_characs = aa_charac_dict, chemical_class_dict = None, add_residue_cols = False,
                       in_place = False, return_array = True):
    '''
    Function to apply the score_seqs() function to all sequences in the source df and add residue cols for sorting

    Args:
        input_df (pd.DataFrame):    dataframe containing the motif sequences to back-apply motif scores onto, that were originally used to generate the scoring system
        slim_length (int): 		    the length of the motif being studied
        weighted_matrices (dict):   dictionary of type-position rule --> position-weighted matrix
        matrix_type (str):          denotes whether each matrix is "df" (pd.DataFrame) or "numpy" (np.ndarray)
        matrix_index (pd.Index):    only required if matrix_type is "numpy"; it is the index that matches the matrices'
                                    dataframe counterparts, and is used for the get_indexer_for() method
        seq_col (str): 			    the column in dens_df that contains the peptide sequence to score (unphosphorylated, if model phospho-residues were collapsed to non-phospho during building)
        score_col (str): 		    the column in dens_df that will contain the score values
        convert_phospho (bool):     whether to convert phospho-residues to non-phospho before doing lookups
        dict_of_aa_characs (dict):  the dictionary of chemical_class --> [amino acid members]
        chemical_class_dict (dict): an inverted dictionary of amino_acid --> chemical_class; auto-generated if not given
        add_residue_cols (bool):    whether to add columns containing individual residue letters, for sorting, in a df
        in_place (bool):            whether to apply operations in-place; cannot be True if add_residue_cols is True

    Returns:
        output_df (pd.DataFrame): dens_df with scores added
    '''

    output_df = input_df if in_place else input_df.copy()

    sequences = input_df[seq_col].values.astype("<U")
    scores = score_seqs(sequences, slim_length, weighted_matrices, matrix_type, matrix_index,
                        convert_phospho, dict_of_aa_characs, chemical_class_dict)
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
                                Define main function for generating unweighted matrices
   ------------------------------------------------------------------------------------------------------------------'''

def make_unweighted_matrices(input_df, percentiles_dict = None, slim_length = None, always_allowed_dict = None,
                             aa_charac_dict = aa_charac_dict, data_params = None, matrix_params = None, verbose = True):
    '''
    Main function for making pairwise position-weighted matrices

    Args:
        input_df (pd.DataFrame): 	the dataframe containing densitometry values for the peptides being analyzed
        percentiles_dict (dict): 	dictionary of percentile --> mean signal value, for use in thesholding
        slim_length (int): 			the length of the motif being studied
        always_allowed_dict (dict): a dictionary of position number (int) --> always-permitted residues at that position (list)
        aa_charac_dict (dict):      dictionary of amino acid characteristics --> list of member residues as single-letter codes
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
        matrices_dict (dict): 		a dictionary of position-type rule keys --> unweighted matrix for that rule
        predictive_value_df (pd.DataFrame): a dataframe containing sensitivity/specificity/PPV/NPV values for different
                                            score thresholds
    '''
    print("Starting the pairwise matrices generation process...") if verbose else None

    # Get default params if any of them are set to None, by getting the first truthy value with 'or'
    data_params = data_params or default_data_params.copy()
    matrix_params = matrix_params or default_matrix_params.copy()

    # Define the length of the short linear motif (SLiM) being studied
    if slim_length is None:
        slim_length = int(input("Enter the length of your SLiM of interest as an integer (e.g. 15):  "))

    # Get the minimum number of peptides that must belong to a classified group for them to be used in matrix-building
    if matrix_params.get("min_members") is None:
        matrix_params["min_members"] = get_min_members()

    # Get threshold and point values
    thresholds_points_dict = matrix_params.get("thresholds_points_dict")
    if not isinstance(thresholds_points_dict, dict):
        thresholds_points_dict = get_thresholds(percentiles_dict, use_percentiles = True, show_guidance = True)
        matrix_params["thresholds_points_dict"] = thresholds_points_dict

    # Make the dictionary of weighted matrices based on amino acid composition across positions
    print("Generating matrices...") if verbose else None
    matrices_dict = make_conditional_matrices(slim_length, input_df, aa_charac_dict, data_params, matrix_params)

    # Get list of always-allowed residues (overrides algorithm for those positions)
    print("Collapsing phospho-residues to their non-phospho counterparts and applying always-allowed residues...") if verbose else None
    if always_allowed_dict is None:
        always_allowed_dict = get_always_allowed(slim_length = slim_length)

    # Collapse phospho-residues into non-phospho counterparts and apply always-allowed residues
    for key, df in matrices_dict.items():
        collapse_phospho(df, in_place = True)
        apply_always_allowed(df, always_allowed_dict, in_place = True)
        matrices_dict[key] = df

    return matrices_dict

'''------------------------------------------------------------------------------------------------------------------
                        Define functions for parallelized optimization of matrix weights
   ------------------------------------------------------------------------------------------------------------------'''

def process_weights_chunk(chunk, matrix_arrays_dict, matrix_index, source_df, slim_length, sequence_col,
                          significance_col, score_col, significant_str = "Yes", dict_of_aa_characs = aa_charac_dict,
                          chemical_class_dict = None, convert_phospho = True):
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

    passes_bools = source_df[significance_col]==significant_str

    # Initialize what will become a list of tuples of (best_score, best_fdr, best_for) matching the indices of chunk
    optimal_values = []

    for weights_array in chunk:
        # Apply the current weights_array to the dict of matrices as a numpy multiplication operation
        current_weighted_matrices_dict = add_matrix_weights(weights_array, matrices_dict = matrix_arrays_dict)

        # Get the array of scores for peptide entries in source_df using the current set of weighted matrices
        scores_array = apply_motif_scores(output_df, slim_length, current_weighted_matrices_dict, "numpy", matrix_index,
                                          sequence_col, score_col, convert_phospho, dict_of_aa_characs,
                                          chemical_class_dict, return_array = True)

        # Determine the optimal threshold score that gives balanced FDR/FOR values, which are inversely correlated
        score_range_series = np.linspace(scores_array.min(), scores_array.max(), num=100)
        current_best_score, current_best_fdr, current_best_for = apply_threshold(None, score_range_series,
                                                                                 passes_bools = passes_bools,
                                                                                 scores_array = scores_array,
                                                                                 return_optimized_fdr = True)

        optimal_values.append((current_best_score, current_best_fdr, current_best_for))

    # Find the chunk index for the weights array that produces the lowest optimal FDR value
    optimal_values_array = np.array(optimal_values)
    optimal_values_array[np.isnan(optimal_values_array)] = np.inf
    best_index = optimal_values_array[:,1].argmin()

    chunk_best_score_threshold, chunk_best_fdr, chunk_best_for = optimal_values_array[best_index]
    chunk_best_weights = chunk[best_index]

    # Get the matching dict of weighted matrices and use it to apply final scores to output_df
    best_weighted_matrices_dict = add_matrix_weights(chunk_best_weights, matrices_dict = matrix_arrays_dict)
    chunk_best_source_df = apply_motif_scores(source_df, slim_length, best_weighted_matrices_dict, "numpy",
                                              matrix_index, sequence_col, score_col, convert_phospho,
                                              dict_of_aa_characs, chemical_class_dict, add_residue_cols = True,
                                              in_place = False, return_array = False)

    results_tuple = (chunk_best_fdr, chunk_best_for, chunk_best_score_threshold,
                     chunk_best_weights, best_weighted_matrices_dict, chunk_best_source_df)

    return results_tuple

def process_weights(weights_array_chunks, matrix_arrays_dict, matrix_index, slim_length, source_df,
                    sequence_col, significance_col, score_col, dict_of_aa_characs = aa_charac_dict,
                    chemical_class_dict = None, convert_phospho = True):
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
        score_col (str):             the name of the column where motif scores are found
        dict_of_aa_characs (dict):   the dictionary of chemical_class --> [amino acid members]
        chemical_class_dict (dict):  an inverted dict of amino_acid --> chemical_characteristic; auto-generated if None
        convert_phospho (bool):      whether to convert phospho-residues to non-phospho before doing lookups

    Returns:
        results_list (list):     the list of results sets for all the weights arrays
    '''

    # Generate inverted version of dict_of_aa_characs if not given upfront
    if not chemical_class_dict:
        chemical_class_dict = {}
        for characteristic, member_list in dict_of_aa_characs.items():
            for member_aa in member_list:
                chemical_class_dict[member_aa] = characteristic

    pool = multiprocessing.Pool()
    process_partial = partial(process_weights_chunk, matrix_arrays_dict = matrix_arrays_dict,
                              matrix_index = matrix_index, source_df = source_df, slim_length = slim_length,
                              sequence_col = sequence_col, significance_col = significance_col, score_col = score_col,
                              dict_of_aa_characs = None, chemical_class_dict = chemical_class_dict,
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
                         score_col, matrix_output_folder, output_folder, dict_of_aa_characs = aa_charac_dict,
                         convert_phospho = True, chunk_size = 1000, save_pickled_matrix_dict = True):
    '''
    Parent function for finding optimal position weights to generate optimally weighted matrices

    Args:
        input_df (pd.DataFrame):         input dataframe containing all of the sequences, values, and significances
        slim_length (int):               length of the motif being studied
        position_copies (dict):          integer-keyed dictionary where values must be integers whose sum is equal to slim_length
        matrix_dataframes_dict (dict):   the dictionary of position-type rules --> unweighted matrices as pd.DataFrame
        sequence_col (str):              the name of the column in chunk where sequences are stored
        significance_col (str):          the name of the column in chunk where significance information is found
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
                              sequence_col, significance_col, score_col, dict_of_aa_characs, None, convert_phospho)
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
                             score_col, matrix_output_folder, output_folder, make_calls):
    '''
    Function that applies and assesses a given set of weights against matrices and source data

    Args:
        input_df (pd.DataFrame): 	the dataframe containing densitometry values for the peptides being analyzed
        position_weights (list):    list of position weights to use; length must be equal to slim_score
        matrices_dict (dict):       the dictionary of position-type rules --> unweighted matrices
        slim_length (int): 			the length of the motif being studied
        sequence_col (str):         the column in the dataframe that contains peptide sequences
        significance_col (str): 	the column in the dataframe that contains significance calls (Yes/No)
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
        output_df, selected_threshold, predictive_value_df = apply_threshold(output_df, sig_col = significance_col,
                                                                             score_col = score_col)
    else:
        selected_threshold = None
        predictive_value_df = apply_threshold(output_df, sig_col = significance_col, score_col = score_col,
                                              return_pred_vals_only = True)

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

    general_params = general_params or default_general_params.copy()
    data_params = data_params or default_data_params.copy()
    matrix_params = matrix_params or default_matrix_params.copy()

    # Declare the output folder for saving pairwise weighted matrices
    output_folder = general_params.get("output_folder")
    if output_folder is None:
        output_folder = os.getcwd()
    matrix_output_folder = os.path.join(output_folder, "Pairwise_Matrices")

    # Obtain the dictionary of matrices that have not yet been weighted
    percentiles_dict = general_params.get("percentiles_dict")
    slim_length = general_params.get("slim_length")
    always_allowed_dict = general_params.get("always_allowed_dict")
    aa_charac_dict = general_params.get("aa_charac_dict")
    matrices_dict = make_unweighted_matrices(input_df, percentiles_dict, slim_length, always_allowed_dict,
                                             aa_charac_dict, data_params, matrix_params, verbose)

    # Apply weights to the generated matrices, or find optimal weights
    optimize_weights = general_params.get("optimize_weights")
    convert_phospho = general_params.get("convert_phospho")
    sequence_col = data_params.get("seq_col")
    significance_col = data_params.get("bait_pass_col")
    score_col = data_params.get("dest_score_col")
    output_statistics = {}

    if optimize_weights:
        # Find the optimal weights that produce the lowest FDR/FOR pair
        position_copies = general_params.get("position_copies")
        results_tuple = find_optimal_weights(input_df, slim_length, position_copies, matrices_dict, sequence_col,
                                             significance_col, score_col, matrix_output_folder, output_folder,
                                             aa_charac_dict, convert_phospho, chunk_size = 1000,
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
                                           significance_col, score_col, matrix_output_folder, output_folder, make_calls)
        scored_df, weighted_matrices_dict, predictive_value_df = results
        output_statistics["predictive_value_df"] = predictive_value_df

    return (scored_df, position_weights, weighted_matrices_dict, output_statistics)
