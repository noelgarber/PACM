# Defines the ConditionalMatrix and ConditionalMatrices classes

import numpy as np
import pandas as pd
from general_utils.general_utils import unravel_seqs, check_seq_lengths
from general_utils.matrix_utils import increment_matrix, make_empty_matrix, collapse_phospho
from general_utils.general_vars import amino_acids_phos
from general_utils.user_helper_functions import get_min_members, get_thresholds

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

class ConditionalMatrix:
    '''
    Class that contains a position-weighted matrix (self.matrix_df) based on input data and a conditional type-position
    rule, e.g. position #1 = [D,E]
    '''
    def __init__(self, motif_length, source_df, data_params, matrix_params):
        '''
        Function for initializing unadjusted conditional matrices from source peptide data,
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
        '''
        # Construct the empty matrix dataframe
        amino_acids = matrix_params.get("amino_acids")
        self.matrix_df = make_empty_matrix(motif_length, amino_acids)
        
        # Extract and unravel sequences
        seq_col = data_params.get("seq_col")
        sequences = source_df[seq_col].to_numpy()
        check_seq_lengths(source_df[seq_col], motif_length)  # check that all seqs in seq_col are the same length
        sequences_2d = unravel_seqs(sequences, motif_length, convert_phospho=False)

        # Get the number of entries that qualify under the current filter position rule
        position_for_filtering = matrix_params.get("position_for_filtering")
        min_members = matrix_params.get("min_members")
        included_residues = matrix_params.get("included_residues")
        num_qualifying_entries = self.qualifying_entries_count(source_df, seq_col, position_for_filtering, included_residues)
        if num_qualifying_entries < min_members: 
            self.sufficient_seqs = False
            included_residues = amino_acids
        else: 
            self.sufficient_seqs = True
        self.rule = (position_for_filtering, included_residues)

        # Get signal values representing binding strength for source peptide sequences against the target protein(s)
        signal_cols = self.get_signal_cols(source_df, data_params)
        mean_signal_values = source_df[signal_cols].mean(axis=1).to_numpy()
        
        # Get boolean calls for whether each peptide passes and is a qualifying member of the matrix's filtering rule
        pass_str = data_params.get("pass_str")
        pass_col = data_params.get("bait_pass_col")
        pass_values = source_df[pass_col].to_numpy()
        pass_calls = pass_values == pass_str
        index_for_filtering = position_for_filtering - 1
        get_nth_position = np.vectorize(lambda x: x[index_for_filtering] if len(x) >= index_for_filtering + 1 else "")
        filter_position_chars = get_nth_position(sequences)
        qualifying_member_calls = np.isin(filter_position_chars, included_residues)

        # Get the sequences and signal values to use for incrementing the matrix
        logical_mask = np.logical_and(pass_calls, qualifying_member_calls)
        masked_sequences_2d = sequences_2d[logical_mask]
        masked_signal_values = mean_signal_values[logical_mask]
        
        # Assign points for positive peptides and increment the matrix
        points_assignment_mode = matrix_params.get("points_assignment_mode")
        thres_points_dict = matrix_params.get("thresholds_points_dict")
        mean_points = self.increment_positives(masked_sequences_2d, masked_signal_values, points_assignment_mode, thres_points_dict)

        # If penalizing negatives, decrement the matrix appropriately for negative peptides
        penalize_negatives = matrix_params.get("penalize_negatives") # boolean on whether to penalize negative peptides
        if penalize_negatives:
            self.decrement_negatives(pass_calls, qualifying_member_calls, sequences_2d, masked_sequences_2d, mean_points)

        # Optionally combine phospho and non-phospho residue rows in the matrix
        include_phospho = matrix_params.get("include_phospho")
        if not include_phospho:
            collapse_phospho(self.matrix_df, in_place=True)

        # Optionally set the filtering position to zero
        clear_filtering_column = matrix_params.get("clear_filtering_column")
        if clear_filtering_column:
            matrix_df["#" + str(position_for_filtering)] = 0

        # Remove negatives
        self.matrix_df[self.matrix_df < 0] = 0
        self.matrix_df = self.matrix_df.astype("float32")

    def qualifying_entries_count(self, source_df, seq_col, position_for_filtering, residues_included_at_filter_position):
        # Helper function to get the number of sequences that qualify under the current type-position rule

        index_for_filtering = position_for_filtering - 1
    
        # Default to no filtering if the number of members is below the minimum
        num_qualifying_entries = 0
        for i in np.arange(len(source_df)):
            seq = source_df.at[i, seq_col]
            if position_for_filtering > len(seq):
                raise IndexError(f"Position {position_for_filtering} (index {index_for_filtering}) is out of bounds for sequence {seq}")
            else:
                aa_at_filter_index = seq[index_for_filtering]
                if aa_at_filter_index in residues_included_at_filter_position:
                    num_qualifying_entries += 1
    
        return num_qualifying_entries

    def get_signal_cols(self, source_df, data_params):
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

    def increment_positives(self, masked_sequences_2d, masked_signal_values, points_assignment_mode = "continuous",
                            thresholds_points_dict = None):
        if points_assignment_mode == "continuous":
            self.matrix_df, mean_points = increment_matrix(None, self.matrix_df, masked_sequences_2d,
                                                           signal_values = masked_signal_values,
                                                           points_mode = "continuous", return_mean_points = True)
        elif points_assignment_mode == "thresholds":
            thresholds_points_dict = matrix_params.get("thresholds_points_dict")
            sorted_thresholds = sorted(thresholds_points_dict.items(), reverse=True)
            self.matrix_df, mean_points = increment_matrix(None, self.matrix_df, masked_sequences_2d, sorted_thresholds,
                                                           masked_signal_values, return_mean_points = True)
        else:
            raise ValueError(f"ConditionalMatrix initialization error: matrix_params[\"points_assignment_mode\"] is set to {points_assignment_mode}, but must be either `continuous` or `thresholds`")

        return mean_points

    def decrement_negatives(self, pass_calls, qualifying_member_calls, sequences_2d,
                            masked_sequences_2d, mean_positive_points):
        # Helper function that decrements the matrix based on negative peptides if indicated

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

            self.matrix_df = increment_matrix(None, self.matrix_df, inverse_masked_sequences_2d,
                                              enforced_points = negative_points, points_mode = "enforced_points")

class ConditionalMatrices:
    '''
    Class that contains a set of conditional matrices defined using ConditionalMatrix()
    '''
    def __init__(self, motif_length, source_df, percentiles_dict, residue_charac_dict, data_params, matrix_params):
        '''
        Initialization function for generating conditional matrices properties, dict, and 3D representation

        Args:
           motif_length:        length of the motif represented in the matrices
           source_df:           dataframe with source peptide sequences, signal values, and pass/fail information
           percentiles_dict:    dict of integer percentiles from 1-99 for the signal values
           residue_charac_dict: dict of amino acid chemical characteristics
           data_params:         same as in ConditionalMatrix()
           matrix_params:       same as in ConditionalMatrix()
        '''


        if matrix_params.get("min_members") is None:
           matrix_params["min_members"] = get_min_members()
        if matrix_params.get("points_assignment_mode") == "thresholds" and not isinstance(matrix_params.get("thresholds_points_dict"), dict):
           matrix_params["thresholds_points_dict"] = get_thresholds(percentiles_dict, use_percentiles=True, show_guidance=True)

        # Declare dict where keys are position-type rules (e.g. "#1=Acidic") and values are corresponding weighted matrices
        self.matrices_dict = {}

        # For generating a 3D matrix, create an empty list to hold the matrices to stack, and the mapping
        self.residue_charac_dict = residue_charac_dict
        self.chemical_class_count = len(residue_charac_dict.keys())
        self.encoded_chemical_classes = {}
        self.chemical_class_decoder = {}
        matrices_list = []

        # Iterate over dict of chemical characteristic --> list of member amino acids (e.g. "Acidic" --> ["D","E"]
        for i, (chemical_characteristic, member_list) in enumerate(residue_charac_dict.items()):
            # Map the encodings for the chemical classes
            for aa in member_list:
               self.encoded_chemical_classes[aa] = i
            self.chemical_class_decoder[i] = chemical_characteristic

            # Iterate over columns for the weighted matrix (position numbers)
            for filter_position in np.arange(1, motif_length + 1):
               # Assign parameters for the current type-position rule
               current_matrix_params = matrix_params.copy()
               current_matrix_params["included_residues"] = member_list
               current_matrix_params["position_for_filtering"] = filter_position

               # Generate the weighted matrix
               conditional_matrix = ConditionalMatrix(motif_length, source_df, data_params, current_matrix_params)
               current_matrix = conditional_matrix.matrix_df

               # Standardize the weighted matrix so that the max value is 1
               max_values = np.max(current_matrix.values, axis=0)
               max_values = np.maximum(max_values, 1)  # prevents divide-by-zero errors
               current_matrix /= max_values

               # Assign the weighted matrix to the dictionary
               dict_key_name = "#" + str(filter_position) + "=" + chemical_characteristic
               self.matrices_dict[dict_key_name] = current_matrix
               matrices_list.append(current_matrix)

        # Make an array representation of matrices_dict
        self.index = matrices_list[0].index
        self.columns = matrices_list[0].columns
        self.matrix_arrays_dict = {}
        for key, matrix_df in self.matrices_dict.items():
            self.matrix_arrays_dict[key] = matrix_df.to_numpy()

        # Make the 3D matrix
        self.stacked_matrices = np.stack(matrices_list)

    def apply_weights(self, weights_array, only_3d = True):
        # Method for assigning weights to the 3D matrix of matrices

        self.stacked_weighted_matrices = self.stacked_matrices * weights_array
        self.weights_array = weights_array

        if not only_3d:
            self.weighted_matrices_dict = {}
            self.weighted_arrays_dict = {}
            for key, matrix_df in self.matrices_dict.items():
                weighted_matrix = matrix_df * weights_array
                self.weighted_matrices_dict[key] = weighted_matrix
                self.weighted_arrays_dict[key] = weighted_matrix.to_numpy()
