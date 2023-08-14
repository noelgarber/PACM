# Defines the ConditionalMatrix and ConditionalMatrices classes

import numpy as np
import pandas as pd
import os
from general_utils.general_utils import unravel_seqs, check_seq_lengths
from general_utils.matrix_utils import make_empty_matrix, collapse_phospho
from general_utils.user_helper_functions import get_thresholds
try:
    from Matrix_Generator.config_local import data_params, matrix_params, aa_equivalence_dict
except:
    from Matrix_Generator.config import data_params, matrix_params, aa_equivalence_dict

class ConditionalMatrix:
    '''
    Describes a position-weighted suboptimal element matrix (self.matrix_df) based on input data and a conditional
    type-position rule, e.g. position #1 = ["D","E"].

    The underlying principle is to determine which suboptimal residues are 'lethal', which are disfavoured, and how many
    non-lethal disfavoured residues are required to collectively kill a putative peptide's binding to target proteins.
    '''

    def __init__(self, motif_length, source_df, data_params = data_params,
                 matrix_params = matrix_params, aa_equivalence_dict = aa_equivalence_dict):
        '''
        Function for initializing unadjusted conditional matrices from source peptide data,
        based on type-position rules (e.g. #1=Acidic)

        Args:
            motif_length (int):                the length of the motif being assessed
            source_df (pd.DataFrame):          dataframe containing peptide-protein binding data
            data_params (dict):                dictionary of data-specific params described in config.py
            matrix_params (dict):              dictionary of matrix-specific params described in config.py
            aa_equivalence_dict (dict):        dictionary of similar amino acids described in config.py
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
        pass_calls = np.equal(pass_values, pass_str)
        index_for_filtering = position_for_filtering - 1
        get_nth_position = np.vectorize(lambda x: x[index_for_filtering] if len(x) >= index_for_filtering + 1 else "")
        filter_position_chars = get_nth_position(sequences)
        qualifying_member_calls = np.isin(filter_position_chars, included_residues)

        # Assign points to the suboptimal element matrix based on which amino acids are more or less disfavoured
        masked_sequences_2d = sequences_2d[qualifying_member_calls]
        masked_signal_values = mean_signal_values[qualifying_member_calls]
        masked_pass_calls = pass_calls[qualifying_member_calls]
        self.increment_suboptimal(masked_sequences_2d, masked_signal_values, masked_pass_calls, aa_equivalence_dict)

        # Optionally combine phospho and non-phospho residue rows in the matrix
        include_phospho = matrix_params.get("include_phospho")
        if not include_phospho:
            collapse_phospho(self.matrix_df, in_place=True)

        # Optionally set the filtering position to zero
        clear_filtering_column = matrix_params.get("clear_filtering_column")
        if clear_filtering_column:
            matrix_df["#" + str(position_for_filtering)] = 0

        # Convert to float32 and round values
        self.matrix_df = self.matrix_df.astype("float32")
        self.matrix_df = self.matrix_df.round(2)

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

    def increment_suboptimal(self, sequences_2d, signal_values, pass_calls, aa_equivalence_dict = aa_equivalence_dict):
        '''
        Function to increment the conditional suboptimal element matrix based on disfavoured amino acids by position

        Args:
            sequences_2d (np.ndarray):  array of sequences where each row is a peptide as an array of amino acids
            signal_values (np.ndarray): signal values for each peptide
            pass_calls (np.ndarray):    pass/fail calls as bools for each peptide
            aa_equivalence_dict (dict): used for pooling similar residues when testing whether residues are disfavoured

        Returns:
            None; operation is performed in-place
        '''

        signal_values[signal_values < 0] = 0
        index_for_filtering = self.rule[0] - 1

        # Iterate over the matrix columns representing motif positions
        cols = list(self.matrix_df.columns)
        cols.pop(index_for_filtering)

        for col_index, col in enumerate(self.matrix_df.columns):
            col_residues = sequences_2d[:, col_index]

            # Iterate over possible amino acids in the matrix index
            for amino_acid in self.matrix_df.index:
                peptides_with_aa = np.sum(col_residues == amino_acid)

                if peptides_with_aa > 0:
                    # Get disfavourability ratio
                    residue_matches_aa = col_residues == amino_acid
                    matching_signal_values = signal_values[residue_matches_aa]
                    other_signal_values = signal_values[~residue_matches_aa]
                    disfavoured_ratio = matching_signal_values.mean() / signal_values.mean()
                    suboptimal_points = (1 - disfavoured_ratio) ** 2
                    if matching_signal_values.mean() < other_signal_values.mean():
                        self.matrix_df.at[amino_acid, col] = suboptimal_points

# --------------------------------------------------------------------------------------------------------------------

class ConditionalMatrices:
    '''
    Class that contains a set of conditional matrices defined using ConditionalMatrix()
    '''
    def __init__(self, motif_length, source_df, percentiles_dict, residue_charac_dict,
                 data_params = data_params, matrix_params = matrix_params, aa_equivalence_dict = aa_equivalence_dict):
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

        if matrix_params.get("points_assignment_mode") == "thresholds":
            if not isinstance(matrix_params.get("thresholds_points_dict"), dict):
                matrix_params["thresholds_points_dict"] = get_thresholds(percentiles_dict, use_percentiles = True,
                                                                         show_guidance = True)

        # Declare dict where keys are position-type rules (e.g. "#1=Acidic") and values are corresponding weighted matrices
        self.matrices_dict = {}

        # For generating a 3D matrix, create an empty list to hold the matrices to stack, and the mapping
        self.residue_charac_dict = residue_charac_dict
        self.chemical_class_count = len(residue_charac_dict.keys())
        self.encoded_chemical_classes = {}
        self.chemical_class_decoder = {}
        matrices_list = []

        # Iterate over dict of chemical characteristic --> list of member amino acids (e.g. "Acidic" --> ["D","E"]
        self.sufficient_keys = []
        self.insufficient_keys = []
        self.report = ["Conditional Matrix Generation Report\n\n"]
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
                conditional_matrix = ConditionalMatrix(motif_length, source_df, data_params, current_matrix_params,
                                                       aa_equivalence_dict)
                current_matrix = conditional_matrix.matrix_df

                # Assign the weighted matrix to the dictionary
                dict_key_name = "#" + str(filter_position) + "=" + chemical_characteristic
                self.matrices_dict[dict_key_name] = current_matrix
                matrices_list.append(current_matrix)

                # Display a warning message if insufficient seqs were passed
                sufficient_seqs = conditional_matrix.sufficient_seqs
                if not sufficient_seqs:
                    line = f"Matrix status for {dict_key_name}: not enough source seqs meeting rule; defaulting to all"
                    print(line)
                    self.report.append(line+f"\n")
                    self.insufficient_keys.append(dict_key_name)
                else:
                    line = f"Matrix status for {dict_key_name}: OK"
                    print(line)
                    self.report.append(line+f"\n")
                    self.sufficient_keys.append(dict_key_name)

        # Make an array representation of matrices_dict
        self.index = matrices_list[0].index
        self.columns = matrices_list[0].columns
        self.matrix_arrays_dict = {}
        for key, matrix_df in self.matrices_dict.items():
            self.matrix_arrays_dict[key] = matrix_df.to_numpy()

        # Make the 3D matrix
        self.stacked_matrices = np.stack(matrices_list)

        # Apply weights
        weights_array = matrix_params.get("position_weights")
        if weights_array is not None:
            self.apply_weights(weights_array, only_3d = False)

    def apply_weights(self, weights_array, only_3d = True):
        # Method for assigning weights to the 3D matrix of matrices

        self.stacked_weighted_matrices = self.stacked_matrices * weights_array
        self.weights_array = weights_array

        if not only_3d:
            self.weighted_matrices_dict = {}
            self.weighted_arrays_dict = {}
            for key, matrix_df in self.matrices_dict.items():
                weighted_matrix = matrix_df * weights_array
                weighted_matrix = weighted_matrix.round(2)
                self.weighted_matrices_dict[key] = weighted_matrix
                self.weighted_arrays_dict[key] = weighted_matrix.to_numpy()

    def save(self, output_folder, save_weighted = True):
        # User-called function to save the conditional matrices as CSVs to folders for both unweighted and weighted

        parent_folder = os.path.join(output_folder, "Conditional_Matrices")

        # Save unweighted matrices
        unweighted_folder = os.path.join(parent_folder, "Unweighted")
        if not os.path.exists(unweighted_folder):
            os.makedirs(unweighted_folder)
        for key, unweighted_matrix in self.matrices_dict.items():
            file_path = os.path.join(unweighted_folder, key + ".csv")
            unweighted_matrix.to_csv(file_path)

        # Save weighted matrices
        if save_weighted:
            weighted_folder = os.path.join(parent_folder, "Weighted")
            if not os.path.exists(weighted_folder):
                os.makedirs(weighted_folder)
            for key, weighted_matrix in self.weighted_matrices_dict.items():
                file_path = os.path.join(weighted_folder, key + ".csv")
                weighted_matrix.to_csv(file_path)

        # Save output report
        output_report_path = os.path.join(parent_folder, "conditional_matrices_report.txt")
        with open(output_report_path, "w") as file:
            file.writelines(self.report)

        # Display saved message
        unweighted_count = len(self.matrices_dict)
        weighted_count = len(self.weighted_matrices_dict)
        print(f"Saved {unweighted_count} unweighted matrices, {weighted_count} weighted matrices,",
              f"and output report to {parent_folder}")