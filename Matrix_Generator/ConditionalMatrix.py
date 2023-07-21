# Defines the ConditionalMatrix and ConditionalMatrices classes

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from functools import partial
from math import e
from scipy.stats import fisher_exact
from general_utils.general_utils import unravel_seqs, check_seq_lengths
from general_utils.matrix_utils import increment_matrix, make_empty_matrix, collapse_phospho
from general_utils.user_helper_functions import get_thresholds
from Matrix_Generator.config import data_params, matrix_params

class ConditionalMatrix:
    '''
    Class that contains a position-weighted matrix (self.matrix_df) based on input data and a conditional type-position
    rule, e.g. position #1 = [D,E]
    '''
    def __init__(self, motif_length, source_df, residue_charac_dict,
                 data_params = data_params, matrix_params = matrix_params):
        '''
        Function for initializing unadjusted conditional matrices from source peptide data,
        based on type-position rules (e.g. #1=Acidic)

        Args:
            motif_length (int):                the length of the motif being assessed
            source_df (pd.DataFrame):          dataframe containing peptide-protein binding data
            residue_charac_dict:               dict of amino acid chemical characteristics
            data_params (dict):                dictionary of data-specific params described in config.py
            matrix_params (dict):              dictionary of matrix-specific params described in config.py
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
            self.decrement_negatives(pass_calls, qualifying_member_calls, sequences_2d, masked_sequences_2d,
                                     mean_points, residue_charac_dict)

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
                            masked_sequences_2d, mean_positive_points, residue_charac_dict, alpha = 0.1):
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

                # Only use significantly disfavoured residues for decrementation, and set others to X (not used)
                for aa_group in residue_charac_dict.values():
                    # Perform Fisher's exact test on pooled amino acids from this group
                    aa_group = np.array(aa_group, dtype="U")

                    negative_group_count = np.sum(np.isin(negative_masked_col, aa_group))
                    negative_nongroup_count = negative_count - negative_group_count
                    positive_group_count = np.sum(np.isin(positive_masked_col, aa_group))
                    positive_nongroup_count = positive_count - positive_group_count

                    group_contingency_table = [[negative_group_count, negative_nongroup_count],
                                               [positive_group_count, positive_nongroup_count]]
                    group_odds_ratio, group_p_value = fisher_exact(group_contingency_table)

                    for aa in aa_group:
                        # Perform Fisher's exact test on this amino acid specifically
                        negative_aa_count = np.sum(negative_masked_col == aa)
                        negative_other_count = negative_count - negative_aa_count

                        positive_aa_count = np.sum(positive_masked_col == aa)
                        positive_other_count = positive_count - positive_aa_count

                        contingency_table = [[negative_aa_count, negative_other_count],
                                             [positive_aa_count, positive_other_count]]
                        odds_ratio, p_value = fisher_exact(contingency_table)

                        # Set non-disfavoured residues to X such that they will not be used to decrement the matrix
                        if (positive_negative_ratio * negative_aa_count) < positive_aa_count:
                            negative_masked_col[negative_masked_col == aa] = "X"
                        elif p_value > alpha and group_p_value > alpha:
                            negative_masked_col[negative_masked_col == aa] = "X"

                inverse_masked_sequences_2d[:, col_number] = negative_masked_col

            self.matrix_df = increment_matrix(None, self.matrix_df, inverse_masked_sequences_2d,
                                              enforced_points = negative_points, points_mode = "enforced_points")

def get_sigmoid(x, k, inflection):
    # Basic function that applies a sigmoid function to x value

    base_value = 1 / (1 + e ** (k * inflection))
    upper_value = 1 / (1 + e ** (-k * (1 - inflection))) - base_value
    y = (1 / (1 + e ** (-k * (abs(x) - inflection))) - base_value) / upper_value

    if x < 0:
        y = y * -1 # apply original sign of x
        
    return y

def apply_sigmoid(matrix_df, strength = 1, inflection = 0.5):
    '''
    Function for scaling matrix values by a sigmoid function, suppressing small values and enhancing big ones

    Args:
        matrix_df (pd.DataFrame): the matrix to apply the sigmoid function to
        strength (int|float):     scales the function severity; must be > 1
        inflection (int/float):   inflection point between 0 and 1

    Returns:
        None: operation is performed in-place
    '''

    k = strength * 10

    sigmoid_function = partial(get_sigmoid, k = k, inflection = inflection)
    sigmoid_matrix_df = matrix_df.applymap(sigmoid_function)
    sigmoid_matrix_df = sigmoid_matrix_df.round(2)

    return sigmoid_matrix_df

class ConditionalMatrices:
    '''
    Class that contains a set of conditional matrices defined using ConditionalMatrix()
    '''
    def __init__(self, motif_length, source_df, percentiles_dict, residue_charac_dict,
                 data_params = data_params, matrix_params = matrix_params):
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

        # Get parameters for whether to use sigmoid-scaled matrices
        use_sigmoid = matrix_params.get("use_sigmoid")
        # Generate sigmoid-scaled version
        sigmoid_strength = matrix_params.get("sigmoid_strength")
        if sigmoid_strength is None:
            sigmoid_strength = 1

        sigmoid_inflection = matrix_params.get("sigmoid_inflection")
        if sigmoid_inflection is None:
            sigmoid_inflection = 0.5

        self.sigmoid_strength = sigmoid_strength
        self.sigmoid_inflection = sigmoid_inflection

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
                conditional_matrix = ConditionalMatrix(motif_length, source_df, residue_charac_dict,
                                                       data_params, current_matrix_params)
                current_matrix = conditional_matrix.matrix_df

                # Standardize the weighted matrix so that the max value is 1
                max_values = np.max(current_matrix.values, axis=0)
                max_values = np.maximum(max_values, 1)  # prevents divide-by-zero errors
                current_matrix /= max_values

                # If sigmoid scaling is enabled, apply the sigmoid function
                if use_sigmoid:
                    current_matrix = apply_sigmoid(current_matrix, sigmoid_strength, sigmoid_inflection)

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

    def save(self, output_folder):
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

    def save_sigmoid_plot(self, output_folder):
        # User-called function to save a plot of the sigmoid function used to adjust the matrix points values

        k = self.sigmoid_strength * 10
        inflection = self.sigmoid_inflection

        # Generate the graph contents
        x_values = np.linspace(0, 1, 101)
        y_values = np.array([get_sigmoid(x, k, inflection) for x in x_values])

        # Create the plot
        plt.plot(x_values, y_values)
        plt.xlabel("Original Score Value")
        plt.ylabel("Sigmoid-Adjusted Score Value")
        plt.title("Sigmoid Scaling of Conditional Matrix Points")
        plt.grid(True)

        # Save the figure as a PNG file
        plt.savefig(os.path.join(output_folder, "sigmoid_points_plot.png"), dpi=600)