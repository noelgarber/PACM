# Defines the ConditionalMatrix and ConditionalMatrices classes

import numpy as np
import pandas as pd
import os
from functools import partial
from scipy.stats import barnard_exact
from scipy.optimize import minimize
from general_utils.general_utils import unravel_seqs, check_seq_lengths
from general_utils.matrix_utils import make_empty_matrix, collapse_phospho
from general_utils.user_helper_functions import get_thresholds
try:
    from Matrix_Generator.config_local import data_params, matrix_params, aa_equivalence_dict
except:
    from Matrix_Generator.config import data_params, matrix_params, aa_equivalence_dict

class ConditionalMatrix:
    '''
    Class that contains a position-weighted matrix (self.matrix_df) based on input data and a conditional type-position
    rule, e.g. position #1 = [D,E]
    '''
    def __init__(self, motif_length, source_df, data_params = data_params, matrix_params = matrix_params,
                 aa_equivalence_dict = aa_equivalence_dict):
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

        # Generate and assign the binding signal prediction matrix
        amino_acids = matrix_params.get("amino_acids")
        include_phospho = matrix_params.get("include_phospho")
        self.generate_signal_matrix(sequences_2d, mean_signal_values, amino_acids, include_phospho)

        # Generate and assign the suboptimal and forbidden element matrices for disfavoured residues
        barnard_alpha = matrix_params["barnard_alpha"]

        passing_mask = np.logical_and(pass_calls, qualifying_member_calls)
        passing_seqs_2d = sequences_2d[passing_mask]
        passing_signal_values = mean_signal_values[passing_mask]

        failed_mask = np.logical_and(~pass_calls, qualifying_member_calls)
        failed_seqs_2d = sequences_2d[failed_mask]
        failed_signal_values = mean_signal_values[failed_mask]

        self.generate_suboptimal_matrix(passing_seqs_2d, passing_signal_values, failed_seqs_2d, failed_signal_values,
                                        amino_acids, aa_equivalence_dict, barnard_alpha, include_phospho)

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

    def generate_signal_matrix(self, all_seqs_2d, all_signal_values, amino_acids, include_phospho):
        '''
        This function generates a matrix for predicting the binding signal that would be observed for a given peptide

        Args:
            all_seqs_2d (np.ndarray):                  all peptide sequences tested
            all_signal_values (np.ndarray):            corresponding signal values for all peptides tested

        Returns:
            None; assigns self.signal_values_matrix
        '''

        motif_length = all_seqs_2d.shape[1]

        # Generate the positive element matrix for predicting signal values
        signal_values_matrix = make_empty_matrix(motif_length, amino_acids)
        matrix_cols = list(signal_values_matrix.columns)

        all_signal_values[all_signal_values < 0] = 0

        for col_index, col_name in enumerate(matrix_cols):
            col_residues = all_seqs_2d[:, col_index]
            unique_residues = np.unique(col_residues)
            for aa in unique_residues:
                matches_aa = col_residues == aa
                signals_when_aa = all_signal_values[matches_aa]
                signal_values_matrix.at[aa, col_name] = signals_when_aa / motif_length

        self.signal_values_matrix = signal_values_matrix.astype("float32")
        if not include_phospho:
            collapse_phospho(self.signal_values_matrix, in_place=True)

    def generate_suboptimal_matrix(self, passing_seqs_2d, passing_signal_values, failed_seqs_2d, failed_signal_values,
                                   amino_acids, aa_equivalence_dict = aa_equivalence_dict, barnard_alpha = 0.2,
                                   include_phospho = False):
        '''
        This function generates a suboptimal element scoring matrix and a forbidden element matrix

        Args:
            passing_seqs_2d (np.ndarray):              peptide sequences that bind the protein of interest
            passing_signal_values (np.ndarray):        corresponding signal values for passing peptides
            failed_seqs_2d (np.ndarray):               peptide sequences that do NOT bind the protein of interest
            failed_signal_values (np.ndarray):         corresponding signal values for failing peptides
            aa_equivalence_dict (dict):                dictionary of amino acid --> tuple of 'equivalent' amino acids
            barnard_alpha (float):                     Barnard exact test threshold to contribute a trend to the matrix

        Returns:
            None; assigns self.suboptimal_elements_matrix and self.forbidden_elements_matrix
        '''

        motif_length = passing_seqs_2d.shape[1]

        # Generate the suboptimal and forbidden element matrices
        suboptimal_elements_matrix = make_empty_matrix(motif_length, amino_acids)
        forbidden_elements_matrix = make_empty_matrix(motif_length, amino_acids)
        matrix_cols = list(suboptimal_elements_matrix.columns)

        failed_signal_values[failed_signal_values < 0] = 0
        passing_signal_mean = passing_signal_values.mean()
        failed_signal_mean = failed_signal_values.mean()
        signal_ratio = failed_signal_mean / passing_signal_mean

        for col_index, col_name in enumerate(matrix_cols):
            passing_col = passing_seqs_2d[:,col_index]
            failing_col = failed_seqs_2d[:,col_index]
            unique_residues = np.unique(np.concatenate(passing_col, failing_col))

            for aa in unique_residues:
                aa_passing_count = np.sum(passing_col == aa)
                other_passing_count = len(passing_col) - aa_passing_count
                aa_failing_count = np.sum(failing_col == aa)
                other_failing_count = len(failing_col) - aa_failing_count
                contingency_table = [[aa_passing_count, other_passing_count],
                                     [aa_failing_count, other_failing_count]]

                pvalue_disfavoured = barnard_exact(contingency_table, alternative="less")
                if pvalue_disfavoured <= barnard_alpha:
                    suboptimal_elements_matrix.at[aa, col_name] = 1 - signal_ratio
                    if aa_passing_count == 0:
                        forbidden_elements_matrix.at[aa, col_name] = True
                    continue

                equivalent_residues = aa_equivalence_dict[aa]
                group_passing_count = np.sum(np.isin(passing_col, equivalent_residues))
                nongroup_passing_count = len(passing_col) - group_passing_count
                group_failing_count = np.sum(np.isin(failing_col, equivalent_residues))
                nongroup_failing_count = len(failing_col) - group_failing_count
                group_contingency_table = [[group_passing_count, nongroup_passing_count],
                                           [group_failing_count, nongroup_failing_count]]

                group_pvalue_disfavoured = barnard_exact(group_contingency_table, alternative="less")
                if group_pvalue_disfavoured <= barnard_alpha and pvalue_disfavoured < 1:
                    suboptimal_elements_matrix.at[aa, col_name] = 1 - signal_ratio
                    if aa_passing_count == 0 and group_passing_count == 0:
                        forbidden_elements_matrix.at[aa, col_name] = True
                    continue

        self.suboptimal_elements_matrix = suboptimal_elements_matrix.astype("float32")
        self.forbidden_elements_matrix = forbidden_elements_matrix.astype("float32")
        if not include_phospho:
            collapse_phospho(self.suboptimal_elements_matrix, in_place=True)
            collapse_phospho(self.forbidden_elements_matrix, in_place=True)

# --------------------------------------------------------------------------------------------------------------------

def get_mcc(predictions, actual_truths):
    '''
    Calculates the Matthews correlation coefficient for predicted and actual boolean arrays

    Args:
        predictions (np.ndarray):   array of boolean predictions; must match shape of actual_truths
        actual_truths (np.ndarray): array of boolean truth values; must match shape of predictions

    Returns:
        mcc (float): the Matthews correlation coefficient as a floating point value
    '''

    TP_count = np.logical_and(predictions, actual_truths).sum()
    FP_count = np.logical_and(predictions, ~actual_truths).sum()
    TN_count = np.logical_and(~predictions, ~actual_truths).sum()
    FN_count = np.logical_and(~predictions, actual_truths).sum()
    mcc_numerator = (TP_count*TN_count) - (FP_count*FN_count)
    mcc_denominator = np.sqrt((TP_count+FP_count) * (TP_count+FN_count) * (TN_count+FP_count) * (TN_count+FN_count))
    mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else np.nan

    return mcc

def negative_accuracy(thresholds, scores_arrays, passes_bools):
    '''
    Helper function for use during threshold optimization by ScoredPeptideResult.optimize_thresholds()

    Args:
        thresholds (np.ndarray):    array of thresholds of shape (score_type_count,)
        scores_arrays (np.ndarray): array of scores values of shape (datapoints_count, score_type_count)
        passes_bools (np.ndarray):  array of actual truth values of shape (datapoints_count,)

    Returns:
        negative_accuracy (float):  negative accuracy value that will be minimized in the minimization algorithm
    '''

    predictions_2d = scores_arrays > thresholds
    predictions = np.all(predictions_2d, axis=1)
    accuracies = np.equal(predictions, passes_bools)
    accuracy = np.mean(accuracies)

    return -accuracy  # Minimize negative accuracy to maximize actual accuracy

class ScoredPeptideResult:
    '''
    Class that represents the result of scoring peptides using ConditionalMatrices.score_peptides()
    '''
    def __init__(self, slice_scores_subsets, weights, predicted_signals_2d, suboptimal_scores_2d, forbidden_scores_2d):
        '''
        Initialization function to generate the score values and assign them to self

        Args:
            slice_scores_subsets (np.ndarray): array of span lengths in the motif to stratify scores by; e.g. if it is
                                               [6,7,2], then subset scores are derived for positions 1-6, 7:13, & 14:15
            weights (np.ndarray):              position weights previously applied to the matrices
            predicted_signals_2d (np.ndarray): unadjusted predicted signal values for each residue for each peptide
            suboptimal_scores_2d (np.ndarray): suboptimal element scores for each residue for each peptide
            forbidden_scores_2d (np.ndarray):  forbidden element scores for each residue for each peptide
        '''

        # Check validity of slice_scores_subsets
        if slice_scores_subsets is not None:
            if slice_scores_subsets.sum() != len(weights):
                raise ValueError(f"ScoredPeptideResult error: slice_scores_subsets sum ({slice_scores_subsets.sum()}) "
                                 f"does not match length of weights_array ({len(weights_array)})")

        # Assign predicted signals score values
        self.predicted_signals_2d = predicted_signals_2d
        self.predicted_signals_raw = predicted_signals_2d.sum(axis=1)
        self.signal_adjustment_factor = weights.mean()
        self.adjusted_predicted_signals = self.predicted_signals_raw / self.signal_adjustment_factor

        # Assign suboptimal element score values
        self.suboptimal_scores_2d = suboptimal_scores_2d
        self.suboptimal_scores = suboptimal_scores_2d.sum(axis=1)

        # Assign forbidden element score values
        self.forbidden_scores_2d = forbidden_scores_2d
        self.forbidden_scores = forbidden_scores_2d.sum(axis=1)

        # Assign sliced score values if slice_scores_subsets was given
        self.slice_scores_subsets = slice_scores_subsets
        if slice_scores_subsets is not None:
            end_position = 0
            sliced_predicted_signals = []
            sliced_suboptimal_scores = []
            sliced_forbidden_scores = []
            for subset in slice_scores_subsets:
                start_position = end_position
                end_position += subset
                subset_predicted_signals = predicted_signals_2d[:,start_position:end_position+1].sum(axis=1)
                sliced_predicted_signals.append(subset_predicted_signals)
                subset_suboptimal_scores = suboptimal_scores_2d[:,start_position:end_position+1].sum(axis=1)
                sliced_suboptimal_scores.append(subset_suboptimal_scores)
                subset_forbidden_scores = forbidden_scores_2d[:,start_position:end_position+1].sum(axis=1)
                sliced_forbidden_scores.append(subset_forbidden_scores)

            self.sliced_predicted_signals = sliced_predicted_signals
            self.sliced_suboptimal_scores = sliced_suboptimal_scores
            self.sliced_forbidden_scores = sliced_forbidden_scores

        self.optimized = False

    def get_stacked_scores(self):
        # Helper function that constructs a 2D array of scores values as columns

        scores = [self.adjusted_predicted_signals, self.suboptimal_scores * -1, self.forbidden_scores * -1]
        sign_mutlipliers = [1, -1, -1]

        if self.slice_scores_subsets is not None:
            for predicted_signals_slice in self.sliced_predicted_signals:
                scores.append(predicted_signals_slice)
                sign_mutlipliers.append(1)
            for suboptimal_scores_slice in self.sliced_suboptimal_scores:
                scores.append(suboptimal_scores_slice * -1)
                sign_mutlipliers.append(-1)
            for forbidden_scores_slice in self.sliced_forbidden_scores:
                scores.append(forbidden_scores_slice * -1)
                sign_mutlipliers.append(-1)

        sign_mutlipliers = np.array(sign_mutlipliers)
        stacked_scores = np.stack(scores).T

        return sign_mutlipliers, stacked_scores

    def optimize_thresholds(self, passes_bools):
        '''
        User-invoked optimization function to determine the optimal thresholds for the scores

        Args:
            passes_bools (np.ndarray):  array of actual truth values

        Returns:
            None; assigns results to self
        '''

        # Construct a 2D array of scores values as columns
        sign_mutlipliers, stacked_scores = self.get_stacked_scores()

        # Nelder-Mead optimization of thresholds
        initial_thresholds = np.median(stacked_scores, axis=0)
        optimization_function = partial(negative_accuracy, scores_arrays = stacked_scores, passes_bools = passes_bools)
        optimization_result = minimize(optimization_function, initial_thresholds, method = "Nelder-Mead")
        self.optimized_thresholds_signed = optimization_result.x
        self.optimized_thresholds = self.optimized_thresholds_signed * sign_mutlipliers
        self.sign_multipliers = sign_mutlipliers
        self.optimized_accuracy = optimization_result.fun * -1

        if optimization_result.status != 0:
            print(optimization_result.message)

        # Get predictions from optimized thresholds and calculate MCC
        optimized_predictions_2d = stacked_scores > optimized_thresholds_signed
        self.optimized_predictions = np.all(optimized_predictions_2d, axis=1)
        self.mcc = get_mcc(optimized_predictions, passes_bools)

        # Construct a user-readable dictionary of threshold values
        thresholds_dict = {"adjusted_predicted_signals": self.optimized_thresholds[0],
                           "suboptimal_scores": self.optimized_thresholds[1],
                           "forbidden_scores": self.optimized_thresholds[2]}

        if self.slice_scores_subsets is not None:
            current_start_idx = len(thresholds_dict)

            current_end_idx = current_start_idx + len(self.sliced_predicted_signals)
            thresholds_dict["sliced_predicted_signals"] = self.optimized_thresholds[current_start_idx:current_end_idx]

            current_start_idx = current_end_idx
            current_end_idx = current_start_idx + len(self.sliced_suboptimal_scores)
            thresholds_dict["sliced_suboptimal_scores"] = self.optimized_thresholds[current_start_idx:current_end_idx]

            current_start_idx = current_end_idx
            current_end_idx = current_start_idx + len(self.sliced_forbidden_scores)
            thresholds_dict["sliced_forbidden_scores"] = self.optimized_thresholds[current_start_idx:current_end_idx]

            thresholds_dict["slice_lengths"] = self.slice_scores_subsets

        self.thresholds_dict = thresholds_dict

        # Update the optimization status of the object
        self.optimized = True

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

        self.motif_length = motif_length

        if matrix_params.get("points_assignment_mode") == "thresholds":
            if not isinstance(matrix_params.get("thresholds_points_dict"), dict):
                matrix_params["thresholds_points_dict"] = get_thresholds(percentiles_dict, use_percentiles = True,
                                                                         show_guidance = True)

        # Declare dict where keys are position-type rules (e.g. "#1=Acidic") and values are corresponding weighted matrices
        self.conditional_matrix_dict = {}

        # For generating a 3D matrix, create an empty list to hold the matrices to stack, and the mapping
        self.residue_charac_dict = residue_charac_dict
        self.chemical_class_count = len(residue_charac_dict.keys())
        self.encoded_chemical_classes = {}
        self.chemical_class_decoder = {}
        signal_matrices_list = []
        suboptimal_matrices_list = []
        forbidden_matrices_list = []

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

                # Generate the matrix object
                conditional_matrix = ConditionalMatrix(motif_length, source_df, data_params, current_matrix_params)

                # Assign the matrix object to the dict of ConditionalMatrix objects
                dict_key_name = "#" + str(filter_position) + "=" + chemical_characteristic
                self.conditional_matrix_dict[dict_key_name] = conditional_matrix

                # Assign the constituent matrices to lists for 3D stacking
                signal_matrices_list.append(conditional_matrix.signal_values_matrix.to_numpy())
                suboptimal_matrices_list.append(conditional_matrix.suboptimal_elements_matrix.to_numpy())
                forbidden_matrices_list.append(conditional_matrix.forbidden_elements_matrix.to_numpy())

                # Assign index and columns objects; these are assumed to be the same for all matrices
                self.index = conditional_matrix.suboptimal_elements_matrix.index
                self.columns = conditional_matrix.suboptimal_elements_matrix.columns

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

        # Make 3D matrices
        stack_matrices(signal_matrices_list, suboptimal_matrices_list, forbidden_matrices_list)

        # Make array representations of the signal, suboptimal, and forbidden matrices
        make_unweighted_dicts(self.conditional_matrix_dict)

        # Apply weights
        weights_array = matrix_params.get("position_weights")
        if weights_array is not None:
            self.apply_weights(weights_array, only_3d = False)
        else:
            self.apply_weights(np.ones(motif_length), only_3d = False)

    def stack_matrices(self, signal_matrices_list, suboptimal_matrices_list, forbidden_matrices_list):
        # Helper function to make 3D matrices for rapid scoring

        self.stacked_signal_matrices = np.stack(signal_matrices_list)
        self.stacked_suboptimal_matrices = np.stack(suboptimal_matrices_list)
        self.stacked_forbidden_matrices = np.stack(forbidden_matrices_list)

    def make_unweighted_dicts(self, conditional_matrix_dict):
        # Helper function to make dataframe and array representations of the signal, suboptimal, and forbidden matrices

        self.unweighted_matrices_dicts = {"signal": {}, "suboptimal": {}, "forbidden": {}}
        self.unweighted_arrays_dicts = {"signal": {}, "suboptimal": {}, "forbidden": {}}

        for key, conditional_matrix in conditional_matrix_dict.items():
            self.unweighted_matrices_dicts["signal"][key] = conditional_matrix.signal_values_matrix
            self.unweighted_arrays_dicts["signal"][key] = conditional_matrix.signal_values_matrix.to_numpy()
            self.unweighted_matrices_dicts["suboptimal"][key] = conditional_matrix.suboptimal_elements_matrix
            self.unweighted_arrays_dicts["suboptimal"][key] = conditional_matrix.suboptimal_elements_matrix.to_numpy()
            self.unweighted_matrices_dicts["forbidden"][key] = conditional_matrix.forbidden_elements_matrix
            self.unweighted_arrays_dicts["forbidden"][key] = conditional_matrix.forbidden_elements_matrix.to_numpy()

    def apply_weights(self, weights, only_3d = True):
        # Method for assigning weights to the 3D matrix of matrices

        # Apply weights to 3D representations of matrices for rapid scoring
        self.stacked_signal_weighted = self.stacked_signal_matrices * weights
        self.stacked_suboptimal_weighted = self.stacked_suboptimal_matrices * weights
        self.stacked_forbidden_weighted = self.stacked_forbidden_matrices * weights
        self.weights_array = weights

        # Optionally also apply weights to the other formats
        if not only_3d:
            self.weighted_matrices_dicts = {}
            self.weighted_arrays_dicts = {}

            for matrix_type, matrix_dict in self.unweighted_matrices_dicts.items():
                weighted_matrix_dict = {}
                weighted_array_dict = {}
                for key, matrix_df in matrix_dict.items():
                    weighted_matrix_dict[key] = matrix_df * weights
                    weighted_array_dict[key] = matrix_df.to_numpy() * weights
                self.weighted_matrices_dicts[matrix_type] = weighted_matrix_dict
                self.weighted_arrays_dicts[matrix_type] = weighted_array_dict

    def save(self, output_folder, save_weighted = True):
        # User-called function to save the conditional matrices as CSVs to folders for both unweighted and weighted

        parent_folder = os.path.join(output_folder, "Conditional_Matrices")

        # Define unweighted matrix output paths
        unweighted_folder_paths = {}
        unweighted_parent = os.path.join(parent_folder, "Unweighted")
        unweighted_folder_paths["signal"] = os.path.join(unweighted_parent, "Signal_Matrices")
        unweighted_folder_paths["suboptimal"] = os.path.join(unweighted_parent, "Suboptimal_Matrices")
        unweighted_folder_paths["forbidden"] = os.path.join(unweighted_parent, "Forbidden_Matrices")
        for path in unweighted_folder_paths.values():
            os.makedirs(path) if not os.path.exists(path) else None

        # Save unweighted matrices
        for matrix_type, matrix_dict in self.unweighted_matrices_dicts.items():
            for key, matrix_df in matrix_dict.items():
                file_path = os.path.join(unweighted_folder_paths[matrix_type], key + ".csv")
                matrix_df.to_csv(file_path)

        if save_weighted:
            # Define weighted matrix output paths
            weighted_folder_paths = {}
            weighted_parent = os.path.join(parent_folder, "Weighted")
            weighted_folder_paths["signal"] = os.path.join(weighted_parent, "Weighted_Signal_Matrices")
            weighted_folder_paths["suboptimal"] = os.path.join(weighted_parent, "Weighted_Suboptimal_Matrices")
            weighted_folder_paths["forbidden"] = os.path.join(weighted_parent, "Weighted_Forbidden_Matrices")
            for path in weighted_folder_paths.values():
                os.makedirs(path) if not os.path.exists(path) else None

            # Save weighted matrices
            for matrix_type, matrix_dict in self.weighted_matrices_dicts.items():
                for key, matrix_df in matrix_dict.items():
                    file_path = os.path.join(weighted_folder_paths[matrix_type], key + ".csv")
                    matrix_df.to_csv(file_path)

        # Save output report
        output_report_path = os.path.join(parent_folder, "conditional_matrices_report.txt")
        with open(output_report_path, "w") as file:
            file.writelines(self.report)

        # Display saved message
        print(f"Saved unweighted matrices, weighted matrices, and output report to {parent_folder}")

    def score_peptides(self, sequences_2d, conditional_matrices, passes_bools = None, slice_scores_subsets = None,
                       use_weighted = False):
        '''
        Vectorized function to score amino acid sequences based on the dictionary of context-aware weighted matrices

        Args:
            sequences_2d (np.ndarray):                  unravelled peptide sequences to score
            conditional_matrices (ConditionalMatrices): conditional weighted matrices for scoring peptides
            passes_bools (np.ndarray):                  array of actual truth values of whether each peptide is a motif
            slice_scores_subsets (np.ndarray):          array of stretches of positions to stratify results into;
                                                        e.g. [6,7,2] is stratified into scores for positions
                                                        1-6, 7-13, & 14-15
            use_weighted (bool):                        whether to use conditional_matrices.stacked_weighted_matrices

        Returns:
            result (ScoredPeptideResult):               signal, suboptimal, and forbidden score values in 1D and 2D
        '''

        motif_length = sequences_2d.shape[1]

        # Get row indices for unique residues
        unique_residues = np.unique(sequences_2d)
        unique_residue_indices = conditional_matrices.index.get_indexer_for(unique_residues)

        if (unique_residue_indices == -1).any():
            failed_residues = unique_residues[unique_residue_indices == -1]
            err = f"score_seqs error: the following residues were not found by the matrix indexer: {failed_residues}"
            raise Exception(err)

        # Get the matrix row indices for all the residues
        aa_row_indices_2d = np.ones(shape=sequences_2d.shape, dtype=int) * -1
        for unique_residue, row_index in zip(unique_residues, unique_residue_indices):
            aa_row_indices_2d[sequences_2d == unique_residue] = row_index

        # Define residues flanking either side of the residues of interest; for out-of-bounds cases, use opposite side
        flanking_left_2d = np.concatenate((sequences_2d[:, 0:1], sequences_2d[:, 0:-1]), axis=1)
        flanking_right_2d = np.concatenate((sequences_2d[:, 1:], sequences_2d[:, -1:]), axis=1)

        # Get integer-encoded chemical classes for each residue
        left_encoded_classes_2d = np.zeros(flanking_left_2d.shape, dtype=int)
        right_encoded_classes_2d = np.zeros(flanking_right_2d.shape, dtype=int)
        for member_aa, encoded_class in conditional_matrices.encoded_chemical_classes.items():
            left_encoded_classes_2d[flanking_left_2d == member_aa] = encoded_class
            right_encoded_classes_2d[flanking_right_2d == member_aa] = encoded_class

        # Find the matrix identifier number (1st dim of 3D matrix) for each encoded class, depending on seq position
        encoded_positions = np.arange(motif_length) * conditional_matrices.chemical_class_count
        left_encoded_matrix_refs = left_encoded_classes_2d + encoded_positions
        right_encoded_matrix_refs = right_encoded_classes_2d + encoded_positions

        # Flatten the encoded matrix refs, which serve as the 1st dimension referring to 3D matrices
        left_encoded_matrix_refs_flattened = left_encoded_matrix_refs.flatten()
        right_encoded_matrix_refs_flattened = right_encoded_matrix_refs.flatten()

        # Flatten the amino acid row indices into a matching array serving as the 2nd dimension
        aa_row_indices_flattened = aa_row_indices_2d.flatten()

        # Tile the column indices into a matching array serving as the 3rd dimension
        column_indices = np.arange(motif_length)
        column_indices_tiled = np.tile(column_indices, len(sequences_2d))

        # Assign matrices to use for scoring
        if use_weighted:
            stacked_signal_matrices = conditional_matrices.stacked_signal_weighted
            stacked_suboptimal_matrices = conditional_matrices.stacked_suboptimal_weighted
            stacked_forbidden_matrices = conditional_matrices.stacked_forbidden_weighted
        else:
            stacked_signal_matrices = conditional_matrices.stacked_signal_matrices
            stacked_suboptimal_matrices = conditional_matrices.stacked_suboptimal_matrices
            stacked_forbidden_matrices = conditional_matrices.stacked_forbidden_matrices

        # Define dimensions for 3D matrix indexing
        shape_2d = sequences_2d.shape
        left_dim1 = left_encoded_matrix_refs_flattened
        right_dim1 = right_encoded_matrix_refs_flattened
        dim2 = aa_row_indices_flattened
        dim3 = column_indices_tiled

        # Calculate predicted signal values
        left_signal_2d = stacked_signal_matrices[left_dim1, dim2, dim3].reshape(shape_2d)
        right_signal_2d = stacked_signal_matrices[right_dim1, dim2, dim3].reshape(shape_2d)
        predicted_signals_2d = (left_signal_2d + right_signal_2d) / 2

        # Calculate suboptimal element scores
        left_suboptimal_2d = stacked_suboptimal_matrices[left_dim1, dim2, dim3].reshape(shape_2d)
        right_suboptimal_2d = stacked_suboptimal_matrices[right_dim1, dim2, dim3].reshape(shape_2d)
        suboptimal_scores_2d = (left_suboptimal_2d + right_suboptimal_2d) / 2

        # Calculate forbidden element scores
        left_forbidden_2d = stacked_forbidden_matrices[left_dim1, dim2, dim3].reshape(shape_2d)
        right_forbidden_2d = stacked_forbidden_matrices[right_dim1, dim2, dim3].reshape(shape_2d)
        forbidden_scores_2d = (left_forbidden_2d + right_forbidden_2d) / 2

        result = ScoredPeptideResult(slice_scores_subsets, self.weights_array, predicted_signals_2d,
                                     suboptimal_scores_2d, forbidden_scores_2d)
        if passes_bools is not None:
            result.optimize_thresholds(passes_bools)

        return result