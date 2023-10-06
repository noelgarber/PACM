# Defines the ConditionalMatrix and ConditionalMatrices classes

import numpy as np
import pandas as pd
import os
import warnings
import multiprocessing
from copy import deepcopy
from tqdm import trange
from functools import partial
from scipy.stats import barnard_exact, ttest_ind, ks_2samp
from general_utils.general_utils import unravel_seqs, check_seq_lengths
from general_utils.matrix_utils import make_empty_matrix, collapse_phospho
from general_utils.user_helper_functions import get_thresholds
from Matrix_Generator.ScoredPeptideResult import ScoredPeptideResult
try:
    from Matrix_Generator.config_local import data_params, matrix_params, aa_equivalence_dict, amino_acids_phos
except:
    from Matrix_Generator.config import data_params, matrix_params, aa_equivalence_dict, amino_acids_phos

def barnard_disfavoured(test_aa, aa_col_passing, aa_col_failing, equivalent_residues = None, alternative = "less"):
    '''
    Helper function that uses Barnard's exact test to determine if a test amino acid appears less often in a sliced col
    of peptide sequences for the passing peptides vs. the failing ones.

    Args:
        test_aa (str):                               single letter code for the amino acid to be tested
        aa_col_passing (np.ndarray):                 sliced col of peptide sequences that bind the target
        aa_col_failing (np.ndarray):                 sliced col of peptide sequences that do not bind the target
        equivalent_residues (np.ndarray|tuple|list): if given, residues are pooled with test_aa based on similarity

    Returns:
        barnard_pvalue (float): the p-value of the test where the alternative hypothesis is "less"
    '''

    if equivalent_residues is None:
        aa_passing_count = np.sum(aa_col_passing == test_aa)
        aa_failing_count = np.sum(aa_col_failing == test_aa)
    else:
        aa_passing_count = np.sum(np.isin(aa_col_passing, equivalent_residues))
        aa_failing_count = np.sum(np.isin(aa_col_failing, equivalent_residues))

    other_passing_count = len(aa_col_passing) - aa_passing_count
    other_failing_count = len(aa_col_failing) - aa_failing_count
    contingency_table = [[aa_passing_count, other_passing_count],
                         [aa_failing_count, other_failing_count]]

    barnard_pvalue = barnard_exact(contingency_table, alternative = alternative).pvalue

    return barnard_pvalue

def welch_ttest_catch(a, b, alternative = "less"):
    # Helper function that performs Welch's t-test on two independent samples where the alternative hypothesis is "less"

    with warnings.catch_warnings(record=True) as warning_list:
        result = ttest_ind(a, b, axis=None, equal_var=False, nan_policy="omit", alternative=alternative)
        pvalue = result.pvalue

        runtime_warnings = []
        for warning in warning_list:
            if issubclass(warning.category, RuntimeWarning):
                runtime_warnings.append(warning)

        if runtime_warnings and np.isfinite(pvalue):
                print(f"Welch's t-test gave the following warning: {str(warning.message)}")
                print(f"\tsample1: mean={a.mean()} | count={len(a)} | finite_count={np.isfinite(a).sum()}")
                print(f"\tsample2: mean={b.mean()} | count={len(b)} | finite_count={np.isfinite(b).sum()}")
                print(f"\tp-value: {pvalue}")

    return pvalue, runtime_warnings

class ConditionalMatrix:
    '''
    Class that contains a position-weighted matrix (self.matrix_df) based on input data and a conditional type-position
    rule, e.g. position #1 = [D,E]
    '''
    def __init__(self, motif_length, source_df, data_params = data_params, matrix_params = matrix_params,
                 aa_equivalence_dict = aa_equivalence_dict, reference_suboptimal_matrix = None):
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
            included_residues = matrix_params["amino_acids"]
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

        '''
        Generate and assign the binding signal prediction matrix
            - To do this, we use signal values from both passing and failing peptides that qualify under the current
              type-position rule (e.g. #1=Acidic). 
            - Including the failing peptides ensures that residues associated with 
              very low signal values are represented correctly. 
        '''
        amino_acids = matrix_params.get("amino_acids")
        include_phospho = matrix_params.get("include_phospho")
        qualifying_seqs_2d = sequences_2d[qualifying_member_calls] # includes both pass and fail
        qualifying_signals_2d = mean_signal_values[qualifying_member_calls] # includes both pass and fail

        self.generate_positive_matrix(qualifying_seqs_2d, qualifying_signals_2d, amino_acids, include_phospho)

        '''
        Generate and assign the suboptimal and forbidden element matrices for disfavoured residues
            - To do this, we first segregate passing and failing peptides
            - We then look for predictors of failure to bind and represent as suboptimal/forbidden element points
        '''
        barnard_alpha = matrix_params["barnard_alpha"]
        suboptimal_points_mode = matrix_params["suboptimal_points_mode"]
        min_aa_entries = matrix_params["min_aa_entries"]

        passing_mask = np.logical_and(pass_calls, qualifying_member_calls)
        passing_seqs_2d = sequences_2d[passing_mask]

        failed_mask = np.logical_and(~pass_calls, qualifying_member_calls)
        failed_seqs_2d = sequences_2d[failed_mask]

        self.generate_suboptimal_matrix(sequences_2d, mean_signal_values, passing_seqs_2d, failed_seqs_2d, amino_acids,
                                        aa_equivalence_dict, barnard_alpha, suboptimal_points_mode, min_aa_entries,
                                        reference_suboptimal_matrix, include_phospho)

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

    def generate_positive_matrix(self, all_seqs_2d, all_signal_values, amino_acids, include_phospho):
        '''
        This function generates a matrix for predicting the binding signal that would be observed for a given peptide

        Args:
            all_seqs_2d (np.ndarray):                  all peptide sequences tested
            all_signal_values (np.ndarray):            corresponding signal values for all peptides tested

        Returns:
            None; assigns self.positive_matrix
        '''

        motif_length = all_seqs_2d.shape[1]

        # Generate the positive element matrix for predicting signal values
        positive_matrix = make_empty_matrix(motif_length, amino_acids)
        matrix_cols = list(positive_matrix.columns)

        all_signal_values[all_signal_values < 0] = 0
        standardized_signal_values = all_signal_values / np.max(all_signal_values)

        for col_index, col_name in enumerate(matrix_cols):
            col_residues = all_seqs_2d[:, col_index]
            unique_residues = np.unique(col_residues)
            for aa in unique_residues:
                # Generate standardized positive matrix
                standardized_signals_when = standardized_signal_values[col_residues == aa]
                standardized_points = np.median(standardized_signals_when)
                positive_matrix.at[aa, col_name] = standardized_points

        self.positive_matrix = positive_matrix.astype("float32")
        if not include_phospho:
            collapse_phospho(self.positive_matrix, in_place=True)

    def generate_suboptimal_matrix(self, sequences_2d, signal_values, passing_seqs_2d, failed_seqs_2d,
                                   amino_acids, aa_equivalence_dict = aa_equivalence_dict, alpha = 0.2,
                                   suboptimal_points_mode = "counts", min_aa_entries = 4,
                                   reference_suboptimal_matrix = None, include_phospho = False):
        '''
        This function generates a suboptimal element scoring matrix and a forbidden element matrix

        Args:
            passing_seqs_2d (np.ndarray):               peptide sequences that bind the protein of interest
            passing_signal_values (np.ndarray):         corresponding signal values for passing peptides
            failed_seqs_2d (np.ndarray):                peptide sequences that do NOT bind the protein of interest
            failed_signal_values (np.ndarray):          corresponding signal values for failing peptides
            aa_equivalence_dict (dict):                 dictionary of amino acid --> tuple of 'equivalent' amino acids
            alpha (float):                              Barnard exact test threshold to contribute a trend to the matrix
            suboptimal_points_mode (str):               if "counts", points are assigned as ratio of % positive counts;
                                                        if "signal", points are assigned as ratio of signal values
            min_aa_entries (int):                       min number of qualifying peptides to derive suboptimal points
            reference_suboptimal_matrix (pd.DataFrame): baseline suboptimal element matrix, if known, for default values

        Returns:
            None; assigns self.suboptimal_elements_matrix and self.forbidden_elements_matrix
        '''

        motif_length = passing_seqs_2d.shape[1]

        # Generate the suboptimal and forbidden element matrices
        suboptimal_elements_matrix = make_empty_matrix(motif_length, amino_acids)
        forbidden_elements_matrix = make_empty_matrix(motif_length, amino_acids)
        matrix_cols = list(suboptimal_elements_matrix.columns)

        signal_values[signal_values < 0] = 0

        # Check suboptimal points mode
        if suboptimal_points_mode == "counts":
            use_counts = True
        elif suboptimal_points_mode == "signal" or suboptimal_points_mode == "signals":
            use_counts = False
        else:
            raise ValueError(f"got suboptimal_points_mode = {suboptimal_points_mode}, but must be `counts` or `signal`")

        # Loop over the matrix
        for col_index, col_name in enumerate(matrix_cols):
            passing_col = passing_seqs_2d[:,col_index]
            failing_col = failed_seqs_2d[:,col_index]
            full_col = sequences_2d[:,col_index]
            unique_residues = np.unique(full_col)

            for aa in unique_residues:
                # Test whether peptides with this aa at the current position have significantly lower signal values
                signals_while = signal_values[full_col == aa]
                signals_other = signal_values[full_col != aa]
                ttest_pvalue, _ = welch_ttest_catch(signals_while, signals_other)

                # Determine the signal ratio of peptides with this aa at the current position and peptides without it
                mean_signal_while = signals_while.mean()
                mean_signal_other = signals_other.mean()
                signal_ratio = mean_signal_while / mean_signal_other if mean_signal_other > 0 else np.nan

                # Find the pass rate ratio
                qualifying_count = np.sum(full_col == aa)
                if qualifying_count >= min_aa_entries and use_counts:
                    qualifying_pass_count = np.sum(passing_col == aa)
                    qualifying_fail_count = np.sum(failing_col == aa)
                    qualifying_count = qualifying_pass_count + qualifying_fail_count
                    other_pass_count = len(passing_col) - qualifying_pass_count
                    other_fail_count = len(failing_col) - qualifying_fail_count
                    other_count = other_pass_count + other_fail_count
                    qualifying_pass_rate = qualifying_pass_count / qualifying_count if qualifying_count != 0 else np.nan
                    other_pass_rate = other_pass_count / other_count if other_count != 0 else np.nan
                    pass_rate_ratio = qualifying_pass_rate / other_pass_rate # is less than 1 when aa is disfavoured
                    if not np.isfinite(pass_rate_ratio):
                        pass_rate_ratio = 1
                    use_default = False
                else:
                    pass_rate_ratio = 1 # prevents high variability when only a few example peptides exist
                    use_default = True

                # Test whether peptides with this aa at the current position are more likely to be classed as passing
                barnard_pvalue = barnard_disfavoured(aa, passing_col, failing_col)

                # Assign values to matrix only if one of the tests passes the alpha; if not, proceed to checking related
                if ttest_pvalue <= alpha or barnard_pvalue <= alpha:
                    if use_counts and pass_rate_ratio <= 1:
                        points_value = 1 - pass_rate_ratio
                    elif use_counts and use_default:
                        points_value = reference_suboptimal_matrix.at[aa, col_name]
                    elif use_counts:
                        points_value = 0
                    else:
                        points_value = 1 - signal_ratio if mean_signal_while < mean_signal_other else 0

                    suboptimal_elements_matrix.at[aa, col_name] = points_value
                    if np.sum(passing_col == aa) == 0:
                        forbidden_elements_matrix.at[aa, col_name] = 1
                    continue

                # Test whether peptides with similar residues at this position have significantly lower signal values
                equivalent_residues = aa_equivalence_dict[aa]
                has_equivalent_residue = np.isin(full_col, equivalent_residues)
                signals_while_group = signal_values[has_equivalent_residue]
                signals_while_nongroup = signal_values[~has_equivalent_residue]
                group_ttest_pvalue, _ = welch_ttest_catch(signals_while_group, signals_while_nongroup)

                # Determine the signal ratio of peptides with this aa at the current position and peptides without it
                mean_signal_group = signals_while_group.mean()
                mean_signal_nongroup = signals_while_nongroup.mean()
                signal_ratio_group = mean_signal_group / mean_signal_nongroup if mean_signal_nongroup > 0 else np.nan

                # Find the group pass rate ratio
                group_qualifying_count = np.sum(has_equivalent_residue)
                if group_qualifying_count >= min_aa_entries and use_counts:
                    group_passing_count = np.sum(np.isin(passing_col, equivalent_residues))
                    group_failing_count = np.sum(np.isin(failing_col, equivalent_residues))
                    group_count = group_passing_count + group_failing_count
                    nongroup_passing_count = len(passing_col) - group_passing_count
                    nongroup_failing_count = len(failing_col) - group_failing_count
                    nongroup_count = nongroup_passing_count + nongroup_failing_count
                    group_pass_rate = group_passing_count / group_count if group_count != 0 else np.nan
                    nongroup_pass_rate = nongroup_passing_count / nongroup_count if nongroup_count != 0 else np.nan
                    group_rate_ratio = group_pass_rate / nongroup_pass_rate # is less than 1 when aa is disfavoured
                    if not np.isfinite(group_rate_ratio):
                        group_rate_ratio = 1
                    group_use_default = False
                else:
                    group_rate_ratio = 1 # prevents high variability when only a few example peptides exist
                    group_use_default = True

                # Test whether peptides with similar residues at this position are more likely to be classed as passing
                group_barnard_pvalue = barnard_disfavoured(aa, passing_col, failing_col, equivalent_residues)

                # Assign values to matrix only if one of the group tests passes alpha
                if group_ttest_pvalue <= alpha or group_barnard_pvalue <= alpha:
                    if use_counts and group_rate_ratio <= 1:
                        both_less = pass_rate_ratio < 1 and group_rate_ratio < 1
                        less_intense_ratio = np.max([pass_rate_ratio, group_rate_ratio])
                        points_value = 1 - less_intense_ratio if both_less else 0
                    elif use_counts and group_use_default:
                        points_value = reference_suboptimal_matrix.at[aa, col_name]
                    elif use_counts:
                        points_value = 0
                    else:
                        both_less = mean_signal_while < mean_signal_other and mean_signal_group < mean_signal_nongroup
                        more_similar_ratio = np.max([signal_ratio, signal_ratio_group])
                        points_value = 1 - more_similar_ratio if both_less else 0
                    suboptimal_elements_matrix.at[aa, col_name] = points_value
                    if np.sum(np.isin(passing_col, equivalent_residues)) == 0:
                        forbidden_elements_matrix.at[aa, col_name] = 1
                    continue

        self.suboptimal_elements_matrix = suboptimal_elements_matrix.astype("float32")
        self.forbidden_elements_matrix = forbidden_elements_matrix.astype("float32")
        if not include_phospho:
            collapse_phospho(self.suboptimal_elements_matrix, in_place=True)
            collapse_phospho(self.forbidden_elements_matrix, in_place=True)

    def copy_matrix_col(self, col_idx):
        # User-initiated function for copying a specific column from each sub-matrix

        positive_matrix_col = self.positive_matrix.values[:,col_idx]
        suboptimal_matrix_col = self.suboptimal_elements_matrix.values[:,col_idx]
        forbidden_matrix_col = self.forbidden_elements_matrix.values[:,col_idx]

        return (positive_matrix_col, suboptimal_matrix_col, forbidden_matrix_col)

    def substitute_matrix_col(self, col_idx, substituted_positive_col, substituted_suboptimal_col,
                              substituted_forbidden_col, substitution_bools = (True, True, True)):
        # User-initiated function for substituting a specific column into each sub-matrix

        col = self.positive_matrix.columns[col_idx]

        if substitution_bools[0]:
            self.positive_matrix[col] = substituted_positive_col
        if substitution_bools[1]:
            self.suboptimal_elements_matrix[col] = substituted_suboptimal_col
        if substitution_bools[2]:
            self.forbidden_elements_matrix[col] = substituted_forbidden_col

    def set_positive(self, new_positive_matrix):
        self.positive_matrix = new_positive_matrix
        
    def set_suboptimal(self, new_suboptimal_matrix): 
        self.suboptimal_elements_matrix = new_suboptimal_matrix
        
    def set_forbidden(self, new_forbidden_matrix):
        self.forbidden_elements_matrix = new_forbidden_matrix

# --------------------------------------------------------------------------------------------------------------------

def substitute_forbidden(baseline_matrix, test_matrix, verbose = True):
    '''
    Statistically tests whether the values in cols of a test matrix differ from baseline;
    if not, baseline values are substituted into the test matrix

    Args:
        baseline_matrix (ConditionalMatrix): baseline matrix trained on all data
        test_matrix (ConditionalMatrix):     conditional matrix trained on a subset of the data
        verbose (bool):                      whether to print progress information about substitutions

    Returns:
        substitution_report (list):          descriptive report as a list of text lines
    '''

    # Get type-position rule for current conditional matrix (test_matrix)
    test_filter_position, test_included_residues = test_matrix.rule
    rule_str = f"#{test_filter_position}=[" + ",".join(test_included_residues) + "]"
    substitution_report = [f"{rule_str} Substantially Conditional Positions\n"]

    if verbose:
        print(f"---")
        print(f"Rule: {rule_str}")

    # Check if the test matrix is already the same as the baseline matrix
    same_positive_matrix = np.all(np.equal(baseline_matrix.positive_matrix.values,
                                           test_matrix.positive_matrix.values))
    same_suboptimal_matrix = np.all(np.equal(baseline_matrix.suboptimal_elements_matrix.values,
                                             test_matrix.suboptimal_elements_matrix.values))
    same_forbidden_matrix = np.all(np.equal(baseline_matrix.forbidden_elements_matrix.values,
                                            test_matrix.forbidden_elements_matrix.values))
    all_same_matrices = np.all([same_positive_matrix, same_suboptimal_matrix, same_forbidden_matrix])

    # If test matrix is the same as the baseline matrix, skip substitution
    if all_same_matrices:
        line = "\tAll conditional sub-matrices and unconditional baseline sub-matrices are already the same\n"
        substitution_report.append(line)
        print(line) if verbose else None
        return substitution_report
    elif same_forbidden_matrix:
        line = "\tForbidden conditional matrix and forbidden unconditional matrix are already the same\n"
        substitution_report.append(line)
        print(line) if verbose else None
        return substitution_report
    else: 
        baseline_forbidden_matrix = baseline_matrix.forbidden_elements_matrix
        test_matrix.set_forbidden(baseline_forbidden_matrix)
        line = "\tReplaced forbidden conditional matrix with forbidden unconditional matrix\n"
        substitution_report.append(line)
        print(line) if verbose else None
        return substitution_report

# --------------------------------------------------------------------------------------------------------------------

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

        # Generate a baseline non-conditional matrix for statistical comparisons
        self.generate_baseline_matrix(motif_length, source_df, data_params, matrix_params)

        # Declare dict where keys are position-type rules (e.g. "#1=Acidic") and values are corresponding weighted matrices
        self.conditional_matrix_dict = {}

        # For generating a 3D matrix, create an empty list to hold the matrices to stack, and the mapping
        self.residue_charac_dict = residue_charac_dict
        self.chemical_class_count = len(residue_charac_dict.keys())
        self.encoded_chemical_classes = {}
        self.chemical_class_decoder = {}
        positive_matrices_list = []
        suboptimal_matrices_list = []
        forbidden_matrices_list = []

        # Iterate over dict of chemical characteristic --> list of member amino acids (e.g. "Acidic" --> ["D","E"]
        self.sufficient_keys = []
        self.insufficient_keys = []
        self.report = ["Conditional Matrix Generation Report\n"]

        rule_tuples = self.get_rule_tuples(residue_charac_dict, motif_length)
        results_list = self.generate_matrices(rule_tuples, motif_length, source_df, data_params, matrix_params)
        replace_forbidden = matrix_params["replace_forbidden"]
        self.substitution_reports = {}
        substitution_report = ["---\n"]

        for results in results_list:
            conditional_matrix, dict_key_name, report_line = results

            # Assign results to dict; replace forbidden matrix if necessary
            if replace_forbidden:
                current_report = substitute_forbidden(self.baseline_matrix, conditional_matrix)
                substitution_report.extend(current_report)
                substitution_report.append("---\n")

            self.conditional_matrix_dict[dict_key_name] = conditional_matrix
            self.substitution_reports[dict_key_name] = substitution_report

            # Assign the constituent matrices to lists for 3D stacking
            positive_matrices_list.append(conditional_matrix.positive_matrix.to_numpy())
            suboptimal_matrices_list.append(conditional_matrix.suboptimal_elements_matrix.to_numpy())
            forbidden_matrices_list.append(conditional_matrix.forbidden_elements_matrix.to_numpy())

            # Assign index and columns objects; these are assumed to be the same for all matrices
            self.index = conditional_matrix.suboptimal_elements_matrix.index
            self.columns = conditional_matrix.suboptimal_elements_matrix.columns

            # Assemble the status reports
            self.report.append(report_line)
            sufficient_seqs = conditional_matrix.sufficient_seqs
            if not sufficient_seqs:
                self.insufficient_keys.append(dict_key_name)
            else:
                self.sufficient_keys.append(dict_key_name)

        # Print the matrix generation report
        print("".join(self.report))

        # Make 3D matrices
        self.stack_matrices(positive_matrices_list, suboptimal_matrices_list, forbidden_matrices_list)

        # Make array representations of the signal, suboptimal, and forbidden matrices
        self.make_unweighted_dicts(self.conditional_matrix_dict)

        # Apply weights
        weights_array = matrix_params.get("position_weights")
        if weights_array is not None:
            self.apply_weights(weights_array, only_3d = False)
        else:
            self.apply_weights(np.ones(motif_length), only_3d = False)

    def get_rule_tuples(self, residue_charac_dict, motif_length):
        # Helper function that generates a list of rule tuples to be used by generate_matrices()

        rule_tuples = []
        for i, (chemical_characteristic, member_list) in enumerate(residue_charac_dict.items()):
            # Map the encodings for the chemical classes
            for aa in member_list:
                self.encoded_chemical_classes[aa] = i
            self.chemical_class_decoder[i] = chemical_characteristic

            # Iterate over columns for the weighted matrix (position numbers)
            for filter_position in np.arange(1, motif_length + 1):
                rule_tuple = (member_list, filter_position, chemical_characteristic)
                rule_tuples.append(rule_tuple)

        return rule_tuples

    def generate_baseline_matrix(self, motif_length, source_df, data_params, matrix_params,
                                 amino_acids = amino_acids_phos):
        # Simple function to generate a non-conditional baseline matrix for statistical comparison

        current_matrix_params = matrix_params.copy()
        current_matrix_params["min_members"] = np.inf # forces non-conditional matrix to be generated
        current_matrix_params["included_residues"] = amino_acids
        current_matrix_params["position_for_filtering"] = 0

        # Generate the matrix object and dict key name
        self.baseline_matrix = ConditionalMatrix(motif_length, source_df, data_params, current_matrix_params)

    def generate_matrices(self, rule_tuples, motif_length, source_df, data_params, matrix_params):
        '''
        Parallelized application of self.generate_matrix()

        Args:
            rule_tuples (list):       list of tuples of (member amino acid list, filter position, chemical characteristic)
            motif_length (int):       length of the motif being studied
            source_df (pd.DataFrame): dataframe of source data for building matrices
            data_params (dict):       user-defined params for accessing source data, as described in ConditionalMatrix()
            matrix_params (dict):     shared dict of matrix params defined by the user as described in ConditionalMatrix()

        Returns:
            results_list (list):  list of results tuples of (conditional_matrix, dict_key_name, report_line)
        '''

        cpus_to_use = os.cpu_count() - 1
        pool = multiprocessing.Pool(processes=cpus_to_use)

        process_partial = partial(self.generate_matrix, motif_length = motif_length, source_df = source_df,
                                  data_params = data_params, matrix_params = matrix_params)

        results_list = []

        desc = "Generating conditional matrices (signal, suboptimal, & forbidden types)..."
        with trange(len(rule_tuples), desc=desc) as pbar:
            for results in pool.imap_unordered(process_partial, rule_tuples):
                results_list.append(results)
                pbar.update()

        pool.close()
        pool.join()

        return results_list

    def generate_matrix(self, rule_tuple, motif_length, source_df, data_params, matrix_params):
        '''
        Function for generating an individual conditional matrix according to a filter position and member aa list

        Args:
            rule_tuple (tuple):       tuple of (amino acid member list, filter position, chemical characteristic)
                                      representing the current type/position rule, e.g. #1=Acidic
            motif_length (int):       length of the motif being studied
            source_df (pd.DataFrame): dataframe of source data for building matrices
            data_params (dict):       user-defined params for accessing source data, as described in ConditionalMatrix()
            matrix_params (dict):     user-defined params for generating matrices, as described in ConditionalMatrix()

        Returns:
            conditional_matrix (ConditionalMatrix): conditional matrix object for the current type/position rule
            dict_key_name (str):                    key name representing the current type/position rule
            report_line (str):                      status line for the output report
        '''

        member_list, filter_position, chemical_characteristic = rule_tuple

        # Assign parameters for the current type-position rule
        current_matrix_params = matrix_params.copy()
        current_matrix_params["included_residues"] = member_list
        current_matrix_params["position_for_filtering"] = filter_position

        # Generate the matrix object and dict key name
        conditional_matrix = ConditionalMatrix(motif_length, source_df, data_params, current_matrix_params)
        dict_key_name = "#" + str(filter_position) + "=" + chemical_characteristic

        # Display a warning message if insufficient seqs were passed
        sufficient_seqs = conditional_matrix.sufficient_seqs
        if not sufficient_seqs:
            report_line = f"Matrix status for {dict_key_name}: not enough source seqs meeting rule; defaulting to all"
        else:
            report_line = f"Matrix status for {dict_key_name}: OK"

        return (conditional_matrix, dict_key_name, report_line + f"\n")

    def stack_matrices(self, positive_matrices_list, suboptimal_matrices_list, forbidden_matrices_list):
        # Helper function to make 3D matrices for rapid scoring

        self.stacked_positive_matrices = np.stack(positive_matrices_list)
        self.stacked_suboptimal_matrices = np.stack(suboptimal_matrices_list)
        self.stacked_forbidden_matrices = np.stack(forbidden_matrices_list)

    def make_unweighted_dicts(self, conditional_matrix_dict):
        # Helper function to make dataframe and array representations of the signal, suboptimal, and forbidden matrices

        self.unweighted_matrices_dicts = {"signal": {}, "suboptimal": {}, "forbidden": {}}
        self.unweighted_arrays_dicts = {"signal": {}, "suboptimal": {}, "forbidden": {}}

        for key, conditional_matrix in conditional_matrix_dict.items():
            self.unweighted_matrices_dicts["signal"][key] = conditional_matrix.positive_matrix
            self.unweighted_arrays_dicts["signal"][key] = conditional_matrix.positive_matrix.to_numpy()
            self.unweighted_matrices_dicts["suboptimal"][key] = conditional_matrix.suboptimal_elements_matrix
            self.unweighted_arrays_dicts["suboptimal"][key] = conditional_matrix.suboptimal_elements_matrix.to_numpy()
            self.unweighted_matrices_dicts["forbidden"][key] = conditional_matrix.forbidden_elements_matrix
            self.unweighted_arrays_dicts["forbidden"][key] = conditional_matrix.forbidden_elements_matrix.to_numpy()

    def apply_weights(self, weights, only_3d = True):
        # Method for assigning weights to the 3D matrix of matrices

        # Apply weights to 3D representations of matrices for rapid scoring
        self.stacked_positive_weighted = self.stacked_positive_matrices * weights
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
        path_list = []
        for matrix_type, matrix_dict in self.unweighted_matrices_dicts.items():
            for key, matrix_df in matrix_dict.items():
                file_path = os.path.join(unweighted_folder_paths[matrix_type], key + ".csv")
                if file_path not in path_list:
                    matrix_df.to_csv(file_path)
                    path_list.append(file_path)
                else:
                    raise Exception("Error saving matrices: tried to save over a previously saved file!")

        # Save unweighted baseline matrices
        baseline_positive_path = os.path.join(unweighted_parent, "baseline_positive_matrix.csv")
        self.baseline_matrix.positive_matrix.to_csv(baseline_positive_path)
        baseline_suboptimal_path = os.path.join(unweighted_parent, "baseline_suboptimal_matrix.csv")
        self.baseline_matrix.suboptimal_elements_matrix.to_csv(baseline_suboptimal_path)
        baseline_forbidden_path = os.path.join(unweighted_parent, "baseline_forbidden_matrix.csv")
        self.baseline_matrix.forbidden_elements_matrix.to_csv(baseline_forbidden_path)

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
        final_report = deepcopy(self.report)
        for rule_name, substitution_report in self.substitution_reports.items():
            final_report.append("---\n")
            final_report.append(f"{rule_name} Substitution Report\n")
            final_report.extend(substitution_report)
        with open(output_report_path, "w") as file:
            file.writelines(final_report)

        # Display saved message
        print(f"Saved unweighted matrices, weighted matrices, and output report to {parent_folder}")

    def score_peptides(self, sequences_2d, actual_truths, signal_values = None, use_r2 = False, slice_scores_subsets = None,
                       use_weighted = False, precision_recall_path = None):
        '''
        Vectorized function to score amino acid sequences based on the dictionary of context-aware weighted matrices

        Args:
            sequences_2d (np.ndarray):                  unravelled peptide sequences to score
            actual_truths (np.ndarray):                 array of boolean calls for whether peptides bind in experiments
            signal_values (np.ndarray):                 array of binding signal values for peptides against protein bait(s)
            use_r2 (bool):                              whether to maximize linear R2 (if False, f1-score will be maximized)
            conditional_matrices (ConditionalMatrices): conditional weighted matrices for scoring peptides
            slice_scores_subsets (np.ndarray):          array of stretches of positions to stratify results into;
                                                        e.g. [6,7,2] is stratified into scores for positions
                                                        1-6, 7-13, & 14-15
            use_weighted (bool):                        whether to use conditional_matrices.stacked_weighted_matrices
            precision_recall_path (str):                desired file path for saving the precision/recall graph

        Returns:
            result (ScoredPeptideResult):               signal, suboptimal, and forbidden score values in 1D and 2D
        '''

        motif_length = sequences_2d.shape[1]

        # Get row indices for unique residues
        unique_residues = np.unique(sequences_2d)
        unique_residue_indices = self.index.get_indexer_for(unique_residues)

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
        for member_aa, encoded_class in self.encoded_chemical_classes.items():
            left_encoded_classes_2d[flanking_left_2d == member_aa] = encoded_class
            right_encoded_classes_2d[flanking_right_2d == member_aa] = encoded_class

        # Find the matrix identifier number (1st dim of 3D matrix) for each encoded class, depending on seq position
        encoded_positions = np.arange(motif_length) * self.chemical_class_count
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
            stacked_positive_matrices = self.stacked_positive_weighted
            stacked_suboptimal_matrices = self.stacked_suboptimal_weighted
            stacked_forbidden_matrices = self.stacked_forbidden_weighted
        else:
            stacked_positive_matrices = self.stacked_positive_matrices
            stacked_suboptimal_matrices = self.stacked_suboptimal_matrices
            stacked_forbidden_matrices = self.stacked_forbidden_matrices

        # Define dimensions for 3D matrix indexing
        shape_2d = sequences_2d.shape
        left_dim1 = left_encoded_matrix_refs_flattened
        right_dim1 = right_encoded_matrix_refs_flattened
        dim2 = aa_row_indices_flattened
        dim3 = column_indices_tiled

        # Calculate predicted signal values
        left_positive_2d = stacked_positive_matrices[left_dim1, dim2, dim3].reshape(shape_2d)
        right_positive_2d = stacked_positive_matrices[right_dim1, dim2, dim3].reshape(shape_2d)
        positive_scores_2d = (left_positive_2d + right_positive_2d) / 2

        # Calculate suboptimal element scores
        left_suboptimal_2d = stacked_suboptimal_matrices[left_dim1, dim2, dim3].reshape(shape_2d)
        right_suboptimal_2d = stacked_suboptimal_matrices[right_dim1, dim2, dim3].reshape(shape_2d)
        suboptimal_scores_2d = (left_suboptimal_2d + right_suboptimal_2d) / 2

        # Calculate forbidden element scores
        left_forbidden_2d = stacked_forbidden_matrices[left_dim1, dim2, dim3].reshape(shape_2d)
        right_forbidden_2d = stacked_forbidden_matrices[right_dim1, dim2, dim3].reshape(shape_2d)
        forbidden_scores_2d = (left_forbidden_2d + right_forbidden_2d) / 2

        result = ScoredPeptideResult(sequences_2d, slice_scores_subsets, positive_scores_2d, suboptimal_scores_2d,
                                     forbidden_scores_2d, actual_truths, signal_values, use_r2,
                                     fig_path = precision_recall_path)

        return result