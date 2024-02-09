# Defines the ConditionalMatrix and ConditionalMatrices classes

import numpy as np
import pandas as pd
import os
import multiprocessing
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import trange
from functools import partial
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import mannwhitneyu
from general_utils.general_utils import unravel_seqs, check_seq_lengths
from general_utils.matrix_utils import make_empty_matrix
from general_utils.user_helper_functions import get_thresholds
from Matrix_Generator.ScoredPeptideResult import ScoredPeptideResult
try:
    from Matrix_Generator.config_local import data_params, matrix_params, aa_equivalence_dict, amino_acids_phos
except:
    from Matrix_Generator.config import data_params, matrix_params, aa_equivalence_dict, amino_acids_phos

def get_yule_q(aa_col_passing, aa_col_failing, test_aa = None, equivalent_residues = None):
    '''
    Helper function that constructs a contingency table to calculate Yule's Q coefficient

    Args:
        aa_col_passing (np.ndarray):                 sliced col of peptide sequences that bind the target
        aa_col_failing (np.ndarray):                 sliced col of peptide sequences that do not bind the target
        test_aa (str):                               amino acid to be tested; if None, equivalent_residues must be given
        equivalent_residues (np.ndarray|tuple|list): if given, residues are pooled with test_aa based on similarity

    Returns:
        yule_q (float):                   Yule's Q coefficient; -1 is perfectly suboptimal, +1 is perfectly optimal
    '''

    '''
                                  Contingency Table Setup
                            |--------------------------------------------|
                            |     Current AA     :       Other AAs       |
    - - - - - - - - - - - - |--------------------|-----------------------|
    Pass (interacting)      |  aa_passing_count  |  other_passing_count  |
    - - - - - - - - - - - - |--------------------|-----------------------|
    Fail (non-interacting)  |  aa_failing_count  |  other_failing_count  |
    - - - - - - - - - - - - |--------------------|-----------------------|
    '''

    if equivalent_residues is None:
        aa_passing_count = np.sum(aa_col_passing == test_aa)
        aa_failing_count = np.sum(aa_col_failing == test_aa)
    else:
        aa_passing_count = np.sum(np.isin(aa_col_passing, equivalent_residues))
        aa_failing_count = np.sum(np.isin(aa_col_failing, equivalent_residues))

    other_passing_count = len(aa_col_passing) - aa_passing_count
    other_failing_count = len(aa_col_failing) - aa_failing_count

    # Calculate Yule's Q coefficient; -1 is a perfect negative association and +1 is a perfect positive association
    yule_numerator = aa_passing_count * other_failing_count - other_passing_count * aa_failing_count
    yule_denominator = aa_passing_count * other_failing_count + other_passing_count * aa_failing_count
    yule_q = yule_numerator / yule_denominator if yule_denominator > 0 else 0

    return yule_q

class ConditionalMatrix:
    '''
    Class that contains a position-weighted matrix (self.matrix_df) based on input data and a conditional type-position
    rule, e.g. position #1 = [D,E]
    '''
    def __init__(self, motif_length, source_df, data_params = data_params, matrix_params = matrix_params,
                 aa_equivalence_dict = aa_equivalence_dict, baseline_matrix = None):
        '''
        Function for initializing unadjusted conditional matrices from source peptide data,
        based on type-position rules (e.g. #1=Acidic)

        Args:
            motif_length (int):                  length of the motif being assessed
            source_df (pd.DataFrame):            dataframe containing peptide-protein binding data
            residue_charac_dict:                 dict of amino acid chemical characteristics
            data_params (dict):                  dictionary of data-specific params described in config.py
            matrix_params (dict):                dictionary of matrix-specific params described in config.py
            baseline_matrix (ConditionalMatrix): baseline unconditional matrix (no filter);
                                                 used when making the forbidden elements matrix
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

        # Separate sequences by passing or failing and whether they are qualifying members
        passing_mask = np.logical_and(pass_calls, qualifying_member_calls)
        passing_seqs_2d = sequences_2d[passing_mask]
        passing_signals = mean_signal_values[passing_mask]

        failed_mask = np.logical_and(~pass_calls, qualifying_member_calls)
        failed_seqs_2d = sequences_2d[failed_mask]
        failed_signals = mean_signal_values[failed_mask]

        '''
        Generate and assign the binding signal prediction matrix
            - To do this, we use signal values from both passing and failing peptides that qualify under the current
              type-position rule (e.g. #1=Acidic). 
            - Including the failing peptides ensures that residues associated with 
              very low signal values are represented correctly. 
        '''
        mann_whitney_alpha = matrix_params["mann_whitney_alpha"]
        min_aa_entries = matrix_params["min_aa_entries"]
        amino_acids = matrix_params.get("amino_acids")
        include_phospho = matrix_params.get("include_phospho")
        generate_individual_regressions = matrix_params.get("generate_individual_regressions")
        ignore_failed_peptides = matrix_params.get("ignore_failed_peptides")

        qualifying_seqs_2d = sequences_2d[qualifying_member_calls] # includes both pass and fail
        qualifying_signals = mean_signal_values[qualifying_member_calls] # includes both pass and fail

        if ignore_failed_peptides:
            positive_training_seqs_2d = passing_seqs_2d
            positive_training_signals = passing_signals
        else:
            positive_training_seqs_2d = qualifying_seqs_2d
            positive_training_signals = qualifying_signals

        self.generate_positive_matrix(positive_training_seqs_2d, positive_training_signals, amino_acids,
                                      include_phospho, aa_equivalence_dict, mann_whitney_alpha, min_aa_entries,
                                      baseline_matrix, generate_individual_regressions, generate_individual_regressions,
                                      plot_unweighted_points = False)

        '''
        Generate and assign the suboptimal and forbidden element matrices for disfavoured residues
            - To do this, we first segregate passing and failing peptides
            - We then look for predictors of failure to bind and represent as suboptimal/forbidden element points
        '''
        forbidden_threshold = matrix_params["forbidden_threshold"]
        forced_required_dict = matrix_params["forced_required_dict"] # forced required residues at defined positions
        always_assign_sub = matrix_params["always_assign_suboptimal"]

        self.generate_suboptimal_matrix(qualifying_seqs_2d, qualifying_signals, passing_seqs_2d, failed_seqs_2d,
                                        amino_acids, aa_equivalence_dict, min_aa_entries, forbidden_threshold,
                                        forced_required_dict, baseline_matrix, include_phospho, always_assign_sub)

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

    def generate_positive_matrix(self, seqs_2d, signal_values, amino_acids, include_phospho = False,
                                 aa_equivalence_dict = aa_equivalence_dict, mann_whitney_alpha = 0.1,
                                 min_aa_entries = 4, baseline_matrix = None, generate_multivariate_model = False,
                                 generate_linear_model = False, plot_unweighted_points = False):
        '''
        This function generates a matrix for predicting the binding signal that would be observed for a given peptide

        Args:
            seqs_2d (np.ndarray):                peptide sequences used for training the positive matrix
            signal_values (np.ndarray):          corresponding signal values for training peptides
            amino_acids (tuple|list):            amino acid alphabet to use
            include_phospho (bool):              whether to include phosphorylated residues in the matrix
            aa_equivalence_dict (dict):          dictionary of amino acid --> tuple of 'equivalent' amino acids
            mann_whitney_alpha (float):          alpha value to use for Mann-Whitney U test
            min_aa_entries (int):                min number of qualifying peptides to derive points from
            baseline_matrix (ConditionalMatrix): baseline matrix to reference for unrepresented residues in subgroup
            generate_multivariate_model (bool):  whether to generate a multivariate LinearRegression() model (log signal)
            generate_linear_model (bool):        whether to generate a point-sum unweighted LinearRegression() model
            plot_unweighted_points (bool):       whether to plot unweighted positive element points sums against signals
            ignore_failed_peptides (bool):       whether to only use signals from passing peptides for positive matrix
            preview_scatter_plot (bool):         whether to show a scatter plot of summed positive points against signals

        Returns:
            None; assigns self.positive_matrix
        '''

        motif_length = seqs_2d.shape[1]

        # Generate the positive element matrix for predicting signal values
        positive_matrix = make_empty_matrix(motif_length, amino_acids)
        matrix_cols = list(positive_matrix.columns)

        signal_values[signal_values < 0] = 0
        standardized_signal_values = signal_values / np.max(signal_values)

        # Iterate over columns to assign points
        for col_index, col_name in enumerate(matrix_cols):
            col_residues = seqs_2d[:, col_index]
            unique_residues = np.unique(col_residues)
            for aa in unique_residues:
                # Generate standardized positive matrix
                mask = np.char.equal(col_residues, aa)
                standardized_signals_when = standardized_signal_values[mask]
                standardized_signals_other = standardized_signal_values[~mask]

                # Calculate correlation coefficient from Mann-Whitney U statistic
                r_alone = 0
                if len(standardized_signals_when) > min_aa_entries and len(standardized_signals_other) > 0:
                    u_statistic, pvalue = mannwhitneyu(standardized_signals_when, standardized_signals_other)
                    n1n2 = len(standardized_signals_when) * len(standardized_signals_other)
                    r_alone = (u_statistic / n1n2) - ((n1n2 - u_statistic) / n1n2)
                    if pvalue <= mann_whitney_alpha:
                        positive_matrix.at[aa, col_name] = r_alone
                        continue

                # If the individual amino acid failed, group with similar residues and retry
                equivalent_residues = aa_equivalence_dict[aa]
                has_equivalent_residue = np.isin(col_residues, equivalent_residues)
                signals_while_group = standardized_signal_values[has_equivalent_residue]
                signals_while_nongroup = standardized_signal_values[~has_equivalent_residue]

                # Since the residue did not pass the Mann-Whitney test alone, we group it with similar residues
                if len(signals_while_group) > 0 and len(signals_while_nongroup) > 0:
                    group_u_statistic, group_pvalue = mannwhitneyu(signals_while_group, signals_while_nongroup)
                    group_n1n2 = len(signals_while_group) * len(signals_while_nongroup)
                    if group_pvalue <= mann_whitney_alpha:
                        r_group = (group_u_statistic / group_n1n2) - ((group_n1n2 - group_u_statistic) / group_n1n2)

                        # Check that the direction of correlation is the same for the group and the individual residue
                        both_plus = r_alone > 0, r_group > 0
                        both_minus = r_alone < 0, r_group < 0
                        signs_match = both_plus or both_minus

                        if signs_match:
                            positive_matrix.at[aa, col_name] = r_group
                            continue

                positive_matrix.at[aa, col_name] = 0 # triggered if no passing conditions are met

        # For unrepresented residue-position groups, fill in zeros using equivalent residues as necessary
        baseline_exists = True if baseline_matrix is not None else False
        if baseline_exists:
            baseline_index = baseline_matrix.positive_matrix.index
        else:
            baseline_index = []

        for col_index, col_name in enumerate(matrix_cols):
            col_residues = seqs_2d[:,col_index]
            unique_residues = np.unique(col_residues)
            unrepresented_residues = [aa for aa in amino_acids if aa not in unique_residues]

            # Iterate over unrepresented amino acids
            for aa in unrepresented_residues:
                # Handle suboptimal elements matrix entry
                current_points = positive_matrix.at[aa, col_name]
                if current_points == 0:
                    # Check if there is a nonzero value from baseline_matrix to obtain
                    if baseline_exists and aa in baseline_index:
                        baseline_points = baseline_matrix.positive_matrix.at[aa, col_name]
                        if baseline_points != 0:
                            positive_matrix.at[aa, col_name] = baseline_points
                            continue

                    # If no nonzero value exists in the baseline matrix, try to infer one from similar residues
                    equivalent_residues = aa_equivalence_dict[aa]
                    equivalent_points_list = []
                    for equivalent_aa in equivalent_residues:
                        if aa != equivalent_aa:
                            current_equivalent_points = positive_matrix.at[equivalent_aa, col_name]
                            if current_equivalent_points != 0:
                                equivalent_points_list.append(current_equivalent_points)
                    if len(equivalent_points_list) > 0:
                        new_points = np.mean(equivalent_points_list)
                        positive_matrix.at[aa, col_name] = new_points

        # Back-calculate positive element scores for input peptides if needed
        if generate_multivariate_model or generate_linear_model or plot_unweighted_points:
            log_signal_values = np.log(signal_values + 1)
            positive_matrix_arr = positive_matrix.to_numpy()

            col_indices = np.tile(np.arange(seqs_2d.shape[1]), seqs_2d.shape[0])
            row_indices = positive_matrix.index.get_indexer_for(seqs_2d.flatten())
            flattened_points = positive_matrix_arr[row_indices, col_indices]
            raw_positive_scores_2d = flattened_points.reshape(seqs_2d.shape)
        else:
            raw_positive_scores_2d, log_signal_values = None, None

        # Plot positive score sums against signal values
        if plot_unweighted_points:
            print("\t\t\tCurrent rule:", self.rule)
            plt.scatter(raw_positive_scores_2d.sum(axis=1), standardized_signal_values)
            plt.xlabel("Unweighted Positive Element Scores")
            plt.ylabel("Relative Signal Values")
            plt.show()

        # Check the final R2 correlations for a multivariate log-linear (exponential) model
        if generate_multivariate_model:
            model = LinearRegression()
            model.fit(raw_positive_scores_2d, log_signal_values)
            self.log_multivariate_positive_model = model
            predicted_signal_values = model.predict(raw_positive_scores_2d)
            self.positive_multivariate_r2 = r2_score(log_signal_values, predicted_signal_values)
        else:
            self.log_multivariate_positive_model, self.positive_multivariate_r2 = None, None

        # Check the final R2 correlations for an unweighted-sum log-linear (exponential) model
        if generate_linear_model:
            model = LinearRegression()
            model.fit(raw_positive_scores_2d.sum(axis=1).reshape(-1,1), log_signal_values)
            self.unweighted_positive_model = model
            predicted_signal_values = model.predict(raw_positive_scores_2d.sum(axis=1).reshape(-1,1))
            self.unweighted_positive_r2 = r2_score(log_signal_values, predicted_signal_values)
        else:
            self.unweighted_positive_model, self.unweighted_positive_r2 = None, None

        # Collapse phospho if necessary
        positive_matrix = positive_matrix.astype("float32")

        if not include_phospho:
            pairs = [("B","S"), ("J","T"), ("O","Y")]
            for phospho_residue, corresponding_residue in pairs:
                phospho_points_arr = positive_matrix.loc[phospho_residue].to_numpy()
                corresponding_points_arr = positive_matrix.loc[corresponding_residue].to_numpy()

                # Take larger effect sizes
                stacked_arr = np.stack([phospho_points_arr, corresponding_points_arr])
                row_indices = np.nanargmax(np.abs(stacked_arr), axis=0)
                col_indices = np.arange(len(phospho_points_arr))
                best_vals = stacked_arr[row_indices,col_indices]

                # Assign back to dataframe
                positive_matrix.loc[corresponding_residue] = best_vals
                positive_matrix.drop(phospho_residue, axis=0, inplace=True)

        # Assign to self
        self.positive_matrix = positive_matrix

    def generate_suboptimal_matrix(self, sequences_2d, signal_values, passing_seqs_2d, failed_seqs_2d, amino_acids,
                                   aa_equivalence_dict = aa_equivalence_dict, min_aa_entries = 4,
                                   forbidden_threshold = 3, forced_required_dict = None, baseline_matrix = None,
                                   include_phospho = False, always_assign_value = False):
        '''
        This function generates a suboptimal element scoring matrix and a forbidden element matrix

        Args:
            passing_seqs_2d (np.ndarray):               peptide sequences that bind the protein of interest
            passing_signal_values (np.ndarray):         corresponding signal values for passing peptides
            failed_seqs_2d (np.ndarray):                peptide sequences that do NOT bind the protein of interest
            failed_signal_values (np.ndarray):          corresponding signal values for failing peptides
            aa_equivalence_dict (dict):                 dictionary of amino acid --> tuple of 'equivalent' amino acids
            min_aa_entries (int):                       min number of qualifying peptides to derive points from
            forbidden_threshold (int):                  number of peptides that must possess a particular residue
                                                        or group of residues before it can be considered "forbidden"
            forced_required_dict (dict):                dictionary of position indices --> specified required residues
            baseline_matrix (ConditionalMatrix):        baseline matrix; used for unrepresented residues
            include_phospho (bool):                     whether to include phospho-residues in the matrices
            always_assign_value (bool):                 if True, then the equivalent residue group Q value is used

        Returns:
            None; assigns self.suboptimal_elements_matrix and self.forbidden_elements_matrix
        '''

        baseline_matrix_exists = True if baseline_matrix is not None else False
        motif_length = passing_seqs_2d.shape[1]

        # Generate the suboptimal and forbidden element matrices
        suboptimal_elements_matrix = make_empty_matrix(motif_length, amino_acids)
        forbidden_elements_matrix = make_empty_matrix(motif_length, amino_acids)
        matrix_cols = list(suboptimal_elements_matrix.columns)

        signal_values[signal_values < 0] = 0

        # Loop over the matrix
        for col_index, col_name in enumerate(matrix_cols):
            passing_col = passing_seqs_2d[:,col_index]
            failing_col = failed_seqs_2d[:,col_index]
            full_col = sequences_2d[:,col_index]
            unique_residues = np.unique(full_col)

            for aa in unique_residues:
                # Test whether peptides with this aa at the current position are more likely to be classed as passing
                yule_q = get_yule_q(passing_col, failing_col, test_aa = aa)
                above_min_aa = np.sum(full_col == aa) >= min_aa_entries
                appears_forbidden = np.sum(passing_col == aa) == 0 and np.sum(failing_col == aa) >= forbidden_threshold

                if above_min_aa:
                    '''
                    Yule's Q coefficient ranges from -1 to +1.
                    When Q approaches -1, it means the current residue is disfavoured. 
                    When Q approaches +1, it means the current residue is favoured. 
                    '''
                    points_value = -yule_q if yule_q <= 0 else 0
                    suboptimal_elements_matrix.at[aa, col_name] = points_value
                    if appears_forbidden:
                        forbidden_elements_matrix.at[aa, col_name] = 1
                    continue

                # Test whether peptides with similar residues at this position have significantly lower signal values
                equivalent_residues = aa_equivalence_dict[aa]
                group_above_min_aa = np.sum(np.isin(full_col, equivalent_residues)) >= min_aa_entries

                if group_above_min_aa or always_assign_value:
                    group_q = get_yule_q(passing_col, failing_col, equivalent_residues = equivalent_residues)
                    consistent_sign = (yule_q >= 0 and group_q >= 0) or (yule_q <= 0 and group_q <= 0)

                    never_passes = np.sum(np.isin(passing_col, equivalent_residues)) == 0
                    fails_exceed_threshold = np.sum(np.isin(failing_col, equivalent_residues)) >= forbidden_threshold
                    group_appears_forbidden = never_passes and fails_exceed_threshold

                    if consistent_sign:
                        points_value = -group_q if group_q <= 0 else 0
                        suboptimal_elements_matrix.at[aa, col_name] = points_value
                        if group_appears_forbidden:
                            forbidden_elements_matrix.at[aa, col_name] = 1
                    elif always_assign_value:
                        points_value = -group_q if np.abs(group_q) < np.abs(yule_q) else -yule_q
                        points_value = points_value if points_value >= 0 else 0
                        suboptimal_elements_matrix.at[aa, col_name] = points_value

                elif baseline_matrix_exists:
                    if aa in baseline_matrix.suboptimal_elements_matrix.index:
                        suboptimal_val = baseline_matrix.suboptimal_elements_matrix.at[aa, col_name]
                        suboptimal_elements_matrix.at[aa, col_name] = suboptimal_val
                        if appears_forbidden:
                            forbidden_val = baseline_matrix.forbidden_elements_matrix.at[aa, col_name]
                            forbidden_elements_matrix.at[aa, col_name] = forbidden_val

        # For unrepresented residue-position groups, fill in zeros using equivalent residues as necessary
        for col_index, col_name in enumerate(matrix_cols):
            full_col = sequences_2d[:,col_index]
            unique_residues = np.unique(full_col)
            unrepresented_residues = [aa for aa in suboptimal_elements_matrix.index if aa not in unique_residues]

            # Iterate over unrepresented amino acids
            for aa in unrepresented_residues:
                # Handle suboptimal elements matrix entry
                current_suboptimal_points = suboptimal_elements_matrix.at[aa, col_name]
                if current_suboptimal_points == 0:
                    equivalent_residues = aa_equivalence_dict[aa]
                    equivalent_points_list = []
                    for equivalent_aa in equivalent_residues:
                        if aa != equivalent_aa:
                            current_equivalent_points = suboptimal_elements_matrix.at[equivalent_aa, col_name]
                            equivalent_points_list.append(current_equivalent_points)
                    if len(equivalent_points_list) > 0:
                        new_points = np.mean(equivalent_points_list)
                        suboptimal_elements_matrix.at[aa, col_name] = new_points

                # Handle forbidden elements matrix entry
                current_forbidden_points = forbidden_elements_matrix.at[aa, col_name]
                if current_forbidden_points == 0:
                    equivalent_residues = aa_equivalence_dict[aa]
                    all_equivalent_forbidden = True
                    for equivalent_aa in equivalent_residues:
                        if aa != equivalent_aa:
                            current_equivalent_forbidden = forbidden_elements_matrix.at[equivalent_aa, col_name]
                            if current_equivalent_forbidden == 0:
                                all_equivalent_forbidden = False
                    if all_equivalent_forbidden:
                        for equivalent_aa in equivalent_residues:
                            if aa != equivalent_aa:
                                forbidden_elements_matrix.at[equivalent_aa, col_name] = 1

        # Apply forced_required_dict to forbidden elements matrix
        if forced_required_dict is not None:
            for position_idx, required_residues in forced_required_dict.items():
                col_name = matrix_cols[position_idx]
                forbidden_residues = [aa for aa in amino_acids if aa not in required_residues]
                forbidden_elements_matrix.loc[forbidden_residues, col_name] = 1

        # Collapse phospho if necessary
        if not include_phospho:
            for phospho_residue, corresponding_residue in [("B","S"), ("J","T"), ("O","Y")]:
                for col_idx, col_name in enumerate(matrix_cols):
                    phospho_suboptimal_points = suboptimal_elements_matrix.at[phospho_residue, col_name]
                    corresponding_suboptimal_points = suboptimal_elements_matrix.at[corresponding_residue, col_name]
                    phospho_smaller_effect = np.abs(phospho_suboptimal_points) < np.abs(corresponding_suboptimal_points)
                    if corresponding_suboptimal_points == 0 or phospho_smaller_effect:
                        suboptimal_elements_matrix.at[corresponding_residue, col_name] = phospho_suboptimal_points

                    phospho_forbidden_points = forbidden_elements_matrix.at[phospho_residue, col_name]
                    if phospho_forbidden_points == 0:
                        forbidden_elements_matrix.at[corresponding_residue, col_name] = 0

            suboptimal_elements_matrix.drop(["B","J","O"], axis=0, inplace=True)
            forbidden_elements_matrix.drop(["B","J","O"], axis=0, inplace=True)

        self.suboptimal_elements_matrix = suboptimal_elements_matrix.astype("float32")
        self.forbidden_elements_matrix = forbidden_elements_matrix.astype("float32")

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
    def __init__(self, motif_length, source_df, percentiles_dict, residue_charac_dict, output_folder,
                 data_params = data_params, matrix_params = matrix_params, test_df = None):
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

        self.suppress_positive_positions = matrix_params["suppress_positive_positions"]
        self.suppress_suboptimal_positions = matrix_params["suppress_suboptimal_positions"]
        self.suppress_forbidden_positions = matrix_params["suppress_forbidden_positions"]

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
        positive_matrices_list = []
        suboptimal_matrices_list = []
        forbidden_matrices_list = []

        # Iterate over dict of chemical characteristic --> list of member amino acids (e.g. "Acidic" --> ["D","E"]
        self.sufficient_keys = []
        self.insufficient_keys = []
        self.report = ["Conditional Matrix Generation Report\n"]

        rule_tuples = self.get_rule_tuples(residue_charac_dict, motif_length)
        results_list = self.generate_matrices(rule_tuples, motif_length, source_df, data_params, matrix_params,
                                              self.baseline_matrix)
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

        # Apply weights if given
        weights_array = matrix_params.get("position_weights")
        if weights_array is not None:
            self.apply_weights(weights_array, weights_array, weights_array, weights_array, only_3d = False)

        # Optimize weights and apply scores to source data
        slice_scores_subsets = matrix_params.get("slice_scores_subsets")
        pass_values = source_df[data_params["bait_pass_col"]].to_numpy()
        actual_truths = np.equal(pass_values, data_params["pass_str"])

        bait_signal_cols = list(set([col for cols in data_params["bait_cols_dict"].values() for col in cols]))
        mean_signal_values = source_df[bait_signal_cols].values.mean(axis=1)

        seq_col = data_params["seq_col"]
        seqs = source_df[seq_col].values.astype("<U")
        convert_phospho = not matrix_params.get("include_phospho")
        seqs_2d = unravel_seqs(seqs, motif_length, convert_phospho = convert_phospho)

        # Get predefined weights if they exist
        predefined_weights = matrix_params.get("predefined_weights")

        # Extract test set info if test_df exists
        precision_recall_path = os.path.join(output_folder, "precision_recall_graph.pdf")
        if test_df is not None:
            test_seqs = test_df[seq_col].values.astype("<U")
            test_seqs_2d = unravel_seqs(test_seqs, motif_length, convert_phospho = convert_phospho)
            test_pass_values = test_df[data_params["bait_pass_col"]].to_numpy()
            test_actual_truths = np.equal(test_pass_values, data_params["pass_str"])
            test_mean_signals = test_df[bait_signal_cols].values.mean(axis=1)
        else:
            test_seqs_2d, test_actual_truths, test_mean_signals = None, None, None

        # Optimize scoring weights
        self.scored_result = self.optimize_scoring_weights(seqs_2d, actual_truths, mean_signal_values,
                                                           slice_scores_subsets, precision_recall_path,
                                                           output_folder, test_seqs_2d, test_actual_truths,
                                                           test_mean_signals, predefined_weights)

        self.output_df = self.make_output_df(source_df, seqs_2d, seq_col, self.scored_result, test_df, test_seqs_2d, assign_residue_cols=True)

    def get_rule_tuples(self, residue_charac_dict, motif_length):
        # Helper function that generates a list of rule tuples to be used by generate_matrices()

        self.encoded_chemical_classes = {}
        self.chemical_class_decoder = {}

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

    def encode_seqs_2d(self, seqs_2d):
        '''
        This function finds dim1 coords in stacked conditional matrices matching residues in the given seqs_2d array

        Args:
            seqs_2d (np.ndarray): 2D array where rows are individual peptide sequences as arrays of residues

        Returns:
            stacked_coords_2d (np.ndarray): 2D array of dim1 coordinate references matching stacked matrices lists
        '''

        motif_length = seqs_2d.shape[1]
        stacked_coords_2d = np.full(shape=seqs_2d.shape, fill_value=-1, dtype=int)

        for aa, encoded_chemical_class in self.encoded_chemical_classes.items():
            displacement = encoded_chemical_class * motif_length
            for i in np.arange(motif_length):
                stacked_coordinate = displacement + i
                col_seqs = seqs_2d[:,i]
                stacked_coords_2d[col_seqs == aa, i] = stacked_coordinate

        return stacked_coords_2d

    def generate_baseline_matrix(self, motif_length, source_df, data_params, matrix_params,
                                 amino_acids = amino_acids_phos):
        # Simple function to generate a non-conditional baseline matrix for statistical comparison

        current_matrix_params = matrix_params.copy()
        current_matrix_params["min_members"] = np.inf # forces non-conditional matrix to be generated
        current_matrix_params["included_residues"] = amino_acids
        current_matrix_params["position_for_filtering"] = 0

        # Generate the matrix object and dict key name
        self.baseline_matrix = ConditionalMatrix(motif_length, source_df, data_params, current_matrix_params)

    def generate_matrices(self, rule_tuples, motif_length, source_df, data_params, matrix_params, baseline_matrix):
        '''
        Parallelized application of self.generate_matrix()

        Args:
            rule_tuples (list):       list of tuples of (member amino acid list, filter position, chemical characteristic)
            motif_length (int):       length of the motif being studied
            source_df (pd.DataFrame): dataframe of source data for building matrices
            data_params (dict):       user-defined params for accessing source data, as described in ConditionalMatrix()
            matrix_params (dict):     shared dict of matrix params defined by the user as described in ConditionalMatrix()
            baseline_matrix (ConditionalMatrix): baseline matrix

        Returns:
            results_list (list):  list of results tuples of (conditional_matrix, dict_key_name, report_line)
        '''

        cpus_to_use = os.cpu_count() - 1
        pool = multiprocessing.Pool(processes=cpus_to_use)

        process_partial = partial(self.generate_matrix, motif_length = motif_length, source_df = source_df,
                                  data_params = data_params, matrix_params = matrix_params,
                                  baseline_matrix = baseline_matrix)

        results_list = []

        desc = "Generating conditional matrices (positive, suboptimal, & forbidden element types)..."
        with trange(len(rule_tuples), desc=desc) as pbar:
            for results in pool.imap(process_partial, rule_tuples):
                results_list.append(results)
                pbar.update()

        pool.close()
        pool.join()

        return results_list

    def generate_matrix(self, rule_tuple, motif_length, source_df, data_params, matrix_params, baseline_matrix):
        '''
        Function for generating an individual conditional matrix according to a filter position and member aa list

        Args:
            rule_tuple (tuple):                  tuple of (AA member list, filter position, chemical characteristic)
                                                 representing the current type/position rule, e.g. #1=Acidic
            motif_length (int):                  length of the motif being studied
            source_df (pd.DataFrame):            dataframe of source data for building matrices
            data_params (dict):                  user-defined params for accessing source data
            matrix_params (dict):                user-defined params for generating matrices
            baseline_matrix (ConditionalMatrix): baseline matrix

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
        conditional_matrix = ConditionalMatrix(motif_length, source_df, data_params, current_matrix_params,
                                               baseline_matrix = baseline_matrix)
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

        self.unweighted_matrices_dicts = {"positive": {}, "suboptimal": {}, "forbidden": {}}
        self.unweighted_arrays_dicts = {"positive": {}, "suboptimal": {}, "forbidden": {}}

        for key, conditional_matrix in conditional_matrix_dict.items():
            self.unweighted_matrices_dicts["positive"][key] = conditional_matrix.positive_matrix
            self.unweighted_arrays_dicts["positive"][key] = conditional_matrix.positive_matrix.to_numpy()
            self.unweighted_matrices_dicts["suboptimal"][key] = conditional_matrix.suboptimal_elements_matrix
            self.unweighted_arrays_dicts["suboptimal"][key] = conditional_matrix.suboptimal_elements_matrix.to_numpy()
            self.unweighted_matrices_dicts["forbidden"][key] = conditional_matrix.forbidden_elements_matrix
            self.unweighted_arrays_dicts["forbidden"][key] = conditional_matrix.forbidden_elements_matrix.to_numpy()

    def apply_weights(self, binding_weights, positive_weights, suboptimal_weights, forbidden_weights, only_3d = True):
        # Method for assigning weights to the 3D matrix of matrices

        # Apply weights to 3D representations of matrices for rapid scoring
        self.stacked_binding_weighted = self.stacked_positive_matrices * binding_weights
        self.stacked_positive_weighted = self.stacked_positive_matrices * positive_weights
        self.stacked_suboptimal_weighted = self.stacked_suboptimal_matrices * suboptimal_weights
        self.stacked_forbidden_weighted = self.stacked_forbidden_matrices * forbidden_weights

        # Assign weights to self
        self.binding_positive_weights = binding_weights
        self.accuracy_positive_weights = positive_weights
        self.accuracy_suboptimal_weights = suboptimal_weights
        self.accuracy_forbidden_weights = forbidden_weights

        # Optionally also apply weights to the other formats
        if not only_3d:
            self.weighted_matrices_dicts = {}
            self.weighted_arrays_dicts = {}

            # Handle positive matrices (used for both classification and binding strength prediction)
            binding_positive_matrix_dict, binding_positive_array_dict = {}, {}
            accuracy_positive_matrix_dict, accuracy_positive_array_dict = {}, {}
            for key, positive_matrix_df in self.unweighted_matrices_dicts["positive"].items():
                binding_positive_matrix_dict[key] = positive_matrix_df * binding_weights
                binding_positive_array_dict[key] = positive_matrix_df.to_numpy() * binding_weights
                accuracy_positive_matrix_dict[key] = positive_matrix_df * positive_weights
                accuracy_positive_array_dict[key] = positive_matrix_df.to_numpy() * positive_weights

            self.weighted_matrices_dicts["binding_positive"] = binding_positive_matrix_dict
            self.weighted_matrices_dicts["accuracy_positive"] = accuracy_positive_matrix_dict
            self.weighted_arrays_dicts["binding_positive"] = binding_positive_array_dict
            self.weighted_arrays_dicts["accuracy_positive"] = accuracy_positive_array_dict

            # Handle suboptimal and forbidden matrices (used only for classification prediction)
            accuracy_suboptimal_matrix_dict, accuracy_suboptimal_array_dict = {}, {}
            for key, suboptimal_matrix_df in self.unweighted_matrices_dicts["suboptimal"].items():
                accuracy_suboptimal_matrix_dict[key] = suboptimal_matrix_df * suboptimal_weights
                accuracy_suboptimal_array_dict[key] = suboptimal_matrix_df.to_numpy() * suboptimal_weights
            self.weighted_matrices_dicts["accuracy_suboptimal"] = accuracy_suboptimal_matrix_dict
            self.weighted_arrays_dicts["accuracy_suboptimal"] = accuracy_suboptimal_array_dict

            accuracy_forbidden_matrix_dict, accuracy_forbidden_array_dict = {}, {}
            for key, forbidden_matrix_df in self.unweighted_matrices_dicts["forbidden"].items():
                accuracy_forbidden_matrix_dict[key] = forbidden_matrix_df * forbidden_weights
                accuracy_forbidden_array_dict[key] = forbidden_matrix_df.to_numpy() * forbidden_weights
            self.weighted_matrices_dicts["accuracy_forbidden"] = accuracy_forbidden_matrix_dict
            self.weighted_arrays_dicts["accuracy_forbidden"] = accuracy_forbidden_array_dict

    def save(self, output_folder, save_weighted = True):
        # User-called function to save the conditional matrices as CSVs to folders for both unweighted and weighted

        parent_folder = os.path.join(output_folder, "Conditional_Matrices")

        # Define unweighted matrix output paths
        unweighted_folder_paths = {}
        unweighted_parent = os.path.join(parent_folder, "Unweighted")

        unweighted_folder_paths["positive"] = os.path.join(unweighted_parent, "Positive_Matrices")
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
            weighted_folder_paths["binding_positive"] = os.path.join(weighted_parent, "Binding_Positive_Matrices")
            weighted_folder_paths["accuracy_positive"] = os.path.join(weighted_parent, "Weighted_Positive_Matrices")
            weighted_folder_paths["accuracy_suboptimal"] = os.path.join(weighted_parent, "Weighted_Suboptimal_Matrices")
            weighted_folder_paths["accuracy_forbidden"] = os.path.join(weighted_parent, "Weighted_Forbidden_Matrices")
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

    def score_seqs_2d(self, sequences_2d, use_weighted = False):
        '''
        Vectorized function to score amino acid sequences based on the dictionary of context-aware weighted matrices

        Args:
            sequences_2d (np.ndarray):           unravelled peptide sequences to score
            use_weighted (bool):                 whether to use conditional_matrices.stacked_weighted_matrices

        Returns:
            positive_scores_2d (np.ndarray):     positive element scores; shape is (peptide_count, peptide_length)
            suboptimal_scores_2d (np.ndarray):   suboptimal element scores; shape is (peptide_count, peptide_length)
            forbidden_scores_2d (np.ndarray):    forbidden element scores; shape is (peptide_count, peptide_length)
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

        # Find the matrix identifier numbers (1st dim of 3D matrix) for flanking left and right residues of each residue
        encoded_matrix_refs = self.encode_seqs_2d(sequences_2d)
        left_encoded_matrix_refs = np.concatenate((encoded_matrix_refs[:, 0:1], encoded_matrix_refs[:, 0:-1]), axis=1)
        right_encoded_matrix_refs = np.concatenate((encoded_matrix_refs[:, 1:], encoded_matrix_refs[:, -1:]), axis=1)

        # Flatten the encoded matrix refs, which serve as the 1st dimension referring to 3D matrices
        left_encoded_matrix_refs_flattened = left_encoded_matrix_refs.flatten()
        right_encoded_matrix_refs_flattened = right_encoded_matrix_refs.flatten()

        # Flatten the amino acid row indices into a matching array serving as the 2nd dimension
        aa_row_indices_flattened = aa_row_indices_2d.flatten()

        # Tile the column indices into a matching array serving as the 3rd dimension
        column_indices = np.arange(motif_length)
        column_indices_tiled = np.tile(column_indices, len(sequences_2d))

        # Define dimensions for 3D matrix indexing
        shape_2d = sequences_2d.shape
        left_dim1 = left_encoded_matrix_refs_flattened
        right_dim1 = right_encoded_matrix_refs_flattened
        dim2 = aa_row_indices_flattened
        dim3 = column_indices_tiled

        # Assign matrices to use for scoring
        if use_weighted:
            # Calculate predicted binding strength if specified
            stacked_binding_matrices = self.stacked_binding_weighted
            left_binding_2d = stacked_binding_matrices[left_dim1, dim2, dim3].reshape(shape_2d)
            right_binding_2d = stacked_binding_matrices[right_dim1, dim2, dim3].reshape(shape_2d)
            binding_scores_2d = (left_binding_2d + right_binding_2d) / 2

            # Assign other matrices as well
            stacked_positive_matrices = self.stacked_positive_weighted
            stacked_suboptimal_matrices = self.stacked_suboptimal_weighted
            stacked_forbidden_matrices = self.stacked_forbidden_weighted

        else:
            # Assign unweighted matrices
            stacked_positive_matrices = self.stacked_positive_matrices
            stacked_suboptimal_matrices = self.stacked_suboptimal_matrices
            stacked_forbidden_matrices = self.stacked_forbidden_matrices
            binding_scores_2d = None

        # Calculate predicted positive element scores
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

        return (binding_scores_2d, positive_scores_2d, suboptimal_scores_2d, forbidden_scores_2d)

    def optimize_scoring_weights(self, training_seqs_2d, training_actual_truths, training_signal_values = None,
                                 slice_scores_subsets = None, precision_recall_path = None, coefficients_path = None,
                                 test_seqs_2d = None, test_actual_truths = None, test_signal_values = None,
                                 predefined_weights = None):
        '''
        Vectorized function to score amino acid sequences based on the dictionary of context-aware weighted matrices

        Args:
            training_seqs_2d (np.ndarray):              unravelled peptide sequences to score
            training_actual_truths (np.ndarray):        arr of boolean calls for whether peptides bind in experiments
            training_signal_values (np.ndarray):        arr of binding signal vals for peptides against protein bait(s)
            conditional_matrices (ConditionalMatrices): conditional weighted matrices for scoring peptides
            slice_scores_subsets (np.ndarray):          array of stretches of positions to stratify results into;
                                                        e.g. [6,7,2] is stratified into scores for positions
                                                        1-6, 7-13, & 14-15
            precision_recall_path (str):                desired file path for saving the precision/recall graph
            coefficients_path (str):                    path to save score standardization coefficients to for later use
            test_seqs_2d (np.ndarray):                  if a train/test split was performed, include test sequences
            test_actual_truths (np.ndarray):            if a train/test split was performed, include test seq bools
            test_signal_values (np.ndarray):            if a train/test split was performed, include test signal values
            predefined_weights (tuple):                 tuple of (binding_positive_weights, positive_weights,
                                                        suboptimal_weights, forbidden_weights)

        Returns:
            result (ScoredPeptideResult):               signal, suboptimal, and forbidden score values in 1D and 2D
        '''

        scored_training_arrays = self.score_seqs_2d(training_seqs_2d, use_weighted=False)
        positive_scores_2d, suboptimal_scores_2d, forbidden_scores_2d = scored_training_arrays[1:]

        if test_seqs_2d is not None:
            scored_test_arrays = self.score_seqs_2d(test_seqs_2d, use_weighted=False)
            test_positive_2d, test_suboptimal_2d, test_forbidden_2d = scored_test_arrays[1:]
        else:
            test_positive_2d, test_suboptimal_2d, test_forbidden_2d = None, None, None

        ignore_failed_peptides = True
        preview_scatter_plot = True
        result = ScoredPeptideResult(training_seqs_2d, slice_scores_subsets, positive_scores_2d, suboptimal_scores_2d,
                                     forbidden_scores_2d, training_actual_truths, training_signal_values,
                                     precision_recall_path, True, coefficients_path, self.suppress_positive_positions,
                                     self.suppress_suboptimal_positions, self.suppress_forbidden_positions,
                                     ignore_failed_peptides, preview_scatter_plot, test_seqs_2d,
                                     test_positive_2d, test_suboptimal_2d, test_forbidden_2d, test_actual_truths,
                                     test_signal_values, predefined_weights)

        # Assign metrics to self
        self.train_binding_r2 = result.binding_score_r2
        self.train_accuracy = result.standardized_threshold_accuracy
        if result.test_set_exists:
            self.test_binding_r2 = result.test_binding_r2
            self.test_accuracy = result.test_accuracy
        else:
            self.test_binding_r2, self.test_accuracy = None, None

        # Assign weights and other information to self
        self.apply_weights(result.binding_positive_weights, result.positives_weights,
                           result.suboptimals_weights, result.forbiddens_weights, only_3d=False)
        self.best_accuracy_method = result.best_accuracy_method

        # Assign standardization coefficients to self
        self.binding_standardization_coefficients = result.binding_standardization_coefficients
        self.accuracy_standardization_coefficients = result.standardization_coefficients

        # Assign thresholds to self
        self.standardized_weighted_threshold = result.standardized_weighted_threshold
        self.standardized_threshold_accuracy = result.standardized_threshold_accuracy

        # Assign binding exponential function params to self, where y = ae^b(x-c) + d, and params = (a,b,c,d)
        self.binding_exp_params = result.binding_exp_params
        self.standardized_binding_exp_params = result.standardized_binding_exp_params
        self.binding_score_r2 = result.binding_score_r2
        self.standardized_binding_r2 = result.standardized_binding_r2

        return result

    def make_output_df(self, source_df, seqs_2d, seq_col, scored_result, test_df = None, test_seqs_2d = None,
                       assign_residue_cols = True):
        '''
        Function to generate the output dataframe with scoring results

        Args:
            source_df (pd.DataFrame):                 source peptide dataframe
            seqs_2d (np.ndarray):                     2D array of motif sequences where each row represents a peptide
            seq_col (str):                            column containing sequences
            scored_result (ScoredPeptideResult):      training set output from optimize_conditional_weights()
            test_df (pd.DataFrame):                   peptide dataframe containing the test set
            test_seqs_2d (np.ndarray):                2D array of test set motif sequences
            assign_residue_cols (bool):               whether to assign cols for motif positions for better sortability

        Returns:
            output_df (pd.DataFrame):            output dataframe with new scores added
        '''

        output_df = source_df.reset_index()
        if test_df is not None:
            test_output_df = test_df.set_index(np.arange(len(output_df), len(output_df) + len(test_df)))
            output_df["Set"] = "train"
            test_output_df["Set"] = "test"
            output_df = pd.concat([output_df, test_output_df], axis=0, ignore_index=False)
        scored_df = scored_result.scored_df.set_index(output_df.index)
        output_df = pd.concat([output_df, scored_df], axis=1, ignore_index=False)

        current_cols = list(output_df.columns)
        insert_index = current_cols.index(seq_col) + 1

        # Assign residue columns
        if assign_residue_cols:
            residue_cols = ["#" + str(i) for i in np.arange(1, seqs_2d.shape[1] + 1)]
            residues_df = pd.DataFrame(seqs_2d, columns=residue_cols)
            if test_seqs_2d is not None and test_df is not None:
                test_residues_df = pd.DataFrame(test_seqs_2d, columns=residue_cols)
                residues_df = pd.concat([residues_df, test_residues_df], axis=0)
                residues_df = residues_df.set_index(output_df.index)
            output_df = pd.concat([output_df, residues_df], axis=1, ignore_index=False)

            # Define list of columns in the desired order
            final_columns = current_cols[0:insert_index]
            final_columns.extend(residue_cols)
            final_columns.extend(current_cols[insert_index:])

            # Reassign the output df with the ordered columns
            output_df = output_df[final_columns]

        return output_df