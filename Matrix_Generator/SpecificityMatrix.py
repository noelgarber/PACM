# This script describes the SpecificityMatrix object used in make_specificity_matrices.py

import numpy as np
import pandas as pd
import os
from scipy.stats import ttest_ind
from sklearn.metrics import precision_recall_curve
from general_utils.user_helper_functions import get_comparator_baits
from general_utils.matrix_utils import collapse_phospho
from Matrix_Generator.sigmoid_regression import fit_sigmoid
try:
    from Matrix_Generator.config_local import amino_acids, amino_acids_phos, comparator_info, specificity_params
except:
    from Matrix_Generator.config import amino_acids, amino_acids_phos, comparator_info, specificity_params

def optimize_f1(actual_truths, score_values):
    # Helper function that optimizes f1-score using a precision-recall curve

    valid_mask = np.logical_and(np.isfinite(actual_truths), np.isfinite(score_values))
    valid_actual_truths = actual_truths[valid_mask]
    valid_score_values = score_values[valid_mask]

    precision, recall, thresholds = precision_recall_curve(valid_actual_truths, valid_score_values)
    precision_recall_products = precision * recall
    precision_recall_sums = precision + recall
    precision_recall_sums[precision_recall_sums == 0] = np.nan
    f1_scores = 2 * precision_recall_products / precision_recall_sums
    best_idx = np.nanargmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_threshold = thresholds[best_idx]

    return best_f1, best_threshold

class SpecificityMatrix:
    '''
    Specificity matrix class for unweighted and weighted matrices that predict bait selectivity over a peptide motif
    '''

    def __init__(self, source_df, comparator_info = comparator_info, specificity_params = specificity_params):
        '''
        Main function for generating and assessing optimal specificity position-weighted matrices

        Args:
            source_df (pd.DataFrame):  dataframe containing sequences, pass/fail info, and log2fc values
            comparator_info (dict):    dict of info about comparators and data locations as described in config.py
            specificity_params (dict): dict of specificity matrix generation parameters as described in config.py
        '''

        self.scored_source_df = source_df.copy()

        # Calculate least different pair of baits and corresponding log2fc for each row in output_df
        self.comparator_set_1 = comparator_info.get("comparator_set_1")
        self.comparator_set_2 = comparator_info.get("comparator_set_2")
        if self.comparator_set_1 is None or self.comparator_set_2 is None:
            self.comparator_set_1, self.comparator_set_2 = get_comparator_baits()
        self.find_least_different()

        # Get the multiplier to adjust for asymmetric distribution of bait specificities in the data
        self.thresholds = specificity_params["thresholds"]
        self.matching_points = specificity_params["matching_points"]
        self.matching_points = specificity_params["matching_points"]
        self.extreme_thresholds = (self.thresholds[0], self.thresholds[3])
        passes_col, pass_str = comparator_info.get("bait_pass_col"), comparator_info.get("pass_str")
        self.set_bias_ratio(self.extreme_thresholds, passes_col, pass_str)

        # Define required arguments for making the specificity matrix
        sequence_col = comparator_info.get("seq_col")
        self.source_sequences = source_df[sequence_col].to_numpy()
        self.significance_array = self.scored_source_df[passes_col].to_numpy() == pass_str
        max_bait_mean_col = specificity_params["max_bait_mean_col"]
        self.max_signal_vals = source_df[max_bait_mean_col].to_numpy()
        self.include_phospho = specificity_params.get("include_phospho")

        # Generate the unweighted specificity matrix, calculate unweighted scores, and generate statistics
        self.make_specificity_matrix()
        self.score_source_peptides(use_weighted = False)
        self.plus_threshold, self.minus_threshold = specificity_params["plus_threshold"], specificity_params["minus_threshold"]
        self.set_specificity_statistics(use_weighted = False)

        # If predefined weights exist, apply them, otherwise leave self.weighted_matrix_df undefined
        predefined_weights = specificity_params.get("predefined_weights")
        if predefined_weights:
            self.apply_weights(predefined_weights)
            self.score_source_peptides(use_weighted = True)
            self.set_specificity_statistics(use_weighted = True)

    def set_bias_ratio(self, thresholds, passes_col, pass_str):
        '''
        Function for finding the ratio of entries in the dataframe specific to one bait set vs. the other bait set;
        necessary for statistical adjustment when the data is not evenly distributed between baits

        Args:
            thresholds (tuple):                  tuple of floats as (positive_thres, negative_thres)
            passes_col (str):                    col name in source_df containing pass/fail info (significance calls)
            pass_str (str):                      the string that indicates a pass in source_df[pass_col], e.g. "Yes"

        Returns:
            ratio (float): the ratio of entries above pos_thres to entries below neg_thres
        '''

        positive_thres, negative_thres = thresholds

        # Get boolean series for thresholds and pass/fail info
        above_thres = (self.least_different_values > positive_thres)
        below_neg_thres = (self.least_different_values < negative_thres)
        passes = self.scored_source_df[passes_col] == pass_str

        # Count the number of qualifying entries that are above/below the relevant threshold and are marked as pass
        above_thres_count = (above_thres & passes).sum()
        below_neg_thres_count = (below_neg_thres & passes).sum()

        # Handle divide-by-zero instances by incrementing both values by 1
        if below_neg_thres_count == 0:
            below_neg_thres_count += 1
            above_thres_count += 1

        self.bias_ratio = above_thres_count / below_neg_thres_count

    def find_least_different(self, log2fc_cols = None):
        '''
        Simple function to determine the least different log2fc between permutations of the sets of comparators

        Args:
            log2fc_cols (list):        list of columns; if not given, it will be auto-generated

        Returns:
            None; operates on self.scored_source_df and reassigns it
        '''

        # Define the columns containing log2fc values
        if log2fc_cols is None:
            log2fc_cols = []
            for bait1 in self.comparator_set_1:
                for bait2 in self.comparator_set_2:
                    if bait1 != bait2:
                        log2fc_cols.append(bait1 + "_" + bait2 + "_log2fc")

        # Get least different values and bait pairs
        log2fc_values = self.scored_source_df[log2fc_cols].to_numpy()
        magnitudes = np.abs(log2fc_values)
        all_nan = np.all(np.isnan(magnitudes), axis=1)
        least_different_indices = np.zeros(shape=len(magnitudes), dtype=int)
        least_different_indices[~all_nan] = np.nanargmin(magnitudes[~all_nan], axis=1)
        self.least_different_values = log2fc_values[np.arange(len(log2fc_values)), least_different_indices]
        least_different_cols = np.array(log2fc_cols)[least_different_indices]
        self.least_different_baits = [col.rsplit("_", 1)[0] for col in least_different_cols]

        # Assign columns if specified
        self.scored_source_df["least_different_log2fc"] = self.least_different_values
        self.scored_source_df["least_different_baits"] = self.least_different_baits

    def reorder_matrix(self, matrix_df):
        # Helper function to add missing amino acid rows if necessary and reorder matrix_df by aa_list

        collapse_phospho(matrix_df) if not self.include_phospho else None
        aa_list = list(amino_acids) if not self.include_phospho else list(amino_acids_phos)

        missing_row_indices = [aa for aa in aa_list if aa not in matrix_df.index]
        if len(missing_row_indices) > 0:
            new_rows = pd.DataFrame(0, index=missing_row_indices, columns=matrix_df.columns)
            matrix_df = pd.concat([matrix_df, new_rows])

        # Reorder the matrix rows according to aa_list as the indices
        matrix_df = matrix_df.loc[aa_list]

        return matrix_df

    def make_specificity_matrix(self):
        '''
        Function for generating a position-weighted matrix by assigning points based on seqs and their log2fc values
        '''

        # Extract the significant sequences and log2fc values as numpy arrays
        self.passing_indices = np.where(self.significance_array)[0]
        self.passing_seqs = self.source_sequences[self.passing_indices]
        self.passing_log2fc_values = self.least_different_values[self.passing_indices]

        # Get points scaling values using equation: points = 1 / (1+e**(-k(x-x0)))
        max_signals = self.max_signal_vals.copy()
        max_signals[max_signals < 0] = 0
        passing_max_signals = max_signals[self.passing_indices]
        passing_max_signals = passing_max_signals - passing_max_signals.min()
        passing_max_signals = passing_max_signals / passing_max_signals.max()

        x0 = np.median(passing_max_signals) # sigmoid inflection is at median magnitude
        k = 5
        passing_scaling_values = 1 / (1 + np.exp(-k * (passing_max_signals - x0)))

        # Filter log2fc values to not include small changes
        filtered_log2fc_values = self.passing_log2fc_values.copy()
        filtered_log2fc_values[np.abs(filtered_log2fc_values) < 0.5] = 0

        # Calculate final points to assign based on scaling values and proportions
        passing_points_values = filtered_log2fc_values * passing_scaling_values

        # Adjust for more hits being specific to one bait vs. the other
        minus_points_sum = np.mean(passing_points_values[passing_points_values < 0])
        plus_points_sum = np.mean(passing_points_values[passing_points_values > 0])
        bias_ratio = np.abs(plus_points_sum / minus_points_sum)
        adjusted_points_values = passing_points_values.copy()
        adjusted_points_values[adjusted_points_values < 0] *= bias_ratio

        # Check that all the sequences are the same length
        positive_seq_lengths = np.char.str_len(self.passing_seqs.astype(str))
        same_length = np.all(positive_seq_lengths == positive_seq_lengths[0])
        if not same_length:
            raise ValueError(f"source sequences have varying length, but must all be one length")
        self.motif_length = positive_seq_lengths[0]

        # Convert sequences to array of arrays
        self.passing_seqs = self.passing_seqs.astype("<U")
        positive_seqs_unravelled = self.passing_seqs.view("U1")
        self.passing_seqs_2d = np.reshape(positive_seqs_unravelled, (-1, self.motif_length))

        # Build the matrix
        unique_residues = np.unique(self.passing_seqs_2d)
        cols = np.char.add("#", np.arange(1, self.motif_length + 1).astype(str))
        matrix_df = pd.DataFrame(index=unique_residues, columns=cols, dtype=float).fillna(0.0)

        for col_name, col_slice in zip(cols, np.transpose(self.passing_seqs_2d)):
            col_unique_residues = np.unique(col_slice)

            for aa in col_unique_residues:
                qualifying_points = adjusted_points_values[col_slice == aa]
                qualifying_points_sum = np.sum(qualifying_points[np.isfinite(qualifying_points)])

                if not np.all(~np.isfinite(qualifying_points)):
                    qualifying_log2fc = self.passing_log2fc_values[col_slice == aa]
                    qualifying_log2fc = qualifying_log2fc[np.isfinite(qualifying_log2fc)]
                    other_log2fc = self.passing_log2fc_values[col_slice != aa]
                    other_log2fc = other_log2fc[np.isfinite(other_log2fc)]
                    result = ttest_ind(qualifying_log2fc, other_log2fc, equal_var=False, nan_policy="omit")
                    p_value = result.pvalue

                    if p_value <= 0.5:
                        matrix_df.at[aa, col_name] = qualifying_points_sum

        # Add missing amino acid rows if necessary and reorder matrix_df by aa_list
        matrix_df = self.reorder_matrix(matrix_df)

        # Standardize matrix by max column values
        max_values = matrix_df.max(axis=0)
        max_values[max_values == 0] = 1 # avoid divide-by-zero
        matrix_df = matrix_df / max_values

        self.matrix_df = matrix_df

    def score_source_peptides(self, use_weighted = True):
        # Function to back-calculate specificity scores on peptide sequences based on the generated specificity matrix

        scoring_matrix = self.weighted_matrix_df if use_weighted else self.matrix_df

        # Get the indices for the matrix for each amino acid at each position
        sequence_count = len(self.passing_log2fc_values)
        indexer = scoring_matrix.index.get_indexer
        row_indices = indexer(self.passing_seqs_2d.ravel()).reshape(self.passing_seqs_2d.shape)
        column_indices = np.arange(self.motif_length)[np.newaxis, :].repeat(sequence_count, axis=0)

        # Calculate the points
        all_score_values = np.full(shape=len(self.source_sequences), fill_value=np.nan, dtype=float)
        if use_weighted:
            self.passing_weighted_points = scoring_matrix.values[row_indices, column_indices]
            self.passing_weighted_scores = self.passing_weighted_points.sum(axis=1)
            all_score_values[self.passing_indices] = self.passing_weighted_scores
            self.scored_source_df["Weighted_Specificity_Score"] = all_score_values
        else:
            self.passing_unweighted_points = scoring_matrix.values[row_indices, column_indices]
            self.passing_unweighted_scores = self.passing_unweighted_points.sum(axis=1)
            all_score_values[self.passing_indices] = self.passing_unweighted_scores
            self.scored_source_df["Unweighted_Specificity_Score"] = all_score_values

    def set_specificity_statistics(self, use_weighted = True):
        # Function to evaluate specificity score correlation with actual log2fc values that are observed

        # Get the valid score values and log2fc values
        valid_indices = np.where(np.isfinite(self.passing_log2fc_values))
        if use_weighted:
            valid_score_values = self.passing_weighted_scores[valid_indices]
        else:
            valid_score_values = self.passing_unweighted_scores[valid_indices]
        valid_log2fc_values = self.passing_log2fc_values[valid_indices]

        # Find upper and lower f1-scores
        log2fc_above_upper = np.greater_equal(valid_log2fc_values, self.plus_threshold)
        best_f1_upper, best_threshold_upper = optimize_f1(log2fc_above_upper, valid_score_values)

        log2fc_below_lower = np.less_equal(valid_log2fc_values, self.minus_threshold)
        best_f1_lower, best_threshold_lower = optimize_f1(log2fc_below_lower, valid_score_values * -1)
        best_threshold_lower = best_threshold_lower * -1 # corrected the inverted sign

        upper_lower_ratio = log2fc_above_upper.sum() / log2fc_below_lower.sum()
        weighted_mean_f1 = (best_f1_upper + upper_lower_ratio * best_f1_lower) / (1 + upper_lower_ratio)

        if use_weighted:
            self.weighted_mean_f1 = weighted_mean_f1
            self.weighted_upper_f1, self.weighted_lower_f1 = best_f1_upper, best_f1_lower
            self.weighted_upper_threshold = best_threshold_upper
            self.weighted_lower_threshold = best_threshold_lower
        else:
            self.unweighted_mean_f1 = weighted_mean_f1
            self.unweighted_upper_f1, self.unweighted_lower_f1 = best_f1_upper, best_f1_lower
            self.unweighted_upper_threshold = best_threshold_upper
            self.unweighted_lower_threshold = best_threshold_lower

    def apply_weights(self, weights_array):
        # User-initiated function for applying matrix weights

        # Apply the weights and reconstruct the matrix; this is about 5x faster than direct pandas multiplication
        self.weighted_matrix_df = pd.DataFrame(self.matrix_df.values * weights_array, index=self.matrix_df.index,
                                               columns=self.matrix_df.columns)
        self.position_weights = weights_array

    def save(self, output_folder, verbose = True):
        # User-initiated function for saving the matrices, scored input data, and statistics to a defined output folder

        specificity_scored_path = os.path.join(output_folder, "specificity_scored_data.csv")
        self.scored_source_df.to_csv(specificity_scored_path)
        print(f"Saved specificity-scored source data to {specificity_scored_path}") if verbose else None

        unweighted_matrix_path = os.path.join(output_folder, "unweighted_specificity_matrix.csv")
        self.matrix_df.to_csv(unweighted_matrix_path)
        print(f"Saved unweighted specificity matrix to {unweighted_matrix_path}") if verbose else None

        try:
            weighted_matrix_path = os.path.join(output_folder, "weighted_specificity_matrix.csv")
            self.weighted_matrix_df.to_csv(weighted_matrix_path)
            print(f"Saved weighted specificity matrix to {weighted_matrix_path}")
        except NameError:
            pass

        statistics_path = os.path.join(output_folder, "specificity_statistics.txt")
        output_lines = ["Specificity Matrix Output Statistics\n\n",
                        "---\n",
                        f"Motif length: {self.motif_length}\n\n",
                        f"---\n",
                        f"Matthews correlation coefficients for upper and lower specificity score thresholds\n\n",
                        f"Unweighted matrix: \n",
                        f"\tMerged f1-score: {self.unweighted_mean_f1}\n",
                        f"\tUpper f1-score: {self.unweighted_upper_f1} for scores ≥ {self.unweighted_upper_threshold}\n",
                        f"\tLower f1-score: {self.unweighted_lower_f1} for scores ≤ {self.unweighted_lower_threshold}"]
        try:
            add_lines = ["\n\n",
                         f"Weighted matrix: \n",
                         f"\tSpecificity matrix weights: {self.position_weights}\n",
                         f"\tMerged f1-score: {self.weighted_mean_f1}\n",
                         f"\tUpper f1-score: {self.weighted_upper_f1} for scores ≥ {self.weighted_upper_threshold}\n",
                         f"\tLower f1-score: {self.weighted_lower_f1} for scores ≤ {self.weighted_lower_threshold}"]
            output_lines.extend(add_lines)
        except NameError:
            pass

        with open(statistics_path, "w") as file:
            file.writelines(output_lines)
        print(f"Saved specificity matrices regression statistics to {statistics_path}")

    def plot_regression(self, output_folder, use_weighted = True):
        '''
        Performs logistic regression on scores vs. log2fc values of source data and then graphs the results

        Args:
            output_folder (str): the folder to save the graphs into
            use_weighted (bool): whether to save a graph for weighted scores if they were calculated

        Returns:
            None
        '''

        x_values = self.scored_source_df["least_different_log2fc"].to_numpy()
        y_values = self.scored_source_df["Unweighted_Specificity_Score"].to_numpy()
        x_label = "Least Different log2fc"
        y_label = "Unweighted Specificity Score"
        title = "Correlation of Unweighted Specificity Score and Actual Signal Differences"
        save_as = os.path.join(output_folder, "unweighted_correlation_graph.pdf")
        fit_sigmoid(x_values, y_values, save_as, x_label, y_label, title)

        if use_weighted:
            y_values = self.scored_source_df["Weighted_Specificity_Score"].to_numpy()
            y_label = "Weighted Specificity Score"
            title = "Correlation of Weighted Specificity Score and Actual Signal Differences"
            save_as = os.path.join(output_folder, "weighted_correlation_graph.pdf")
            fit_sigmoid(x_values, y_values, save_as, x_label, y_label, title)

    def score_motifs(self, motif_seqs_2d, use_weighted = True):
        '''
        User function to calculate specificity scores on motif sequences based on the generated specificity matrix

        Args:
            motif_seqs_2d (np.ndarray): 2D array where each row is a motif as an array of residue letter codes
            use_weighted (bool):        whether to use the weighted specificity matrix when scoring

        Returns:
            score_values (np.ndarray):  score values for input motifs
        '''

        if motif_seqs_2d.shape[1] != self.motif_length:
            raise Exception(f"score_motifs error: given motifs had the shape {motif_seqs_2d.shape}, " +
                            f"which does not match the specificity matrix motif length ({self.motif_length})")

        scoring_matrix = self.weighted_matrix_df if use_weighted else self.matrix_df

        # Get the indices for the matrix for each amino acid at each position
        sequence_count = len(motif_seqs_2d)
        indexer = scoring_matrix.index.get_indexer
        row_indices = indexer(motif_seqs_2d.ravel()).reshape(motif_seqs_2d.shape)
        column_indices = np.arange(self.motif_length)[np.newaxis, :].repeat(sequence_count, axis=0)

        # Calculate the points
        points_2d = scoring_matrix.values[row_indices, column_indices]
        scores_values = points_2d.sum(axis=1)

        return scores_values