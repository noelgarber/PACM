# This script describes the SpecificityMatrix object used in make_specificity_matrices.py

import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from Matrix_Generator.config import amino_acids, amino_acids_phos, comparator_info, specificity_params
from general_utils.user_helper_functions import get_comparator_baits
from general_utils.matrix_utils import collapse_phospho

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
        self.extreme_thresholds = (self.thresholds[0], self.thresholds[3])
        passes_col, pass_str = comparator_info.get("bait_pass_col"), comparator_info.get("pass_str")
        self.set_bias_ratio(self.extreme_thresholds, passes_col, pass_str)

        # Define required arguments for making the specificity matrix
        sequence_col = comparator_info.get("seq_col")
        self.source_sequences = source_df[sequence_col].to_numpy()
        self.significance_array = self.scored_source_df[passes_col].to_numpy() == pass_str
        self.include_phospho = specificity_params.get("include_phospho")

        # Generate the unweighted specificity matrix, calculate unweighted scores, and generate statistics
        self.make_specificity_matrix()
        self.score_source_peptides(use_weighted = False)
        abs_extrema_threshold = specificity_params.get("abs_extrema_threshold")
        self.set_specificity_statistics(abs_extrema_threshold, use_weighted = False)

        # If predefined weights exist, apply them, otherwise leave self.weighted_matrix_df undefined
        predefined_weights = specificity_params.get("predefined_weights")
        if predefined_weights:
            self.apply_weights(predefined_weights)
            self.score_source_peptides(use_weighted = True)
            self.set_specificity_statistics(abs_extrema_threshold, use_weighted = True)

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
        log2fc_array = self.scored_source_df[log2fc_cols].to_numpy()
        log2fc_array[np.isnan(log2fc_array)] = np.inf
        least_different_indices = np.nanargmin(log2fc_array, axis=1)
        self.least_different_values = np.min(log2fc_array, axis=1)
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
        # Function for generating a position-weighted matrix by assigning points based on seqs and their log2fc values

        # Check if thresholds are sorted
        if not np.all(np.array(self.thresholds)[:-1] >= np.array(self.thresholds)[1:]) or len(self.thresholds) != 4:
            raise ValueError(f"thresholds were set to {self.thresholds}, but must be in descending order as a tuple of",
                             f"(upper_positive, middle_positive, middle_negative, upper_negative)")

        # Extract the significant sequences and log2fc values as numpy arrays
        self.positive_indices = np.where(self.significance_array)[0]
        self.positive_seqs = self.source_sequences[self.positive_indices]
        self.positive_log2fc_values = self.least_different_values[self.positive_indices]

        # Check that all the sequences are the same length
        positive_seq_lengths = np.char.str_len(self.positive_seqs.astype(str))
        same_length = np.all(positive_seq_lengths == positive_seq_lengths[0])
        if not same_length:
            raise ValueError(f"source sequences have varying length, but must all be one length")
        self.motif_length = positive_seq_lengths[0]

        # Find where log2fc values pass each threshold
        passes_upper_positive = np.where(self.positive_log2fc_values >= self.thresholds[0])
        passes_middle_positive = np.where(np.logical_and(self.positive_log2fc_values >= self.thresholds[1],
                                                         self.positive_log2fc_values < self.thresholds[0]))
        passes_middle_negative = np.where(np.logical_and(self.positive_log2fc_values <= self.thresholds[2],
                                                         self.positive_log2fc_values > self.thresholds[3]))
        passes_upper_negative = np.where(self.positive_log2fc_values <= self.thresholds[3])

        # Get an array of points values matching the sequences
        self.positive_points_values = np.zeros_like(self.positive_log2fc_values)
        self.positive_points_values[passes_upper_positive] = self.matching_points[0]
        self.positive_points_values[passes_middle_positive] = self.matching_points[1]
        self.positive_points_values[passes_middle_negative] = self.matching_points[2] * self.bias_ratio
        self.positive_points_values[passes_upper_negative] = self.matching_points[3] * self.bias_ratio

        # Convert sequences to array of arrays, and do the same for matching points
        self.positive_seqs = self.positive_seqs.astype("<U")
        positive_seqs_unravelled = self.positive_seqs.view("U1")
        self.positive_seqs_2d = np.reshape(positive_seqs_unravelled, (-1, self.motif_length))
        self.positive_points_2d = np.repeat(self.positive_points_values[:, np.newaxis], len(self.positive_seqs), axis=1)

        # Make a new matrix and apply points to it
        matrix_indices = np.unique(self.positive_seqs_2d)
        column_names = np.char.add("#", np.arange(1, self.motif_length + 1).astype(str))
        matrix_df = pd.DataFrame(index=matrix_indices, columns=column_names).fillna(0)

        for column_name, residues_column, points_column in zip(column_names, np.transpose(self.positive_seqs_2d),
                                                               np.transpose(self.positive_points_2d)):
            unique_residues, counts = np.unique(residues_column, return_counts=True)
            indices = np.searchsorted(unique_residues, residues_column)
            sums = np.bincount(indices, weights=points_column)
            matrix_df.loc[unique_residues, column_name] = sums

        # Add missing amino acid rows if necessary and reorder matrix_df by aa_list
        matrix_df = self.reorder_matrix(matrix_df)

        # Standardize matrix by max column values
        max_values = matrix_df.max(axis=0)
        matrix_df = matrix_df / max_values

        self.matrix_df = matrix_df

    def score_source_peptides(self, use_weighted = True):
        # Function to back-calculate specificity scores on peptide sequences based on the generated specificity matrix

        scoring_matrix = self.weighted_matrix_df if use_weighted else self.matrix_df

        # Get the indices for the matrix for each amino acid at each position
        sequence_count = len(self.positive_points_2d)
        indexer = scoring_matrix.index.get_indexer
        row_indices = indexer(self.positive_seqs_2d.ravel()).reshape(self.positive_seqs_2d.shape)
        column_indices = np.arange(self.motif_length)[np.newaxis, :].repeat(sequence_count, axis=0)

        # Calculate the points
        all_score_values = np.full(shape=len(self.source_sequences), fill_value=np.nan, dtype=float)
        if use_weighted:
            self.positive_weighted_points = scoring_matrix.values[row_indices, column_indices]
            self.positive_weighted_scores = self.positive_weighted_points.sum(axis=1)
            all_score_values[self.positive_indices] = self.positive_weighted_scores
            self.scored_source_df["Weighted_Specificity_Score"] = all_score_values
        else:
            self.positive_unweighted_points = scoring_matrix.values[row_indices, column_indices]
            self.positive_unweighted_scores = self.positive_unweighted_points.sum(axis=1)
            all_score_values[self.positive_indices] = self.positive_unweighted_scores
            self.scored_source_df["Unweighted_Specificity_Score"] = all_score_values

    def set_specificity_statistics(self, abs_extrema_threshold = 0.5, use_weighted = True):
        # Function to evaluate specificity score correlation with actual log2fc values that are observed

        # Perform linear regression between log2fc values and scores
        model = LinearRegression()
        valid_indices = np.where(np.isfinite(self.positive_log2fc_values))
        if use_weighted:
            x_actual = self.positive_weighted_scores[valid_indices].reshape(-1, 1)
        else:
            x_actual = self.positive_unweighted_scores[valid_indices].reshape(-1, 1)
        y_actual = self.positive_log2fc_values[valid_indices].reshape(-1, 1)

        model.fit(x_actual, y_actual)
        linear_coef = model.coef_[0][0]
        linear_intercept = model.intercept_[0]
        y_pred = model.predict(x_actual)
        linear_r2 = r2_score(y_actual, y_pred)
        linear_equation = f"y={linear_coef}x{linear_intercept:+.2f}"

        if use_weighted:
            self.weighted_linear_coef, self.weighted_linear_intercept = linear_coef, linear_intercept
            self.weighted_linear_r2, self.weighted_linear_equation = linear_r2, linear_equation
        else:
            self.unweighted_linear_coef, self.unweighted_linear_intercept = linear_coef, linear_intercept
            self.unweighted_linear_r2, self.unweighted_linear_equation = linear_r2, linear_equation

        # Perform linear regression also on only the extrema to get R2 without influence from middle values
        extrema_model = LinearRegression()
        extrema_bools = np.abs(y_actual) > abs_extrema_threshold
        x_actual_extrema = x_actual[extrema_bools].reshape(-1, 1)
        y_actual_extrema = y_actual[extrema_bools].reshape(-1, 1)

        extrema_model.fit(x_actual_extrema, y_actual_extrema)
        extrema_linear_coef = extrema_model.coef_[0][0]
        extrema_linear_intercept = extrema_model.intercept_[0]
        y_pred_extrema = extrema_model.predict(x_actual_extrema)
        extrema_linear_r2 = r2_score(y_actual_extrema, y_pred_extrema)
        extrema_linear_equation = f"y={extrema_linear_coef}x{extrema_linear_intercept:+.2f}"

        if use_weighted:
            self.weighted_extrema_coef, self.weighted_extrema_intercept = extrema_linear_coef, extrema_linear_intercept
            self.weighted_extrema_r2, self.weighted_extrema_equation = extrema_linear_r2, extrema_linear_equation
        else:
            self.unweighted_extrema_coef, self.unweighted_extrema_intercept = extrema_linear_coef, extrema_linear_intercept
            self.unweighted_extrema_r2, self.unweighted_extrema_equation = extrema_linear_r2, extrema_linear_equation

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
                        "---\n\n",
                        f"Motif length: {self.motif_length}\n\n",
                        f"Unweighted matrix linear regressions where y=log2fc and x=specificity_score:\n",
                        f"\tLinear equation fit to all data: {self.unweighted_linear_equation}, R²={self.unweighted_linear_r2}\n",
                        f"\tLinear equation fit to only the extrema: {self.unweighted_extrema_equation}, R²={self.unweighted_extrema_r2}\n\n"]
        try:
            weighted_lines = [f"Specificity matrix weights: {self.position_weights}\n",
                              f"Weighted matrix linear regressions as above:\n",
                              f"\tLinear equation fit to all data: {self.weighted_linear_equation}, R²={self.weighted_linear_r2}\n",
                              f"\tLinear equation fit to only the extrema: {self.weighted_extrema_equation}, R²={self.weighted_extrema_equation}\n"]
            output_lines.extend(weighted_lines)
        except NameError:
            pass

        with open(statistics_path, "w") as file:
            file.writelines(output_lines)
        print(f"Saved specificity matrices regression statistics to {statistics_path}")