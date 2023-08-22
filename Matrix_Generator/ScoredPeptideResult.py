# Defines the ScoredPeptideResult class, which represents peptide scoring results from ConditionalMatrices objects

import numpy as np
import pandas as pd
import os
from functools import partial
from scipy.optimize import minimize
try:
    from Matrix_Generator.config_local import data_params, matrix_params, aa_equivalence_dict
except:
    from Matrix_Generator.config import data_params, matrix_params, aa_equivalence_dict

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
        self.slice_scores()
        self.stack_scores()

        self.optimized = False

    def slice_scores(self):
        # Function that generates scores based on sliced subsets of peptide sequences

        self.score_cols = ["Adjusted_Predicted_Signal", "Suboptimal_Element_Score", "Forbidden_Element_Score"]

        if self.slice_scores_subsets is not None:
            end_position = 0
            self.sliced_predicted_signals = []
            self.sliced_suboptimal_scores = []
            self.sliced_forbidden_scores = []

            for subset in slice_scores_subsets:
                start_position = end_position
                end_position += subset
                suffix_str = str(start_position) + "-" + str(end_position)

                subset_predicted_signals = self.predicted_signals_2d[:, start_position:end_position + 1].sum(axis=1)
                self.sliced_predicted_signals.append(subset_predicted_signals)
                self.score_cols.append("Predicted_Signal_" + suffix_str)

                subset_suboptimal_scores = self.suboptimal_scores_2d[:, start_position:end_position + 1].sum(axis=1)
                self.sliced_suboptimal_scores.append(subset_suboptimal_scores)
                self.score_cols.append("Suboptimal_Score_" + suffix_str)

                subset_forbidden_scores = self.forbidden_scores_2d[:, start_position:end_position + 1].sum(axis=1)
                self.sliced_forbidden_scores.append(subset_forbidden_scores)
                self.score_cols.append("Forbidden_Score_" + suffix_str)

        else:
            self.sliced_predicted_signals = None
            self.sliced_suboptimal_scores = None
            self.sliced_forbidden_scores = None

    def stack_scores(self):
        '''
        Helper function that constructs a 2D array of scores values as columns
        '''

        scores = [self.adjusted_predicted_signals, self.suboptimal_scores * -1, self.forbidden_scores * -1]
        scores_original = [self.adjusted_predicted_signals, self.suboptimal_scores, self.forbidden_scores]
        sign_mutlipliers = [1, -1, -1]

        if self.slice_scores_subsets is not None:
            for predicted_signals_slice in self.sliced_predicted_signals:
                scores.append(predicted_signals_slice)
                scores_original.append(predicted_signals_slice)
                sign_mutlipliers.append(1)
            for suboptimal_scores_slice in self.sliced_suboptimal_scores:
                scores.append(suboptimal_scores_slice * -1)
                scores_original.append(suboptimal_scores_slice)
                sign_mutlipliers.append(-1)
            for forbidden_scores_slice in self.sliced_forbidden_scores:
                scores.append(forbidden_scores_slice * -1)
                scores_original.append(forbidden_scores_slice)
                sign_mutlipliers.append(-1)

        self.sign_mutlipliers = np.array(sign_mutlipliers)
        self.stacked_scores = np.stack(scores).T
        stacked_scores_original = np.stack(scores_orignal).T
        self.scored_df = pd.DataFrame(stacked_scores_original, columns = self.score_cols)

    def optimize_thresholds(self, passes_bools):
        '''
        Optimization function to determine the optimal thresholds for the scores

        Args:
            passes_bools (np.ndarray):  array of actual truth values

        Returns:
            None; assigns results to self
        '''

        # Determine the optimal thresholds using Nelder-Mead optimization
        initial_thresholds = np.median(self.stacked_scores, axis=0)
        optimization_function = partial(negative_accuracy, scores_arrays = self.stacked_scores,
                                        passes_bools = passes_bools)
        optimization_result = minimize(optimization_function, initial_thresholds, method = "Nelder-Mead")
        self.optimized_thresholds_signed = optimization_result.x
        self.optimized_thresholds = self.optimized_thresholds_signed * self.sign_mutlipliers
        self.optimized_accuracy = optimization_result.fun * -1

        if optimization_result.status != 0:
            raise Exception(optimization_result.message)

        # Use optimized thresholds to make boolean predictions and calculate MCC
        boolean_predictions_2d = self.stacked_scores > self.optimized_thresholds_signed
        self.boolean_predictions = np.all(boolean_predictions_2d, axis=1)
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
