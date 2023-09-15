# Defines the ScoredPeptideResult class, which represents peptide scoring results from ConditionalMatrices objects

import numpy as np
import pandas as pd
import warnings
from functools import partial
from sklearn.metrics import precision_recall_curve, matthews_corrcoef
from scipy.optimize import minimize
try:
    from Matrix_Generator.config_local import aa_charac_dict
except:
    from Matrix_Generator.config import aa_charac_dict

''' ----------------------------------------------------------------------------------------------------------------
                      Define optimization functions for determining position and score weights
    ---------------------------------------------------------------------------------------------------------------- '''

def objective_function(concatenated_weights, actual_truths, positives_2d, suboptimals_2d, forbiddens_2d):
    '''
    Objective function to minimize, based on f1-scoring and application of variable weights

    Args:
        concatenated_weights (np.ndarray): array of weights of shape (positions_count+3,), where there are 3 extra
                                           weights for total positive, total suboptimal, & total forbidden scores
        actual_truths (np.ndarray):        array of actual truth values as binary integer-encoded labels
        positives_2d (np.ndarray):         2D array of positive score values, where axis=1 represents positions
        suboptimals_2d (np.ndarray):       2D array of suboptimal score values, where axis=1 represents positions
        forbiddens_2d (np.ndarray):        2D array of forbidden score values, where axis=1 represents positions

    Returns:
        output (float):  f1-score with reversed sign, such that minimization results in maximizing actual f1-score
    '''

    position_weights = concatenated_weights[:-3]
    positive_weight, suboptimal_weight, forbidden_weight = concatenated_weights[-3:]

    weighted_positives = np.multiply(positives_2d, position_weights).sum(axis=1) * positive_weight
    weighted_suboptimals = np.multiply(suboptimals_2d, position_weights).sum(axis=1) * suboptimal_weight * -1
    weighted_forbiddens = np.multiply(forbiddens_2d, position_weights).sum(axis=1) * forbidden_weight * -1

    weighted_scores = weighted_positives + weighted_suboptimals + weighted_forbiddens

    precision, recall, thresholds = precision_recall_curve(actual_truths, weighted_scores)
    precision_recall_products = precision * recall
    precision_recall_sums = precision + recall
    valid_f1_scores = 2 * precision_recall_products[precision_recall_sums != 0] / precision_recall_sums[precision_recall_sums != 0]
    max_f1_score = np.nanmax(valid_f1_scores)

    minimizable_output = max_f1_score * -1

    return minimizable_output

def optimize_weights(actual_truths, positives_2d, suboptimals_2d, forbiddens_2d):
    '''
    Function that applies Nelder-Mead optimization to find ideal position and score weights to maximize f1-score

    Args:
        actual_truths (np.ndarray):  array of actual truth values as binary integer-encoded labels
        positives_2d (np.ndarray):   2D array of positive score values, where axis=1 represents positions
        suboptimals_2d (np.ndarray): 2D array of suboptimal score values, where axis=1 represents positions
        forbiddens_2d (np.ndarray):  2D array of forbidden score values, where axis=1 represents positions

    Returns:
        None
    '''

    partial_objective = partial(objective_function, actual_truths = actual_truths,
                                positives_2d = positives_2d,
                                suboptimals_2d = suboptimals_2d,
                                forbiddens_2d = forbiddens_2d)

    print("Beginning Nelder-Mead optimization of weights")
    positions_count = positives_2d.shape[1]
    initial_weights = np.ones(positions_count+3, dtype=float)
    callback = lambda xk: print(f"\tCurrent array: {xk}")
    options = {"maxiter": 2000, "adaptive": True}

    result = minimize(partial_objective, initial_weights, method="nelder-mead", callback=callback, options=options)
    position_weights = result.x[:-3]
    positive_score_weight, suboptimal_score_weight, forbidden_score_weight = result.x[-3:]

    return position_weights, positive_score_weight, suboptimal_score_weight, forbidden_score_weight


''' ----------------------------------------------------------------------------------------------------------------
                                           Main ScoredPeptideResult Object
    ---------------------------------------------------------------------------------------------------------------- '''

class ScoredPeptideResult:
    '''
    Class that represents the result of scoring peptides using ConditionalMatrices.score_peptides()
    '''
    def __init__(self, seqs_2d, slice_scores_subsets,
                 positive_scores_2d, suboptimal_scores_2d, forbidden_scores_2d, actual_truths = None,
                 predefined_weights = None):
        '''
        Initialization function to generate the score values and assign them to self

        Args:
            seqs_2d  (np.ndarray):             2D array of single letter code amino acids, where each row is a peptide
            slice_scores_subsets (np.ndarray): array of span lengths in the motif to stratify scores by; e.g. if it is
                                               [6,7,2], then subset scores are derived for positions 1-6, 7:13, & 14:15
            positive_scores_2d (np.ndarray):   standardized predicted signal values for each residue for each peptide
            suboptimal_scores_2d (np.ndarray): suboptimal element scores for each residue for each peptide
            forbidden_scores_2d (np.ndarray):  forbidden element scores for each residue for each peptide
            actual_truths (np.ndarray):        array of actual binary calls for each peptide
            predefined_weights (tuple):        tuple of (position_weights, positive_score_weight,
                                               suboptimal_score_weight, forbidden_score_weight)
        '''

        # Check validity of slice_scores_subsets
        if slice_scores_subsets is not None:
            if slice_scores_subsets.sum() != positive_scores_2d.shape[1]:
                raise ValueError(f"ScoredPeptideResult error: slice_scores_subsets sum ({slice_scores_subsets.sum()}) "
                                 f"does not match axis=1 shape of 2D score arrays ({positive_scores_2d.shape[1]})")

        # Assign constituent sequences to self
        self.sequences_2d = seqs_2d
        self.actual_truths = actual_truths

        # Assign predicted signals score values
        self.positive_scores_2d = positive_scores_2d
        self.positive_scores_raw = positive_scores_2d.sum(axis=1)
        divisor = self.positive_scores_raw.max() * self.positive_scores_2d.shape[1]
        self.positive_scores_adjusted = self.positive_scores_raw / divisor

        # Assign suboptimal element score values
        self.suboptimal_scores_2d = suboptimal_scores_2d
        self.suboptimal_scores = suboptimal_scores_2d.sum(axis=1)

        # Assign forbidden element score values
        self.forbidden_scores_2d = forbidden_scores_2d
        self.forbidden_scores = forbidden_scores_2d.sum(axis=1)

        # Apply and assess score weightings
        self.handle_weights(actual_truths, predefined_weights)

        # Assign sliced score values if slice_scores_subsets was given
        self.slice_scores_subsets = slice_scores_subsets
        self.slice_scores()
        self.stack_scores()

        # Encode source residues with integer values; these will be used for NN training
        self.encode_residues(aa_charac_dict)

    def handle_weights(self, actual_truths = None, predefined_weights = None):
        '''
        Parent function to either optimize weights or apply predefined weights (or all ones if not given)

        Args:
            actual_truths (np.ndarray):        array of actual binary calls for each peptide
            predefined_weights (tuple):        tuple of (position_weights, positive_score_weight,
                                               suboptimal_score_weight, forbidden_score_weight)

        Returns:
            None
        '''

        if actual_truths is not None and predefined_weights is None:
            # Optimize weights for combining sets of scores
            results = optimize_weights(actual_truths, self.positive_scores_2d,
                                       self.suboptimal_scores_2d, self.forbidden_scores_2d)
            position_weights, positive_weight, suboptimal_weight, forbidden_weight = results
            self.apply_weights(position_weights, positive_weight, suboptimal_weight, forbidden_weight)
            self.evaluate_weighted_scores(actual_truths)
        elif predefined_weights is not None:
            # Apply predefined weights and assess them
            position_weights, positive_weight, suboptimal_weight, forbidden_weight = predefined_weights
            self.apply_weights(position_weights, positive_weight, suboptimal_weight, forbidden_weight)
            if actual_truths is not None:
                self.evaluate_weighted_scores(actual_truths)
        else:
            positions_count = self.positive_scores_2d.shape[1]
            position_weights = np.ones(positions_count, dtype=float)
            positive_weight, suboptimal_weight, forbidden_weight = np.ones(3, dtype=float)
            self.apply_weights(position_weights, positive_weight, suboptimal_weight, forbidden_weight)

    def apply_weights(self, position_weights, positive_score_weight, suboptimal_score_weight, forbidden_score_weight):
        '''
        Helper function that applies a set of weights of shape (positions_count+3,)

        Args:
            position_weights (np.ndarray):       array of weights of shape (positions_count,)
            positive_score_weight (int|float):   weight reflecting the contribution of positive_scores to total_scores
            suboptimal_score_weight (int|float): weight reflecting the contribution of positive_scores to total_scores
            forbidden_score_weight (int|float):  weight reflecting the contribution of positive_scores to total_scores

        Returns:
            weighted_scores (np.ndarray):        combined weighted scores; optionally returned in addition to being
                                                 assigned to self
        '''

        # Assign weights to self for future reference
        self.position_weights = position_weights
        self.positive_score_weight = positive_score_weight
        self.suboptimal_score_weight = suboptimal_score_weight
        self.forbidden_score_weight = forbidden_score_weight

        # Apply position weights and collapse to single scores per score type
        self.weighted_positives_2d = self.positive_scores_2d * position_weights
        self.weighted_suboptimals_2d = self.suboptimal_scores_2d * position_weights
        self.weighted_forbiddens_2d = self.forbidden_scores_2d * position_weights

        self.weighted_positives = self.weighted_positives_2d.sum(axis=1)
        self.weighted_suboptimals = self.weighted_suboptimals_2d.sum(axis=1)
        self.weighted_forbiddens = self.weighted_forbiddens_2d.sum(axis=1)

        # Apply score type weights and obtain total weighted scores
        adjusted_weighted_positives = self.weighted_positives * positive_score_weight
        adjusted_weighted_suboptimals = self.weighted_suboptimals * suboptimal_score_weight * -1
        adjusted_weighted_forbiddens = self.weighted_forbiddens * forbidden_score_weight * -1

        weighted_scores = adjusted_weighted_positives + adjusted_weighted_suboptimals + adjusted_weighted_forbiddens
        self.weighted_scores = weighted_scores

        return weighted_scores

    def evaluate_weighted_scores(self, actual_truths):
        '''
        Helper function to evaluate a set of weighted scores against actual truth values

        Args:
            actual_truths (np.ndarray): array of actual truth values as binary integer-encoded labels
            weighted_total_scores:      array of summed weighted scores

        Returns:
            None
        '''

        precision, recall, thresholds = precision_recall_curve(actual_truths, self.weighted_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        best_idx = np.nanargmax(f1_scores)

        self.weighted_score_threshold = thresholds[best_idx]
        above_threshold = np.greater_equal(self.weighted_scores, self.weighted_score_threshold)
        self.weighted_accuracy = np.equal(actual_truths, above_threshold).mean()
        self.weighted_mcc = matthews_corrcoef(actual_truths, above_threshold)

        self.weighted_f1_score = f1_scores[best_idx]
        self.weighted_precision = precision[best_idx]
        self.weighted_recall = recall[best_idx]

        # Generate text report
        self.weights_report = [f"---\n",
                               f"Nelder-Mead Weight Optimization Report\n",
                               f"Final optimized array: {final_weights}\n",
                               f"Weighted score threshold: {self.weighted_score_threshold}\n",
                               f"Accuracy={self.weighted_accuracy}\n",
                               f"MCC={self.weighted_mcc}\n",
                               f"Precision={self.weighted_precision}",
                               f"Recall={self.weighted_recall})\n",
                               f"f1_score={self.weighted_f1_score}\n",
                               f"---\n"]

        for line in self.weights_report:
            print(f"\t" + line)

    def use_predefined_weights(self, position_weights = None, score_weights = None, actual_truths = None):
        '''
        Function that assigns predefined weights to self and optionally evaluates their performance

        Args:
            position_weights (np.ndarray): array of position weights of shape (position_count,)
            score_weights (np.ndarray):    array of score type weights of shape (type_count,), where type_count=3
            actual_truths (np.ndarray): array of actual truth values as binary integer-encoded labels

        Returns:
            None
        '''

        # Declare position weights
        if position_weights is None:
            positions_count = self.positive_scores_2d.shape[1]
            self.position_weights = np.ones(positions_count, dtype=float)
        else:
            self.position_weights = position_weights

        # Declare positive, suboptimal, and forbidden element score weights
        if score_weights is None:
            score_weights = np.ones(3, dtype=float)
        self.positive_score_weight, self.suboptimal_score_weight, self.forbidden_score_weight = score_weights

        # Apply the weights and evaluate if actual_truths is given
        self.weighted_total_scores = self.apply_weights(self.position_weights, self.positive_score_weight,
                                                        self.suboptimal_score_weight, self.forbidden_score_weight)
        if actual_truths is not None:
            self.evaluate_weighted_scores(actual_truths, self.weighted_total_scores)

    def slice_scores(self):
        # Function that generates scores based on sliced subsets of peptide sequences

        self.score_cols = ["Positive_Score_Adjusted", "Suboptimal_Element_Score", "Forbidden_Element_Score"]

        if self.slice_scores_subsets is not None:
            end_position = 0
            self.sliced_positive_scores = []
            self.sliced_suboptimal_scores = []
            self.sliced_forbidden_scores = []

            for subset in self.slice_scores_subsets:
                start_position = end_position
                end_position += subset
                suffix_str = str(start_position) + "-" + str(end_position)

                subset_positive_scores = self.positive_scores_2d[:, start_position:end_position + 1].sum(axis=1)
                self.sliced_positive_scores.append(subset_positive_scores)
                self.score_cols.append("Positive_Score_" + suffix_str)

                subset_suboptimal_scores = self.suboptimal_scores_2d[:, start_position:end_position + 1].sum(axis=1)
                self.sliced_suboptimal_scores.append(subset_suboptimal_scores)
                self.score_cols.append("Suboptimal_Score_" + suffix_str)

                subset_forbidden_scores = self.forbidden_scores_2d[:, start_position:end_position + 1].sum(axis=1)
                self.sliced_forbidden_scores.append(subset_forbidden_scores)
                self.score_cols.append("Forbidden_Score_" + suffix_str)

        else:
            warnings.warn(RuntimeWarning("slice_scores_subsets was not given, so scores have not been sliced"))
            self.sliced_positive_scores = None
            self.sliced_suboptimal_scores = None
            self.sliced_forbidden_scores = None

    def stack_scores(self):
        '''
        Helper function that constructs a 2D array of scores values as columns
        '''

        scores = [self.positive_scores_adjusted, self.suboptimal_scores, self.forbidden_scores]

        if self.slice_scores_subsets is not None:
            for positive_scores_slice in self.sliced_positive_scores:
                scores.append(positive_scores_slice)
            for suboptimal_scores_slice in self.sliced_suboptimal_scores:
                scores.append(suboptimal_scores_slice)
            for forbidden_scores_slice in self.sliced_forbidden_scores:
                scores.append(forbidden_scores_slice)

        # Stack the scores and also use them to construct a dataframe
        self.stacked_scores = np.stack(scores).T
        self.scored_df = pd.DataFrame(self.stacked_scores, columns = self.score_cols)

    def encode_residues(self, aa_charac_dict = aa_charac_dict):
        '''
        Function that creates an encoded representation of sequences by chemical group
        '''

        self.charac_encodings_dict = {}
        binary_encoded_characs = []
        for charac, member_list in aa_charac_dict.items():
            is_member = np.isin(self.sequences_2d, member_list)
            binary_encoded_characs.append(is_member.astype(int))
            self.charac_encodings_dict[charac] = is_member

        # Create a 3D representation of shape (sample_count, position_count, channel_count); each charac is a channel
        self.encoded_characs_3d = np.stack(binary_encoded_characs, axis=2)