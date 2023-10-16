# Defines the ScoredPeptideResult class, which represents peptide scoring results from ConditionalMatrices objects

import numpy as np
import pandas as pd
from functools import partial
from sklearn.metrics import precision_recall_curve, matthews_corrcoef
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from Matrix_Generator.random_search import RandomSearchOptimizer
from visualization_tools.precision_recall import plot_precision_recall
try:
    from Matrix_Generator.config_local import aa_charac_dict
except:
    from Matrix_Generator.config import aa_charac_dict

''' ----------------------------------------------------------------------------------------------------------------
                      Define optimization functions for determining position and score weights
    ---------------------------------------------------------------------------------------------------------------- '''

def accuracy_objective(weights, actual_truths, points_2d, invert_points = False):
    '''
    Objective function for optimizing 2D points array weights based on absolute accuracy

    Args:
        weights (np.ndarray):       array of weights of shape (positions_count,)
        actual_truths (np.ndarray): array of actual truth values as binary integer-encoded labels
        points_2d (np.ndarray):     2D array of points values, where axis=1 represents positions
        invert_points (bool):       set to True if lower points values are better, otherwise set to False

    Returns:
        max_accuracy (float):       best accuracy
    '''

    weighted_points = np.multiply(points_2d, weights).sum(axis=1)
    if invert_points:
        weighted_points = weighted_points * -1

    points_copy = weighted_points.copy()
    points_copy.sort()
    thresholds = (points_copy[:-1] + points_copy[1:]) / 2
    predicted = np.greater_equal(weighted_points, thresholds[:, np.newaxis])
    accuracies = np.mean(predicted == actual_truths, axis=1)
    max_accuracy = np.nanmax(accuracies)

    return max_accuracy

def f1_objective(weights, actual_truths, points_2d, invert_points = False):
    '''
    Objective function for optimizing 2D points array weights based on f1-score (harmonic mean of precision and recall)

    Args:
        weights (np.ndarray):       array of weights of shape (positions_count,)
        actual_truths (np.ndarray): array of actual truth values as binary integer-encoded labels
        points_2d (np.ndarray):     2D array of points values, where axis=1 represents positions
        invert_points (bool):       set to True if lower points values are better, otherwise set to False

    Returns:
        max_f1_score (float):       best f1 score
    '''

    weighted_points = np.multiply(points_2d, weights).sum(axis=1)
    if invert_points:
        weighted_points = weighted_points * -1

    precision, recall, thresholds = precision_recall_curve(actual_truths, weighted_points)
    precision_recall_products = precision * recall
    precision_recall_sums = precision + recall
    valid_f1_scores = 2 * np.divide(precision_recall_products[precision_recall_sums != 0],
                                    precision_recall_sums[precision_recall_sums != 0])
    max_f1_score = np.nanmax(valid_f1_scores)

    return max_f1_score

def r2_objective(weights, signal_values, points_2d, invert_points = False):
    '''
    Objective function for optimizing 2D points array weights based on linear regression R2

    Args:
        weights (np.ndarray):       array of weights of shape (positions_count,)
        signal_values (np.ndarray): array of binding signal values
        points_2d (np.ndarray):     2D array of points values, where axis=1 represents positions
        invert_points (bool):       set to True if lower points values are better, otherwise set to False

    Returns:
        max_f1_score (float):       best f1 score
    '''

    weighted_points = np.multiply(points_2d, weights).sum(axis=1)
    if invert_points:
        weighted_points = weighted_points * -1

    model = LinearRegression()
    model.fit(weighted_points.reshape(-1,1), signal_values.reshape(-1,1))

    predicted_signals = model.predict(weighted_points.reshape(-1,1))
    r2 = r2_score(signal_values, predicted_signals)

    return r2

def optimize_points_2d(points_2d, value_range, mode, actual_truths, signal_values = None, objective_type = "accuracy",
                       invert_points = False):
    '''
    Helper function that applies random search optimization of weights for a 2D points matrix

    Args:
        points_2d (np.ndarray):      2D points array, where rows are scored peptides and columns are sequence positions
        value_range (iterable):      range of allowed weights values
        mode (str):                  optimization mode; must either be "maximize" or "minimize"
        actual_truths (np.ndarray):  array of actual boolean truths
        signal_values (np.ndarray):  array of binding signal values between peptides and the protein bait(s)
        objective_type (str):        defines the objective function as `accuracy`, `f1`, or `r2`
        invert_points (bool):        set to True if lower points values are better, otherwise set to False

    Returns:
        best_weights (np.ndarray):   best position weights for the given points matrix
        x (float):                   value of points_objective for best_weights
    '''

    # Set the appropriate objective function
    if objective_type == "accuracy":
        objective = partial(accuracy_objective, actual_truths = actual_truths, points_2d = points_2d,
                            invert_points = invert_points)
    elif objective_type == "r2":
        objective = partial(r2_objective, signal_values = signal_values, points_2d = points_2d,
                            invert_points = invert_points)
    elif objective_type == "f1":
        objective = partial(f1_objective, actual_truths = actual_truths, points_2d = points_2d,
                            invert_points = invert_points)
    else:
        raise ValueError(f"expected mode to be `f1`, `r2`, or `accuracy`, but got {mode}")

    # Points optimization
    array_len = points_2d.shape[1]
    search_sample = 1000000
    print(f"RandomSearchOptimizer mode: {mode}")
    points_optimizer = RandomSearchOptimizer(objective, array_len, value_range, mode)
    done = False
    while not done:
        points_optimizer.search(search_sample)
        search_again = input("\tSearch again? (Y/N)  ")
        done = search_again != "Y"

    return points_optimizer.best_array, points_optimizer.x


''' ----------------------------------------------------------------------------------------------------------------
                                           Main ScoredPeptideResult Object
    ---------------------------------------------------------------------------------------------------------------- '''

class ScoredPeptideResult:
    '''
    Class that represents the result of scoring peptides using ConditionalMatrices.score_peptides()
    '''
    def __init__(self, seqs_2d, slice_scores_subsets,
                 positive_scores_2d, suboptimal_scores_2d, forbidden_scores_2d, actual_truths = None,
                 signal_values = None, objective_type = "accuracy", predefined_weights = None, fig_path = None,
                 make_df = True):
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
            signal_values (np.ndarray):        array of binding signal values for peptides against protein bait(s)
            objective_type (str):              defines the objective function as `accuracy`, `f1`, or `r2`
            predefined_weights (tuple):        tuple of (position_weights, positive_score_weight,
                                               suboptimal_score_weight, forbidden_score_weight)
            fig_path (str):                    desired file name, as full path, to save precision/recall graph
            make_df (bool):                    whether to generate a dataframe containing scores
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

        # Apply and assess score weightings, and assign them to a dataframe
        self.process_weights(actual_truths, signal_values, objective_type, predefined_weights, fig_path)
        if make_df:
            self.generate_scored_df()

    def process_weights(self, actual_truths = None, signal_values = None, objective_type = "accuracy",
                        predefined_weights = None, fig_path = None):
        '''
        Parent function to either optimize weights or apply predefined weights (or all ones if not given)

        Args:
            actual_truths (np.ndarray):        array of actual binary calls for each peptide
            signal_values (np.ndarray):        array of binding signal values for peptides against protein bait(s)
            objective_type (str):              defines the objective function as `accuracy`, `f1`, or `r2`
            predefined_weights (tuple):        tuple of (positive_score_weights, suboptimal_score_weights,
                                               forbidden_score_weights, type_weights)
            fig_path (str):                    path to save the precision/recall/accuracy plot to

        Returns:
            None
        '''

        if actual_truths is not None and predefined_weights is None:
            # Optimize weights for combining sets of scores
            self.optimize_weights(actual_truths, signal_values, objective_type, fig_path)
            self.evaluate_weighted_scores(actual_truths)
        elif predefined_weights is not None:
            # Apply predefined weights and assess them
            positive_weights, suboptimal_weights, forbidden_weights, type_weights = predefined_weights
            self.apply_weights(positive_weights, suboptimal_weights, forbidden_weights, type_weights)
            if actual_truths is not None:
                self.evaluate_weighted_scores(actual_truths)
        else:
            positions_count = self.positive_scores_2d.shape[1]
            position_weights = np.ones(positions_count, dtype=float)
            positive_weight, suboptimal_weight, forbidden_weight = np.ones(3, dtype=float)
            self.apply_weights(position_weights, positive_weight, suboptimal_weight, forbidden_weight)

    def optimize_weights(self, actual_truths, signal_values = None, objective_type = "accuracy", fig_path = None):
        '''
        Function that applies random search optimization to find ideal position and score weights to maximize f1-score

        Args:
            actual_truths (np.ndarray):  array of actual truth values as binary integer-encoded labels
            signal_values (np.ndarray):  array of binding signal values for peptides against protein bait(s)
            objective_type (str):        defines the objective function as `accuracy`, `f1`, or `r2`
            fig_path (str):              path to save the precision/recall/accuracy plot to

        Returns:
            None
        '''

        # Get weights for each set of points
        print(f"---\n",
              f"Optimizing weights for each 2D points array...")
        weights_range = (0.0, 10.0)
        positions_count = self.positive_scores_2d.shape[1]
        mode = "maximize"
        combined_2d = np.hstack([self.positive_scores_2d,
                                 self.suboptimal_scores_2d * -1,
                                 self.forbidden_scores_2d * -1])
        combined_weights, best_objective_output = optimize_points_2d(combined_2d, weights_range, mode, actual_truths,
                                                                     signal_values, objective_type, invert_points = False)
        self.positives_weights = combined_weights[:positions_count]
        self.suboptimals_weights = combined_weights[positions_count:positions_count*2]
        self.forbiddens_weights = combined_weights[positions_count*2:positions_count*3]

        # Apply weights by points type
        self.weighted_positives_2d = np.multiply(self.positive_scores_2d, self.positives_weights)
        self.weighted_suboptimals_2d = np.multiply(self.suboptimal_scores_2d, self.suboptimals_weights)
        self.weighted_forbiddens_2d = np.multiply(self.forbidden_scores_2d, self.forbiddens_weights)

        self.weighted_positives = self.weighted_positives_2d.sum(axis=1)
        self.weighted_suboptimals = self.weighted_suboptimals_2d.sum(axis=1)
        self.weighted_forbiddens = self.weighted_forbiddens_2d.sum(axis=1)

        # Get total scores
        self.weighted_scores = np.multiply(combined_2d, combined_weights).sum(axis=1)
        self.standardized_weighted_scores = self.weighted_scores - self.weighted_scores.min()
        self.standardized_weighted_scores = self.standardized_weighted_scores / self.standardized_weighted_scores.max()
        print(f"Done! objective function output = {best_objective_output}", "\n---")

        # Plot precisions and recalls for different thresholds
        precisions, recalls, thresholds = precision_recall_curve(actual_truths, self.standardized_weighted_scores)
        threshold_predictions = np.greater_equal(self.standardized_weighted_scores, thresholds[:, np.newaxis])
        accuracies = np.mean(threshold_predictions == actual_truths, axis=1)
        plot_precision_recall(precisions[:-1], recalls[:-1], accuracies, thresholds, fig_path)

    def apply_weights(self, positives_weights, suboptimals_weights, forbiddens_weights, type_weights):
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
        self.positives_weights = positives_weights
        self.suboptimals_weights = suboptimals_weights
        self.forbiddens_weights = forbiddens_weights
        self.type_weights = type_weights
        self.positive_score_weight, self.suboptimal_score_weight, self.forbidden_score_weight = type_weights

        # Apply weights by points type
        self.weighted_positives_2d = np.multiply(self.positive_scores_2d, self.positives_weights)
        self.weighted_suboptimals_2d = np.multiply(self.suboptimal_scores_2d, self.suboptimals_weights)
        self.weighted_forbiddens_2d = np.multiply(self.forbidden_scores_2d, self.forbiddens_weights)

        self.weighted_positives = self.weighted_positives_2d.sum(axis=1)
        self.weighted_suboptimals = self.weighted_suboptimals_2d.sum(axis=1)
        self.weighted_forbiddens = self.weighted_forbiddens_2d.sum(axis=1)

        # Apply score type weights and obtain total weighted scores
        weighted_points_sums = [self.weighted_positives, self.weighted_suboptimals * -1, self.weighted_forbiddens * -1]
        weighted_points_sums = np.stack(weighted_points_sums, axis=1)
        self.weighted_scores = np.multiply(weighted_points_sums, self.type_weights).sum(axis=1)

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
                               f"Weighting Accuracy Report\n",
                               f"Weighted score threshold: {self.weighted_score_threshold}\n",
                               f"Accuracy={self.weighted_accuracy}\n",
                               f"MCC={self.weighted_mcc}\n",
                               f"Precision={self.weighted_precision}\n",
                               f"Recall={self.weighted_recall}\n",
                               f"f1_score={self.weighted_f1_score}\n",
                               f"---\n"]

        for line in self.weights_report:
            print(f"\t" + line)

    def generate_scored_df(self):

        # Fuse the data together into one array
        arrays_list = [self.positive_scores_2d, self.suboptimal_scores_2d, self.forbidden_scores_2d,
                       self.weighted_positives_2d, self.weighted_suboptimals_2d, self.weighted_forbiddens_2d,
                       self.weighted_positives.reshape(-1,1),
                       self.weighted_suboptimals.reshape(-1,1),
                       self.weighted_forbiddens.reshape(-1,1),
                       self.weighted_scores.reshape(-1,1),
                       self.standardized_weighted_scores.reshape(-1,1)]
        arrays_fused = np.hstack(arrays_list)

        # Construct column titles
        positions_count = self.positive_scores_2d.shape[1]
        positions = np.arange(1, positions_count + 1).astype(str)

        col_titles = []
        col_titles.extend(["Unweighted_Positive_#" + position for position in positions])
        col_titles.extend(["Unweighted_Suboptimal_#" + position for position in positions])
        col_titles.extend(["Unweighted_Forbidden_#" + position for position in positions])
        col_titles.extend(["Weighted_Positive_#" + position for position in positions])
        col_titles.extend(["Weighted_Suboptimal_#" + position for position in positions])
        col_titles.extend(["Weighted_Forbidden_#" + position for position in positions])
        col_titles.extend(["Weighted_Positive_Score", "Weighted_Suboptimal_Score", "Weighted_Forbidden_Score"])
        col_titles.extend(["Weighted_Total_Score", "Standardized_Weighted_Score"])

        # Make the dataframe
        self.scored_df = pd.DataFrame(arrays_fused, columns=col_titles)