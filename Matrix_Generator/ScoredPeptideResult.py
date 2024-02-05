# Defines the ScoredPeptideResult class, which represents peptide scoring results from ConditionalMatrices objects

import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import curve_fit, OptimizeWarning
from functools import partial
from sklearn.metrics import precision_recall_curve, matthews_corrcoef, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from Matrix_Generator.random_search import RandomSearchOptimizer
from visualization_tools.precision_recall import plot_precision_recall
try:
    from Matrix_Generator.config_local import aa_charac_dict
except:
    from Matrix_Generator.config import aa_charac_dict

''' ----------------------------------------------------------------------------------------------------------------
                      Define optimization functions for determining position and score weights
    ---------------------------------------------------------------------------------------------------------------- '''

def accuracy_objective(weights, actual_truths, points_2d, disqualified_forbidden = None, invert_points = False,
                       return_threshold = False):
    '''
    Objective function for optimizing 2D points array weights based on absolute accuracy

    Args:
        weights (np.ndarray):                 array of weights of shape (positions_count,)
        actual_truths (np.ndarray):           array of actual truth values as binary integer-encoded labels
        points_2d (np.ndarray):               2D array of points values, where axis=1 represents positions
        disqualified_forbidden (np.ndarray):  bools of whether each row of points_2d has any forbidden residues
        invert_points (bool):                 set to True if lower points values are better, otherwise set to False
        return_threshold (bool):              whether to return the optimal points threshold

    Returns:
        max_accuracy (float):           best accuracy
        optimal_threshold (float):      if return_threshold is True, the optimal threshold will also be returned
    '''

    if points_2d.shape[1] != len(weights):
        raise ValueError(f"points_2d width ({points_2d.shape[1]}) doesn't match weights array (len={len(weights)})")

    weighted_points = np.multiply(points_2d, weights).sum(axis=1)
    if invert_points:
        weighted_points = weighted_points * -1

    if disqualified_forbidden is not None:
        weighted_points[disqualified_forbidden] = np.nan

    points_copy = weighted_points.copy()
    points_copy.sort()
    thresholds = (points_copy[:-1] + points_copy[1:]) / 2
    predicted = np.greater_equal(weighted_points, thresholds[:, np.newaxis])

    accuracies = np.mean(predicted == actual_truths, axis=1)
    max_accuracy = np.nanmax(accuracies)

    if not return_threshold:
        return max_accuracy
    else:
        max_accuracy_idx = np.nanargmax(accuracies)
        optimal_threshold = thresholds[max_accuracy_idx]
        return (max_accuracy, optimal_threshold)

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

def exp_func(x, a, b, c, d):
    # Basic exponential function, where y = ae^(b*(x-c))-d
    return a * np.exp(b * (x - c)) - d

def positive_objective_r2(positive_matrix_weights, signal_values, points_2d, initial_guess_range = (-10, 10)):
    '''
    Objective function for weights optimization of the positive element matrices

    Args:
        positive_matrix_weights (np.ndarray): weights to test
        signal_values (np.ndarray):           signal values
        points_2d (np.ndarray):               matching arrays of points of shape (peptide_count, peptide_length)

    Returns:
        r2 (float):  R2 correlation coefficient
    '''

    weighted_points_2d = np.multiply(points_2d, positive_matrix_weights)
    weighted_points = weighted_points_2d.sum(axis=1)

    initial_guesses = np.random.uniform(initial_guess_range[0], initial_guess_range[1], 400).reshape(100, 4)
    initial_guesses[0] = [1,1,1,1]
    r2 = -np.inf # default if none found

    for initial_guess in initial_guesses:
        try:
            params, _ = curve_fit(exp_func, weighted_points, signal_values, p0 = initial_guess, maxfev = 10000)
            predicted_signal_values = exp_func(weighted_points, *params)
            r2 = r2_score(signal_values, predicted_signal_values)
            break
        except:
            continue

    return r2

def optimize_points_2d(array_len, value_range, mode, objective_function, forced_values_dict = None,
                       search_sample = 100000):
    '''
    Helper function that applies random search optimization of weights for a 2D points matrix

    Args:
        array_len (int):                       number of weights required in each trial array
        value_range (iterable):                range of allowed weights values
        mode (str):                            optimization mode; must either be "maximize" or "minimize"
        objective_function (function|partial): objective function that takes an array of weights as its single argument
        signal_values (np.ndarray):            array of binding signal values between peptides and the protein bait(s)
        search_sample (int):                   number of weights arrays to test per round of selection

    Returns:
        best_weights (np.ndarray):     best position weights for the given points matrix
        x (float):                     value of points_objective for best_weights
    '''

    print(f"\tRandomSearchOptimizer mode: {mode}")
    points_optimizer = RandomSearchOptimizer(objective_function, array_len, value_range, mode, forced_values_dict)
    done = False
    while not done:
        points_optimizer.search(search_sample)
        search_again = input("\tSearch again? (Y/N)  ")
        done = search_again != "Y"

    return (points_optimizer.best_array, points_optimizer.x)


''' ----------------------------------------------------------------------------------------------------------------
                                           Main ScoredPeptideResult Object
    ---------------------------------------------------------------------------------------------------------------- '''

class ScoredPeptideResult:
    '''
    Class that represents the result of scoring peptides using ConditionalMatrices.score_peptides()
    '''
    def __init__(self, seqs_2d, slice_scores_subsets, positive_scores_2d, suboptimal_scores_2d, forbidden_scores_2d,
                 actual_truths = None, signal_values = None, fig_path = None, make_df = True, coefs_path = None,
                 suppress_positives = None, suppress_suboptimals = None, suppress_forbiddens = None,
                 ignore_failed_peptides = True, preview_scatter_plot = True, split_data = False,
                 predefined_weights = None, predefined_binding_std_coefs = None, predefined_binding_exp_params = None,
                 predefined_std_threshold = None):
        '''
        Initialization function to generate the score values and assign them to self

        Args:
            seqs_2d  (np.ndarray):                 2D array of single letter code amino acids; each row is a peptide
            slice_scores_subsets (np.ndarray):     span lengths in the motif to stratify scores by; e.g. if [6,7,2],
                                                   then subset scores are derived for positions 1-6, 7-13, & 14-15
            positive_scores_2d (np.ndarray):       positive element scores matching input sequences
            suboptimal_scores_2d (np.ndarray):     suboptimal element scores matching input sequences
            forbidden_scores_2d (np.ndarray):      forbidden element scores matching input sequences
            actual_truths (np.ndarray):            array of actual binary calls for each peptide
            signal_values (np.ndarray):            array of binding signal values for peptides against protein bait(s)
            fig_path (str):                        desired file name, as full path, to save precision/recall graph
            make_df (bool):                        whether to generate a dataframe containing scores
            coefs_path (str):                      path to save standardization coefficients to
            suppress_positives (np.ndarray):       position indices to suppress effect of positive elements at
            suppress_suboptimals (np.ndarray):     position indices to suppress effect of suboptimal elements at
            suppress_forbiddens (np.ndarray):      position indices to suppress effect of forbidden elements at
            ignore_failed_peptides (bool):         whether to only use signals from passing peptides for positive matrix
            preview_scatter_plot (bool):           whether to show a scatter plot of positive points against signals
            split_data (bool):                     whether to do a train/test split for weights optimization; this is
                                                   separate from the main train/test split
            test_seqs_2d (np.ndarray):             if the main train/test split was done, include test sequences
            predefined_weights (tuple):            predefined tuple of (binding_score_weights, positive_score_weights,
                                                   suboptimal_score_weights, forbidden_score_weights)
            predefined_binding_std_coefs (tuple):  predefined binding score standardization coefficients
            predefined_binding_exp_params (tuple): predefined fitted exponential curve params
            predefined_std_threshold (float): predefined weighted points threshold to consider a peptide positive
        '''

        # Check validity of slice_scores_subsets
        if slice_scores_subsets is not None:
            if slice_scores_subsets.sum() != positive_scores_2d.shape[1]:
                raise ValueError(f"ScoredPeptideResult error: slice_scores_subsets sum ({slice_scores_subsets.sum()}) "
                                 f"does not match axis=1 shape of 2D score arrays ({positive_scores_2d.shape[1]})")

        # Assign constituent sequences and associated values to self
        self.sequences_2d = seqs_2d
        self.actual_truths = actual_truths
        signal_values[signal_values < 0] = 0
        self.all_signal_values = signal_values
        self.passing_signal_values = signal_values[actual_truths]
        self.failing_signal_values = signal_values[~actual_truths]

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
        self.process_weights(fig_path, coefs_path, suppress_positives, suppress_suboptimals, suppress_forbiddens,
                             ignore_failed_peptides, preview_scatter_plot, split_data,
                             predefined_weights, predefined_binding_std_coefs, predefined_binding_exp_params,
                             predefined_std_threshold)
        if make_df:
            self.generate_scored_df()

    def process_weights(self, fig_path = None, coefficients_path = None, suppress_positives = None,
                        suppress_suboptimals = None, suppress_forbiddens = None, ignore_failed_peptides = True,
                        preview_scatter_plot = True, split_data = False, predefined_weights = None,
                        predefined_binding_std_coefs = None, predefined_binding_exp_params = None,
                        predefined_std_threshold = None):
        '''
        Parent function to either optimize weights or apply predefined weights (or all ones if not given)

        Args:
            fig_path (str):                        path to save the precision/recall/accuracy plot to
            coefficients_path (str):               path to save the standardization coefficients to
            suppress_positives (np.ndarray):       position indices to suppress positive elements at
            suppress_suboptimals (np.ndarray):     position indices to suppress suboptimal elements at
            suppress_forbiddens (np.ndarray):      position indices to suppress forbidden elements at
            ignore_failed_peptides (bool):         whether to only use signals from passing peptides for positive matrix
            preview_scatter_plot (bool):           whether to show a scatter plot of positive points against signals
            split_data (bool):                     whether to split the data in a train/test split; this is separate
                                                   from the main train/test split and only refers to weights
                                                   optimization, not training conditional matrix values
            predefined_weights (tuple):            predefined tuple of (binding_score_weights, positive_score_weights,
                                                   suboptimal_score_weights, forbidden_score_weights)
            predefined_binding_std_coefs (tuple):  predefined binding score standardization coefficients
            predefined_binding_exp_params (tuple): predefined fitted exponential curve params
            predefined_std_threshold (float): predefined weighted points threshold to consider a peptide positive
        Returns:
            None
        '''

        if self.actual_truths is not None and predefined_weights is None:
            # Optimize weights for combining sets of scores
            save_path = fig_path.rsplit("/", 1)[0] if fig_path is not None else None
            self.optimize_positive_weights(suppress_positives, ignore_failed_peptides, preview_scatter_plot, save_path)
            if split_data:
                self.optimize_total_weights(fig_path, coefficients_path, suppress_positives, suppress_suboptimals,
                                            suppress_forbiddens, split_data=True)
                retrain_full_dataset = input("\tRetrain on entire dataset to improve generalizability? (Y/N)  ") == "Y"
                if retrain_full_dataset:
                    self.optimize_total_weights(fig_path, coefficients_path, suppress_positives, suppress_suboptimals,
                                                suppress_forbiddens, split_data=False)
            else:
                self.optimize_total_weights(fig_path, coefficients_path, suppress_positives, suppress_suboptimals,
                                            suppress_forbiddens, split_data=False)
            self.evaluate_weighted_scores()

        elif predefined_weights is not None:
            # Apply predefined weights for binding score prediction
            binding_weights, positive_weights, suboptimal_weights, forbidden_weights = predefined_weights
            self.apply_predefined_binding(binding_weights, predefined_binding_std_coefs, predefined_binding_exp_params,
                                          ignore_failed_peptides)
            # Apply predefined weights for binary classification
            self.apply_predefined_weights(positive_weights, suboptimal_weights, forbidden_weights, fig_path,
                                          predefined_std_threshold, coefficients_path, suppress_forbiddens)
            # Evaluate weighted scores
            if self.actual_truths is not None:
                self.evaluate_weighted_scores()

        else:
            positions_count = self.positive_scores_2d.shape[1]
            position_weights = np.ones(positions_count, dtype=float)
            self.apply_predefined_binding(position_weights, (0,1), None, ignore_failed_peptides)
            self.apply_predefined_weights(position_weights, position_weights, position_weights, fig_path,
                                          coefficients_path=coefficients_path, suppress_forbiddens=suppress_forbiddens)
            if self.actual_truths is not None:
                self.evaluate_weighted_scores()

    def apply_predefined_binding(self, binding_positive_weights, binding_standardization_coefs = None,
                                 binding_exp_params = None, ignore_failed_peptides = True):
        '''
        Function that applies random search optimization to find ideal position and score weights to maximize f1-score

        Args:
            binding_positive_weights (np.ndarray): binding positive element score weights
            binding_standardization_coefs (tuple): binding score standardization coefficients
            binding_exp_params (tuple|None):       fitted exponential function parameters
            ignore_failed_peptides (bool):         whether to only use signals from passing peptides for positive matrix

        Returns:
            None
        '''

        # Generate weighted summed points array
        self.binding_positive_weights = binding_positive_weights
        weighted_summed_points = np.sum(np.multiply(self.positive_scores_2d, binding_positive_weights), axis=1)
        self.binding_positive_scores = weighted_summed_points

        # Standardize the binding scores
        if binding_standardization_coefs is not None:
            self.binding_standardization_coefficients = binding_standardization_coefs
            binding_coef_a, binding_coef_b = binding_standardization_coefs
        else:
            binding_coef_a = np.min(weighted_summed_points)
            binding_coef_b = np.max(weighted_summed_points - binding_coef_a)
            self.binding_standardization_coefficients = (binding_coef_a, binding_coef_b)

        standardized_weighted_points = (weighted_summed_points - binding_coef_a) / binding_coef_b
        self.standardized_binding_scores = standardized_weighted_points

        # Calculate R2
        if ignore_failed_peptides:
            weighted_summed_points = weighted_summed_points[self.actual_truths]
            signal_values = self.passing_signal_values
        else:
            signal_values = self.all_signal_values

        if binding_exp_params is not None:
            self.binding_exp_params = binding_exp_params
            predicted_signal_values = exp_func(weighted_summed_points, *binding_exp_params)
            self.binding_score_r2 = r2_score(signal_values, predicted_signal_values)
        else:
            params, _ = curve_fit(exp_func, weighted_summed_points, signal_values, p0=[1,1,1,1])
            self.binding_exp_params = params

            predicted_signal_values = exp_func(weighted_summed_points, *params)
            self.binding_score_r2 = r2_score(signal_values, predicted_signal_values)

            if binding_standardization_coefs is not None:
                # Also fit a curve to standardized binding scores and save the params to self (for future use)
                standardized_weighted_points = self.standardized_binding_scores.copy()
                if ignore_failed_peptides:
                    standardized_weighted_points = standardized_weighted_points[self.actual_truths]

                std_signals = signal_values / np.max(signal_values)
                std_params, _ = curve_fit(exp_func, standardized_weighted_points, std_signals, p0=[0.5,0.5,0.5,0.5])
                self.standardized_binding_exp_params = std_params

                predicted_standardized_signal_values = exp_func(standardized_weighted_points, *std_params)
                self.standardized_binding_r2 = r2_score(std_signals, predicted_standardized_signal_values)

            else:
                self.standardized_binding_exp_params, self.standardized_binding_r2 = None, None

    def apply_predefined_weights(self, positives_weights, suboptimals_weights, forbiddens_weights, fig_path = None,
                                 std_points_threshold = None, coefficients_path = None, suppress_forbiddens = None):
        '''
        Function that applies predefined classification score weights

        Args:
            positives_weights (np.ndarray):    positive element weights for classification
            suboptimals_weights (np.ndarray):  suboptimal element weights for classification
            forbiddens_weights (np.ndarray):   forbidden element weights for classification
            fig_path (str):                    path to save the precision/recall/accuracy plot to
            std_points_threshold (float):      threshold of weighted scores above which a peptide is predicted positive
            coefficients_path (str):           path to save the standardization coefficients to
            suppress_forbiddens (np.ndarray):  position indices to suppress forbidden elements at

        Returns:
            None
        '''

        # Assign weights to self
        self.positives_weights = positives_weights
        self.suboptimals_weights = suboptimals_weights
        self.forbiddens_weights = forbiddens_weights
        self.standardized_weighted_threshold = std_points_threshold

        # Apply weights to scored peptides
        self.weighted_positives = np.sum(self.positive_scores_2d * positives_weights, axis=1)
        self.weighted_suboptimals = np.sum(self.suboptimal_scores_2d * suboptimals_weights, axis=1)
        self.weighted_forbiddens = np.sum(self.forbidden_scores_2d * forbiddens_weights, axis=1)
        self.weighted_accuracy_scores = self.weighted_positives + self.weighted_suboptimals + self.weighted_forbiddens

        # Find disqualified peptides that have forbidden residues
        forbidden_2d = self.forbidden_scores_2d.copy()
        forbidden_2d[:,suppress_forbiddens] = 0
        disqualified_forbidden = np.any(forbidden_2d > 0, axis=1)
        self.weighted_accuracy_scores[disqualified_forbidden] = np.nan

        # Points standardization
        coefficients_path = os.getcwd().rsplit("/")[0] if coefficients_path is None else coefficients_path
        coefficients_path_pkl = os.path.join(coefficients_path, "standardization_coefficients.pkl")
        with open(coefficients_path_pkl, "rb") as f:
            self.standardization_coefficients = pickle.load(f)

        coef_a, coef_b = self.standardization_coefficients[0:2]
        self.standardized_weighted_scores = (self.weighted_accuracy_scores - coef_a) / coef_b

        positives_coef_a, positives_coef_b = self.standardization_coefficients[2:4]
        self.standardized_weighted_positives = (self.weighted_positives - positives_coef_a) / positives_coef_b

        suboptimals_coef_a, suboptimals_coef_b = self.standardization_coefficients[4:6]
        self.standardized_weighted_suboptimals = (self.weighted_suboptimals - suboptimals_coef_a) / suboptimals_coef_b

        forbiddens_coef_a, forbiddens_coef_b = self.standardization_coefficients[6:8]
        self.standardized_weighted_forbiddens = (self.weighted_forbiddens - forbiddens_coef_a) / forbiddens_coef_b

        # Predict the classification
        if std_points_threshold is not None:
            predicted = np.greater_equal(self.standardized_weighted_scores, std_points_threshold)
            self.standardized_threshold_accuracy = np.mean(predicted == self.actual_truths)
        else:
            standardized_weighted_scores = self.standardized_weighted_scores.copy()
            basement_val = np.nanmin(standardized_weighted_scores)
            basement_val = basement_val - (0.01 * basement_val)
            standardized_weighted_scores[np.isnan(standardized_weighted_scores)] = basement_val
            precisions, recalls, thresholds = precision_recall_curve(self.actual_truths,
                                                                     standardized_weighted_scores)
            threshold_predictions = np.greater_equal(standardized_weighted_scores, thresholds[:, np.newaxis])
            accuracies = np.mean(threshold_predictions == self.actual_truths, axis=1)
            top_vals = plot_precision_recall(precisions[:-1], recalls[:-1], accuracies, thresholds, fig_path)
            self.standardized_weighted_threshold, self.standardized_threshold_accuracy = top_vals

    def optimize_positive_weights(self, suppress_positives = None, ignore_failed_peptides = True,
                                  preview_scatter_plot = True, save_path = None):
        '''
        Function that applies random search optimization to find ideal position and score weights to maximize f1-score

        Args:
            suppress_positives (np.ndarray):   position indices to suppress positive elements at
            ignore_failed_peptides (bool):     whether to only use signals from passing peptides for positive matrix
            preview_scatter_plot (bool):       whether to show a scatter plot of summed positive points against signals
            save_path (str):                   path to folder where the scatter plot should be saved

        Returns:
            None
        '''

        # Get weights for each set of points
        print(f"---\n",
              f"Optimizing weights for positive element scores to maximize exponential R2 correlation...")
        weights_range = (0.0, 10.0)

        # Define signal values to use
        if ignore_failed_peptides:
            signal_values = self.passing_signal_values
        else:
            signal_values = self.all_signal_values

        # Optimize positive element score weights
        passing_points_2d = self.positive_scores_2d[self.actual_truths]
        positive_forced_values = {idx: 0 for idx in suppress_positives} if suppress_positives is not None else {}
        positive_objective = partial(positive_objective_r2, signal_values=signal_values, points_2d=passing_points_2d)
        array_len = self.positive_scores_2d.shape[1]

        with warnings.catch_warnings():
            # Bad sets of weights may fail to converge, generating warnings; this code suppresses these warnings
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", OptimizeWarning)
            positive_weights, positive_best_r2 = optimize_points_2d(array_len, weights_range, "maximize",
                                                                    positive_objective, positive_forced_values,
                                                                    search_sample = 100000)

        # Generate weighted summed points array
        self.binding_positive_weights = positive_weights
        weighted_summed_points = np.sum(np.multiply(self.positive_scores_2d, positive_weights), axis=1)
        self.binding_positive_scores = weighted_summed_points

        coefficient_a = np.min(weighted_summed_points)
        standardized_weighted_points = weighted_summed_points - coefficient_a
        coefficient_b = np.abs(np.max(standardized_weighted_points))
        standardized_weighted_points = standardized_weighted_points / coefficient_b
        self.binding_standardization_coefficients = (coefficient_a, coefficient_b)
        self.standardized_binding_scores = standardized_weighted_points.copy()

        # Calculate R2
        if ignore_failed_peptides:
            weighted_summed_points = weighted_summed_points[self.actual_truths]

        params, _ = curve_fit(exp_func, weighted_summed_points, signal_values, p0=[1,1,1,1])
        self.binding_exp_params = params

        predicted_signal_values = exp_func(weighted_summed_points, *params)
        self.binding_score_r2 = r2_score(signal_values, predicted_signal_values)

        # Also fit a curve to standardized binding scores and save the params to self (for future use)
        if ignore_failed_peptides:
            standardized_weighted_points = standardized_weighted_points[self.actual_truths]

        std_signals = signal_values / np.max(signal_values)

        standardized_params, _ = curve_fit(exp_func, standardized_weighted_points, std_signals, p0=[0.5,0.5,0.5,0.5])
        self.standardized_binding_exp_params = standardized_params

        predicted_standardized_signal_values = exp_func(standardized_weighted_points, *standardized_params)
        self.standardized_binding_r2 = r2_score(std_signals, predicted_standardized_signal_values)

        # Show post-weighting scatter plot if necessary
        if preview_scatter_plot:
            print(f"\tPreviewing weighted positive element score sum correlation with signal values...")
            r2 = round(self.standardized_binding_r2, 2)

            a,b,c,d = standardized_params
            c_sign = "+" if c >= 0 else "-"
            d_sign = "+" if d >= 0 else "-"
            equation_text = f"y = {a:.2f}e^({b:.2f}(x{c_sign}{np.abs(c):.2f})) {d_sign}{np.abs(d):.2f}"

            sorted_order = np.argsort(standardized_weighted_points)

            if save_path is not None:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                scatter_path = os.path.join(save_path, "binding_strength_curve.pdf")
            else:
                scatter_path = None

            if ignore_failed_peptides:
                # Also plot failed peptides in a different color
                weighted_points_failed = np.sum(np.multiply(self.positive_scores_2d, positive_weights), axis=1)
                weighted_points_failed = weighted_points_failed[~self.actual_truths]
                standardized_weighted_failed = weighted_points_failed - coefficient_a
                standardized_weighted_failed = standardized_weighted_failed / coefficient_b

                plt.figure(figsize=(8, 6))
                plt.scatter(standardized_weighted_failed, self.failing_signal_values / np.max(signal_values),
                            label = "non-interacting peptides (weighted)", color = "red", alpha = 0.5)
                plt.scatter(standardized_weighted_points, std_signals,
                            label = "interacting peptides (weighted)", color = "blue", alpha = 0.5)
            else:
                plt.figure(figsize=(8, 6))
                plt.scatter(standardized_weighted_points, std_signals,
                            label = "data (weighted)", color = "blue", alpha = 0.5)

            plt.plot(standardized_weighted_points[sorted_order], predicted_standardized_signal_values[sorted_order],
                     label = equation_text, color = "black")
            plt.annotate(f"R2 = {r2}", xy=(0, 1), xycoords="axes fraction", xytext=(10, -10),
                         textcoords="offset points", ha="left", va="top", fontsize = "large")

            plt.legend()
            plt.xlabel("Positive Element Score (standardized weighted sum)", fontsize = "large")
            plt.ylabel("Signal intensity (relative)", fontsize = "large")

            plt.savefig(scatter_path, format="pdf") if scatter_path is not None else None
            plt.show()

    def optimize_total_weights(self, fig_path = None, coefficients_path = None, suppress_positives = None,
                               suppress_suboptimals = None, suppress_forbiddens = None, split_data = True):
        '''
        Function that applies random search optimization to find ideal position and score weights to maximize f1-score

        Args:
            fig_path (str):                    path to save the precision/recall/accuracy plot to
            coefficients_path (str):           path to save the standardization coefficients to
            suppress_suboptimals (np.ndarray): position indices to suppress suboptimal elements at
            suppress_forbiddens (np.ndarray):  position indices to suppress forbidden elements at
            split_data (bool):                 whether to perform a train/test split

        Returns:
            None
        '''

        weights_range = (0.0, 10.0)
        mode = "maximize"
        motif_length = self.positive_scores_2d.shape[1]

        # Optimize weights; first value is a multiplier of summed weighted positive scores
        print(f"\tOptimizing final score values to maximize accuracy in binary calls...")

        # Split into training and testing data
        score_row_indices = np.arange(len(self.positive_scores_2d))
        if split_data: 
            train_indices, test_indices = train_test_split(score_row_indices, test_size=0.25)
    
            train_actual_truths = self.actual_truths[train_indices]
            train_binding_scores = self.binding_positive_scores[train_indices]
            train_positive_2d = self.positive_scores_2d[train_indices,:]
            train_suboptimal_2d = self.suboptimal_scores_2d[train_indices,:]
            train_forbidden_2d = self.forbidden_scores_2d[train_indices,:]
    
            test_actual_truths = self.actual_truths[test_indices]
            test_binding_scores = self.binding_positive_scores[test_indices]
            test_positive_2d = self.positive_scores_2d[test_indices,:]
            test_suboptimal_2d = self.suboptimal_scores_2d[test_indices,:]
            test_forbidden_2d = self.forbidden_scores_2d[test_indices,:]
        
        else: 
            train_indices = score_row_indices
            test_indices = None
            
            train_actual_truths = self.actual_truths
            train_binding_scores = self.binding_positive_scores
            train_positive_2d = self.positive_scores_2d
            train_suboptimal_2d = self.suboptimal_scores_2d
            train_forbidden_2d = self.forbidden_scores_2d
            
            test_actual_truths, test_binding_scores = None, None
            test_positive_2d, test_suboptimal_2d, test_forbidden_2d = None, None, None

        # Generate forced values dicts
        ps_forced = {}
        wps_forced = {}
        suboptimal_forced = {}
        for idx in suppress_positives:
            # Suppressed positive score positions
            ps_forced[idx] = 0
        for idx in suppress_suboptimals:
            # Suppressed suboptimal score positions
            ps_forced[idx + motif_length] = 0
            wps_forced[idx + 1] = 0
            suboptimal_forced[idx] = 0

        # Find disqualified peptides that have forbidden residues
        forbidden_2d = self.forbidden_scores_2d.copy()
        forbidden_2d[:,suppress_forbiddens] = 0
        disqualified_forbidden = np.any(forbidden_2d > 0, axis=1)
        train_disqualified_forbidden = disqualified_forbidden[train_indices]
        if split_data: 
            test_disqualified_forbidden = disqualified_forbidden[test_indices]
        else: 
            test_disqualified_forbidden = None

        # Attempt accuracy optimization with and without positive element scores, then use the better option
        print(f"\t\tAttempting with unweighted positive, suboptimal, and forbidden scores combined...")
        train_ps_2d = np.hstack([train_positive_2d, train_suboptimal_2d * -1])
        complete_ps_2d = np.hstack([self.positive_scores_2d, self.suboptimal_scores_2d * -1])

        train_ps_objective = partial(accuracy_objective, actual_truths = train_actual_truths, points_2d = train_ps_2d,
                                     disqualified_forbidden = train_disqualified_forbidden)

        train_ps_results = optimize_points_2d(train_ps_2d.shape[1], weights_range, mode, train_ps_objective, ps_forced,
                                              search_sample = 500000)
        trained_ps_weights, trained_ps_best_x = train_ps_results

        _, trained_ps_optimal_threshold = accuracy_objective(trained_ps_weights, train_actual_truths, train_ps_2d,
                                                             train_disqualified_forbidden, return_threshold = True)

        print(f"\t\t\tTraining accuracy: {trained_ps_best_x*100:.1f}%")
        if split_data:
            test_ps_2d = np.hstack([test_positive_2d, test_suboptimal_2d * -1])
            test_ps_weighted_scores = np.multiply(test_ps_2d, trained_ps_weights).sum(axis=1)
            test_ps_weighted_scores[test_disqualified_forbidden] = np.nan
            test_ps_predicted = np.greater_equal(test_ps_weighted_scores, trained_ps_optimal_threshold)
            test_ps_accuracy = np.mean(test_ps_predicted == test_actual_truths)
            print(f"\t\t\tTest accuracy: {test_ps_accuracy*100:.1f}%")
        else:
            test_ps_accuracy = None

        print(f"\t\tAttempting with pre-weighted positive, unweighted suboptimal, and unweighted forbidden scores...")
        # Note that weighted positive score sums are multiplied by the motif length to prevent under-weighting
        binding_positive_multiplier = self.suboptimal_scores_2d.shape[1] # scales to prevent underrepresentation
        train_wps_2d = np.hstack([train_binding_scores.reshape(-1,1) * binding_positive_multiplier,
                                  train_suboptimal_2d * -1])
        complete_wps_2d = np.hstack([self.binding_positive_scores.reshape(-1, 1) * binding_positive_multiplier,
                                     self.suboptimal_scores_2d * -1])

        train_wps_objective = partial(accuracy_objective, actual_truths = train_actual_truths, points_2d = train_wps_2d,
                                      disqualified_forbidden = train_disqualified_forbidden)

        train_wps_results = optimize_points_2d(train_wps_2d.shape[1], weights_range, mode, train_wps_objective,
                                               wps_forced, search_sample = 1000000)
        trained_wps_weights, trained_wps_best_x = train_wps_results
        trained_wps_weights[0] = trained_wps_weights[0] * binding_positive_multiplier

        _, trained_wps_optimal_threshold = accuracy_objective(trained_wps_weights, train_actual_truths, train_wps_2d,
                                                              train_disqualified_forbidden, return_threshold = True)

        print(f"\t\t\tTraining accuracy: {trained_wps_best_x*100:.1f}%")
        if split_data:
            test_wps_2d = np.hstack([test_binding_scores.reshape(-1, 1) * binding_positive_multiplier,
                                     test_suboptimal_2d * -1])
            test_wps_weighted_scores = np.multiply(test_wps_2d, trained_wps_weights).sum(axis=1)
            test_wps_weighted_scores[test_disqualified_forbidden] = np.nan
            test_wps_predicted = np.greater_equal(test_wps_weighted_scores, trained_wps_optimal_threshold)
            test_wps_accuracy = np.mean(test_wps_predicted == test_actual_truths)
            print(f"\t\t\tTest accuracy: {test_wps_accuracy*100:.1f}%")
        else:
            test_wps_accuracy = None

        print(f"\t\tAttempting with only suboptimal scores...")
        inverted_suboptimal_2d = self.suboptimal_scores_2d * -1
        train_inverted_suboptimal_2d = train_suboptimal_2d * -1

        train_suboptimal_objective = partial(accuracy_objective, actual_truths = train_actual_truths,
                                             points_2d = train_inverted_suboptimal_2d,
                                             disqualified_forbidden = train_disqualified_forbidden)

        train_suboptimal_results = optimize_points_2d(motif_length, weights_range, mode, train_suboptimal_objective,
                                                      suboptimal_forced, search_sample = 1000000)
        trained_suboptimal_weights, trained_suboptimal_best_x = train_suboptimal_results

        _, trained_suboptimal_threshold = accuracy_objective(trained_suboptimal_weights, train_actual_truths,
                                                             train_inverted_suboptimal_2d, train_disqualified_forbidden,
                                                             return_threshold = True)

        print(f"\t\t\tTraining accuracy: {trained_suboptimal_best_x*100:.1f}%")
        if split_data:
            test_inverted_suboptimal_2d = test_suboptimal_2d * -1
            test_suboptimal_scores = np.multiply(test_inverted_suboptimal_2d, trained_suboptimal_weights).sum(axis=1)
            test_suboptimal_scores[test_disqualified_forbidden] = np.nan
            test_suboptimal_predicted = np.greater_equal(test_suboptimal_scores, trained_suboptimal_threshold)
            test_suboptimal_accuracy = np.mean(test_suboptimal_predicted == test_actual_truths)
            print(f"\t\t\tTest accuracy: {test_suboptimal_accuracy*100:.1f}%")
        else:
            test_suboptimal_accuracy = None

        # Select best
        while True:
            best_accuracy_method = input(f"\t\tSelect a method to proceed (\"ps\", \"wps\", or \"suboptimal\"):  ")
            if best_accuracy_method in ["ps", "wps", "suboptimal"]:
                break
            else:
                print(f"\t\tInvalid option selected ({best_accuracy_method}), please try again.")
        self.best_accuracy_method = best_accuracy_method

        # Assign best to self
        if self.best_accuracy_method == "ps":
            best_objective_output = trained_ps_best_x
            self.positives_weights = trained_ps_weights[:motif_length]
            self.suboptimals_weights = trained_ps_weights[motif_length:2*motif_length]
            self.forbiddens_weights = np.zeros(motif_length, dtype=float)
            if split_data:
                self.training_accuracy = trained_ps_best_x
                self.test_accuracy = test_ps_accuracy
            else:
                self.training_accuracy = trained_ps_best_x
                self.test_accuracy = None
                self.retrained_accuracy = trained_ps_best_x

        elif self.best_accuracy_method == "wps":
            best_objective_output = trained_wps_best_x
            self.positives_weights = self.binding_positive_weights * trained_wps_weights[0]
            self.suboptimals_weights = trained_wps_weights[1:motif_length+1]
            self.forbiddens_weights = np.zeros(motif_length, dtype=float)
            if split_data:
                self.training_accuracy = trained_wps_best_x
                self.test_accuracy = test_wps_accuracy
            else:
                self.training_accuracy = trained_wps_best_x
                self.test_accuracy = None
                self.retrained_accuracy = trained_wps_best_x

        elif self.best_accuracy_method == "suboptimal":
            best_objective_output = trained_suboptimal_best_x
            self.positives_weights = np.zeros(motif_length, dtype=float)
            self.suboptimals_weights = trained_suboptimal_weights
            self.forbiddens_weights = np.zeros(motif_length, dtype=float)
            if split_data:
                self.training_accuracy = trained_suboptimal_best_x
                self.test_accuracy = test_suboptimal_accuracy
            else:
                self.training_accuracy = trained_suboptimal_best_x
                self.test_accuracy = None
                self.retrained_accuracy = trained_suboptimal_best_x

        else:
            raise ValueError(f"self.best_accuracy_method is not valid: {self.best_accuracy_method}")

        print("\tFound optimal weights:")
        print("\t\tPositive element score weights: [", self.positives_weights.round(2), "]")
        print("\t\tSuboptimal element score weights: [", self.suboptimals_weights.round(2), "]")
        print("\t\tForbidden element score weights: [", self.forbiddens_weights.round(2), "]")

        # Apply weights by points type
        self.weighted_positives_2d = np.multiply(self.positive_scores_2d, self.positives_weights)
        self.weighted_suboptimals_2d = np.multiply(self.suboptimal_scores_2d, self.suboptimals_weights)
        self.weighted_forbiddens_2d = np.multiply(self.forbidden_scores_2d, self.forbiddens_weights)

        self.weighted_positives = self.weighted_positives_2d.sum(axis=1)
        self.weighted_suboptimals = self.weighted_suboptimals_2d.sum(axis=1)
        self.weighted_forbiddens = self.weighted_forbiddens_2d.sum(axis=1)

        # Get total raw and standardized scores
        if self.best_accuracy_method == "ps":
            self.weighted_accuracy_scores = np.multiply(complete_ps_2d, trained_ps_weights).sum(axis=1)
        elif self.best_accuracy_method == "wps":
            self.weighted_accuracy_scores = np.multiply(complete_wps_2d, trained_wps_weights).sum(axis=1)
        elif self.best_accuracy_method == "suboptimal":
            self.weighted_accuracy_scores = np.multiply(inverted_suboptimal_2d, trained_suboptimal_weights).sum(axis=1)
        else:
            raise ValueError(f"self.best_accuracy_method is set to {self.best_accuracy_method}")

        coefficient_a = self.weighted_accuracy_scores.min()
        self.standardized_weighted_scores = self.weighted_accuracy_scores - coefficient_a
        coefficient_b = self.standardized_weighted_scores.max()
        self.standardized_weighted_scores = self.standardized_weighted_scores / coefficient_b

        # Plot precisions and recalls for different thresholds
        precisions, recalls, thresholds = precision_recall_curve(self.actual_truths, self.standardized_weighted_scores)
        threshold_predictions = np.greater_equal(self.standardized_weighted_scores, thresholds[:, np.newaxis])
        accuracies = np.mean(threshold_predictions == self.actual_truths, axis=1)
        top_vals = plot_precision_recall(precisions[:-1], recalls[:-1], accuracies, thresholds, fig_path)
        self.standardized_weighted_threshold, self.standardized_threshold_accuracy = top_vals

        # Also get stratified raw and standardized positive/suboptimal/forbidden scores
        positives_coefficient_a = self.weighted_positives.min()
        self.standardized_weighted_positives = self.weighted_positives - positives_coefficient_a
        positives_coefficient_b = self.standardized_weighted_positives.max()
        if positives_coefficient_b != 0:
            self.standardized_weighted_positives = self.standardized_weighted_positives / positives_coefficient_b
        else:
            positives_coefficient_b = 1

        suboptimals_coefficient_a = self.weighted_suboptimals.min()
        self.standardized_weighted_suboptimals = self.weighted_suboptimals - suboptimals_coefficient_a
        suboptimals_coefficient_b = self.standardized_weighted_suboptimals.max()
        if suboptimals_coefficient_b != 0:
            self.standardized_weighted_suboptimals = self.standardized_weighted_suboptimals / suboptimals_coefficient_b
        else:
            suboptimals_coefficient_b = 1

        forbiddens_coefficient_a = self.weighted_forbiddens.min()
        self.standardized_weighted_forbiddens = self.weighted_forbiddens - forbiddens_coefficient_a
        forbiddens_coefficient_b = self.standardized_weighted_forbiddens.max()
        if forbiddens_coefficient_b != 0:
            self.standardized_weighted_forbiddens = self.standardized_weighted_forbiddens / forbiddens_coefficient_b
        else:
            forbiddens_coefficient_b = 1

        # Save standardization coefficients
        self.standardization_coefficients = (coefficient_a, coefficient_b,
                                             positives_coefficient_a, positives_coefficient_b,
                                             suboptimals_coefficient_a, suboptimals_coefficient_b,
                                             forbiddens_coefficient_a, forbiddens_coefficient_b)
        coefficients_path = os.getcwd().rsplit("/")[0] if coefficients_path is None else coefficients_path
        coefficients_path_pkl = os.path.join(coefficients_path, "standardization_coefficients.pkl")
        with open(coefficients_path_pkl, "wb") as f:
            pickle.dump(self.standardization_coefficients, f)

        print(f"\tDone! objective function output = {best_objective_output}", "\n---")

    def evaluate_weighted_scores(self):
        '''
        Helper function to evaluate a set of weighted scores against actual truth values
        '''

        # Find f1 score, precision, recall, and MCC
        calls = self.standardized_weighted_scores >= self.standardized_weighted_threshold
        TPs = np.logical_and(calls, self.actual_truths)
        FPs = np.logical_and(calls, ~self.actual_truths)
        TNs = np.logical_and(~calls, ~self.actual_truths)
        FNs = np.logical_and(~calls, self.actual_truths)
        self.weighted_precision = TPs.sum() / (TPs.sum() + FPs.sum())
        self.weighted_recall = TNs.sum() / (TNs.sum() + FNs.sum())
        self.weighted_f1_score = 2 * np.divide(self.weighted_precision * self.weighted_recall,
                                               self.weighted_precision + self.weighted_recall)
        self.weighted_mcc = matthews_corrcoef(self.actual_truths, calls)

        # Generate text report
        binding_stds = self.binding_standardization_coefficients
        totals_stds = self.standardization_coefficients[0:2]
        pes_stds = self.standardization_coefficients[2:4]
        ses_stds = self.standardization_coefficients[4:6]
        fes_stds = self.standardization_coefficients[6:8]
        self.weights_report = [f"---\n",
                               f"Classification Accuracy Report\n",
                               f"---\n",
                               f"Binding Positive Score Weights: {self.binding_positive_weights}\n",
                               f"Binding Positive Score Std Coefficients: a={binding_stds[0]}, b={binding_stds[1]}\n",
                               f"Binding Weighted Score Exponential R2 = {self.binding_score_r2}\n",
                               f"---\n",
                               f"Positive Element Score Weights: {self.positives_weights}\n",
                               f"Positive Element Score Std Coefficients: a={pes_stds[0]}, b={pes_stds[1]}\n",
                               f"---\n",
                               f"Suboptimal Element Score Weights: {self.suboptimals_weights}\n",
                               f"Suboptimal Element Score Std Coefficients: a={ses_stds[0]}, b={ses_stds[1]}\n",
                               f"---\n",
                               f"Forbidden Element Score Weights: {self.forbiddens_weights}\n",
                               f"Forbidden Element Score Std Coefficients: a={fes_stds[0]}, b={fes_stds[1]}\n",
                               f"---\n",
                               f"Weighted Total Score Std Coefficients: a={totals_stds[0]}, b={totals_stds[1]}\n",
                               f"Standardized Total Score Threshold: {self.standardized_weighted_threshold}\n",
                               f"---\n",
                               f"Thresholded Accuracy: {self.standardized_threshold_accuracy}\n",
                               f"MCC={self.weighted_mcc}\n",
                               f"Precision={self.weighted_precision}\n",
                               f"Recall={self.weighted_recall}\n",
                               f"f1_score={self.weighted_f1_score}\n",
                               f"---\n"]

        for line in self.weights_report:
            print(f"\t" + line.strip("\n"))

    def generate_scored_df(self):

        # Fuse the data together into one array
        calls = np.greater_equal(self.standardized_weighted_scores, self.standardized_weighted_threshold)
        arrays_list = [self.positive_scores_2d, self.suboptimal_scores_2d, self.forbidden_scores_2d,
                       self.weighted_positives_2d, self.weighted_suboptimals_2d, self.weighted_forbiddens_2d,
                       self.binding_positive_scores.reshape(-1,1),
                       self.weighted_positives.reshape(-1,1),
                       self.weighted_suboptimals.reshape(-1,1),
                       self.weighted_forbiddens.reshape(-1,1),
                       self.weighted_accuracy_scores.reshape(-1,1),
                       self.standardized_binding_scores.reshape(-1,1),
                       self.standardized_weighted_positives.reshape(-1,1),
                       self.standardized_weighted_suboptimals.reshape(-1,1),
                       self.standardized_weighted_forbiddens.reshape(-1,1),
                       self.standardized_weighted_scores.reshape(-1,1),
                       calls.reshape(-1,1)]
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
        col_titles.extend(["Binding_Strength_Score", "Weighted_Positive_Score", "Weighted_Suboptimal_Score",
                           "Weighted_Forbidden_Score", "Weighted_Total_Score",
                           "Standardized_Binding_Strength", "Standardized_Weighted_Positive_Score",
                           "Standardized_Weighted_Suboptimal_Score", "Standardized_Weighted_Forbidden_Score",
                           "Standardized_Weighted_Total_Score", "Final_Boolean_Classification"])

        # Make the dataframe
        self.scored_df = pd.DataFrame(arrays_fused, columns=col_titles)