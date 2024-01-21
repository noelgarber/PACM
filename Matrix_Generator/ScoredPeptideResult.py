# Defines the ScoredPeptideResult class, which represents peptide scoring results from ConditionalMatrices objects

import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import curve_fit, OptimizeWarning
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

def accuracy_objective(weights, actual_truths, points_2d, disqualified_forbidden = None, invert_points = False):
    '''
    Objective function for optimizing 2D points array weights based on absolute accuracy

    Args:
        weights (np.ndarray):                 array of weights of shape (positions_count,)
        actual_truths (np.ndarray):           array of actual truth values as binary integer-encoded labels
        points_2d (np.ndarray):               2D array of points values, where axis=1 represents positions
        disqualified_forbidden (np.ndarray):  bools of whether each row of points_2d has any forbidden residues
        invert_points (bool):                 set to True if lower points values are better, otherwise set to False

    Returns:
        max_accuracy (float):           best accuracy
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
                 actual_truths = None, signal_values = None, predefined_weights = None, fig_path = None,
                 make_df = True, coefficients_path = None, suppress_positives = None, suppress_suboptimals = None,
                 suppress_forbiddens = None, ignore_failed_peptides = True, preview_scatter_plot = True):
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
            predefined_weights (tuple):        tuple of (position_weights, positive_score_weight,
                                               suboptimal_score_weight, forbidden_score_weight)
            fig_path (str):                    desired file name, as full path, to save precision/recall graph
            make_df (bool):                    whether to generate a dataframe containing scores
            coefficients_path (str):           path to save standardization coefficients to; required by motif predictor
            suppress_positives (np.ndarray):   array of position indices to suppress effect of positive elements at
            suppress_suboptimals (np.ndarray): array of position indices to suppress effect of suboptimal elements at
            suppress_forbiddens (np.ndarray):  array of position indices to suppress effect of forbidden elements at
            ignore_failed_peptides (bool):     whether to only use signals from passing peptides for positive matrix
            preview_scatter_plot (bool):       whether to show a scatter plot of summed positive points against signals
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
        self.process_weights(predefined_weights, fig_path, coefficients_path, suppress_positives, suppress_suboptimals,
                             suppress_forbiddens, ignore_failed_peptides, preview_scatter_plot)
        if make_df:
            self.generate_scored_df()

    def process_weights(self, predefined_weights = None, fig_path = None, coefficients_path = None,
                        suppress_positives = None, suppress_suboptimals = None, suppress_forbiddens = None,
                        ignore_failed_peptides = True, preview_scatter_plot = True):
        '''
        Parent function to either optimize weights or apply predefined weights (or all ones if not given)

        Args:
            predefined_weights (tuple):        tuple of (positive_score_weights, suboptimal_score_weights,
                                               forbidden_score_weights, type_weights)
            fig_path (str):                    path to save the precision/recall/accuracy plot to
            coefficients_path (str):           path to save the standardization coefficients to
            suppress_positives (np.ndarray):   position indices to suppress positive elements at
            suppress_suboptimals (np.ndarray): position indices to suppress suboptimal elements at
            suppress_forbiddens (np.ndarray):  position indices to suppress forbidden elements at
            ignore_failed_peptides (bool):     whether to only use signals from passing peptides for positive matrix
            preview_scatter_plot (bool):       whether to show a scatter plot of summed positive points against signals

        Returns:
            None
        '''

        if self.actual_truths is not None and predefined_weights is None:
            # Optimize weights for combining sets of scores
            save_path = fig_path.rsplit("/", 1)[0]
            self.optimize_positive_weights(suppress_positives, ignore_failed_peptides, preview_scatter_plot, save_path)
            self.optimize_total_weights(fig_path, coefficients_path,
                                        suppress_positives, suppress_suboptimals, suppress_forbiddens)
            self.evaluate_weighted_scores()
        elif predefined_weights is not None:
            # Apply predefined weights and assess them
            positive_weights, suboptimal_weights, forbidden_weights, type_weights = predefined_weights
            self.apply_weights(positive_weights, suboptimal_weights, forbidden_weights, type_weights)
            if self.actual_truths is not None:
                self.evaluate_weighted_scores()
        else:
            positions_count = self.positive_scores_2d.shape[1]
            position_weights = np.ones(positions_count, dtype=float)
            positive_weight, suboptimal_weight, forbidden_weight = np.ones(3, dtype=float)
            self.apply_weights(position_weights, positive_weight, suboptimal_weight, forbidden_weight)

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
                               suppress_suboptimals = None, suppress_forbiddens = None):
        '''
        Function that applies random search optimization to find ideal position and score weights to maximize f1-score

        Args:
            fig_path (str):                    path to save the precision/recall/accuracy plot to
            coefficients_path (str):           path to save the standardization coefficients to
            suppress_suboptimals (np.ndarray): position indices to suppress suboptimal elements at
            suppress_forbiddens (np.ndarray):  position indices to suppress forbidden elements at

        Returns:
            None
        '''

        weights_range = (0.0, 10.0)
        mode = "maximize"
        motif_length = self.positive_scores_2d.shape[1]

        # Optimize weights; first value is a multiplier of summed weighted positive scores
        print(f"\tOptimizing final score values to maximize accuracy in binary calls...")

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

        # Attempt accuracy optimization with and without positive element scores, then use the better option
        print(f"\t\tAttempting with unweighted positive, suboptimal, and forbidden scores combined...")
        positive_suboptimal_2d = np.hstack([self.positive_scores_2d,
                                            self.suboptimal_scores_2d * -1])

        positive_suboptimal_objective = partial(accuracy_objective, actual_truths = self.actual_truths,
                                points_2d = positive_suboptimal_2d, disqualified_forbidden = disqualified_forbidden)
        combined_array_len = positive_suboptimal_2d.shape[1]

        ps_optimized_results = optimize_points_2d(combined_array_len, weights_range, mode,
                                                   positive_suboptimal_objective, ps_forced,
                                                   search_sample = 500000)
        ps_preliminary_weights, ps_best_x = ps_optimized_results

        print(f"\t\tAttempting with pre-weighted positive, unweighted suboptimal, and unweighted forbidden scores...")
        # Note that weighted positive score sums are multiplied by the motif length to prevent under-weighting
        binding_positive_multiplier = self.suboptimal_scores_2d.shape[1] # scales to prevent underrepresentation
        wps_combined_2d = np.hstack([self.binding_positive_scores.reshape(-1, 1) * binding_positive_multiplier,
                                     self.suboptimal_scores_2d * -1])
        wps_objective = partial(accuracy_objective, actual_truths = self.actual_truths, points_2d = wps_combined_2d,
                                disqualified_forbidden = disqualified_forbidden)

        wps_combined_array_len = wps_combined_2d.shape[1]
        wps_optimized_results = optimize_points_2d(wps_combined_array_len, weights_range, mode, wps_objective,
                                                   wps_forced, search_sample = 1000000)
        wps_preliminary_weights, wps_best_x = wps_optimized_results
        wps_preliminary_weights[0] = wps_preliminary_weights[0] * binding_positive_multiplier

        print(f"\t\tAttempting with only suboptimal scores...")
        inverted_suboptimal_2d = self.suboptimal_scores_2d * -1
        suboptimal_objective = partial(accuracy_objective, actual_truths = self.actual_truths,
                                       points_2d = inverted_suboptimal_2d,
                                       disqualified_forbidden = disqualified_forbidden)

        suboptimal_optimized_results = optimize_points_2d(motif_length, weights_range, mode, suboptimal_objective,
                                                          suboptimal_forced, search_sample = 1000000)
        suboptimal_preliminary_weights, suboptimal_best_x = suboptimal_optimized_results

        # Select best
        if ps_best_x > wps_best_x and ps_best_x > suboptimal_best_x:
            print(f"\t\tBest combination: positive, suboptimal, and forbidden scores (x={ps_best_x})")
            self.best_accuracy_method = "ps"
            best_objective_output = ps_best_x
            self.positives_weights = ps_preliminary_weights[:motif_length]
            self.suboptimals_weights = ps_preliminary_weights[motif_length:2*motif_length]
            self.forbiddens_weights = np.zeros(motif_length, dtype=float)
        elif wps_best_x > ps_best_x and wps_best_x > suboptimal_best_x:
            print(f"\t\tBest combination: pre-weighted positive score sums with suboptimal and forbidden scores",
                  f"(x={wps_best_x})")
            self.best_accuracy_method = "wps"
            best_objective_output = wps_best_x
            self.positives_weights = self.binding_positive_weights * wps_preliminary_weights[0]
            self.suboptimals_weights = wps_preliminary_weights[1:motif_length+1]
            self.forbiddens_weights = np.zeros(motif_length, dtype=float)
        else:
            print(f"\t\tBest combination: suboptimal only (x={suboptimal_best_x})")
            self.best_accuracy_method = "suboptimal"
            best_objective_output = suboptimal_best_x
            self.positives_weights = np.zeros(motif_length, dtype=float)
            self.suboptimals_weights = suboptimal_preliminary_weights
            self.forbiddens_weights = np.zeros(motif_length, dtype=float)

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
            self.weighted_accuracy_scores = np.multiply(positive_suboptimal_2d, ps_preliminary_weights).sum(axis=1)
        elif self.best_accuracy_method == "wps":
            self.weighted_accuracy_scores = np.multiply(wps_combined_2d, wps_preliminary_weights).sum(axis=1)
        elif self.best_accuracy_method == "suboptimal":
            self.weighted_accuracy_scores = np.multiply(inverted_suboptimal_2d, suboptimal_preliminary_weights).sum(axis=1)
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
        self.weighted_accuracy_scores = np.multiply(weighted_points_sums, self.type_weights).sum(axis=1)

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