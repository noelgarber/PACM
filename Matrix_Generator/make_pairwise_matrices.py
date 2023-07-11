# This script conducts residue-residue pairwise analysis to generate context-aware SLiM matrices and back-calculated scores.

import numpy as np
import pandas as pd
import os
import time
import multiprocessing
from tqdm import trange
from functools import partial
from Matrix_Generator.ConditionalMatrix import ConditionalMatrices, default_data_params, default_matrix_params
from Matrix_Generator.conditional_scoring import apply_motif_scores
from general_utils.general_utils import save_dataframe, save_weighted_matrices, unravel_seqs
from general_utils.weights_utils import permute_weights
from general_utils.matrix_utils import add_matrix_weights
from general_utils.general_vars import aa_charac_dict
from general_utils.user_helper_functions import get_position_weights, get_possible_weights
from general_utils.statistics import optimize_threshold_fdr, apply_threshold, fdr_for_optimizer

'''------------------------------------------------------------------------------------------------------------------
                        Define functions for parallelized optimization of matrix weights
   ------------------------------------------------------------------------------------------------------------------'''

def process_weights_chunk(chunk, matrix_arrays_dict, matrix_index, source_df, slim_length, sequence_col,
                          significance_col, score_col, significant_str = "Yes", allowed_rate_divergence = 0.2,
                          dict_of_aa_characs = None, convert_phospho = True, verbose = False):
    '''
    Lower helper function for parallelization of position weight optimization; processes chunks of permuted weights

    Note that matrix_arrays_dict, unlike matrices_dict which appears elsewhere, contains numpy arrays instead of
    pandas dataframes. This greatly improves performance, but requires the matching pd.Index to be passed separately.

    Args:
        chunk (np.ndarray):         the chunk of permuted weights currently being processed
        matrix_arrays_dict (dict):  the dictionary of position-type rules --> unweighted matrices as np.ndarray
        matrix_index (pd.Index):    the original index for the matrix arrays, as pd.Index, which may be queried using
                                    get_indexer_for() without needing to query each matrix df
        source_df (pd.DataFrame):   the dataframe containing source peptide-protein binding data
        slim_length (int):          the length of the motif being studied
        sequence_col (str):         the name of the column where sequences are stored
        significance_col (str):     the name of the column where significance information is found
        score_col (str):            the name of the column where motif scores are found
        significant_str (str):      the string representing a pass in source_df[significance_col]
        allowed_rate_divergence (float): the maximum allowed difference between FDR and FOR pairs
        dict_of_aa_characs (dict):  the dictionary of chemical_class --> [amino acid members]
        chemical_class_dict (dict): an inverted dict of amino_acid --> chemical_characteristic; auto-generated if None
        convert_phospho (bool):     whether to convert phospho-residues to non-phospho before doing lookups

    Returns:
        results_tuple (tuple):  (chunk_best_fdr, chunk_best_for, chunk_best_score_threshold, chunk_best_weights,
                                 chunk_best_weighted_matrices_dict, chunk_best_source_df)
    '''

    output_df = source_df.copy()

    passes_bools = source_df[significance_col].values == significant_str

    start_time = time.time()
    # Convert the dictionary of matrices into a 3D numpy array with a dedicated pandas indexer
    ordered_keys = list(matrix_arrays_dict.keys())
    ordered_arrays = list(matrix_arrays_dict.values())
    matrix_of_matrices = np.stack(ordered_arrays) # shape = (array_count, row_count, column_count)

    # Convert keys to encoded chemical classes
    classes_from_keys = []
    for key in ordered_keys:
        key = key.split("#")[-1]
        elements = key.split("=")
        chemical_class = elements[1]
        classes_from_keys.append(chemical_class)
    chemical_classes = np.unique(classes_from_keys)
    chemical_class_encoder = dict(zip(chemical_classes, np.arange(len(chemical_classes))))

    # Get the encoded chemical class dict
    encoded_chemical_classes = {}
    if not isinstance(dict_of_aa_characs, dict):
        dict_of_aa_characs = aa_charac_dict.copy()
    for chemical_class, members in dict_of_aa_characs.items():
        encoded_chemical_class = chemical_class_encoder.get(chemical_class)
        for aa in members:
            encoded_chemical_classes[aa] = encoded_chemical_class
    encoded_class_count = len(dict_of_aa_characs)

    end_time = time.time()
    print(f"Time elapsed while building encoded chemical class dict: {round(end_time-start_time, 3)} s") if verbose else None

    # Convert sequences to a 2D array upfront for performance improvement
    sequences = source_df[sequence_col].to_numpy()
    sequences_2d = unravel_seqs(sequences, slim_length, convert_phospho)

    # Initialize what will become a list of tuples of (best_score, best_fdr, best_for) matching the indices of chunk
    optimal_values = []

    for weights_array in chunk:
        t0 = time.time()
        # Apply the current weights_array to the dict of matrices with numpy; takes almost no time
        current_matrix_of_matrices = matrix_of_matrices * weights_array
        t1 = time.time()

        # Get the array of scores for peptide entries in source_df using the current set of weighted matrices
        scores_array = apply_motif_scores(output_df, slim_length, current_matrix_of_matrices, matrix_index,
                                          encoded_chemical_classes, encoded_class_count, sequences_2d,
                                          convert_phospho = convert_phospho, return_array = True)
        t2 = time.time()

        # Determine the optimal threshold score that gives balanced FDR/FOR values, which are inversely correlated
        score_range_series = np.linspace(scores_array.min(), scores_array.max(), num=100)
        best_results = optimize_threshold_fdr(None, score_range_series, passes_bools = passes_bools,
                                              scores_array = scores_array, verbose = verbose)
        current_best_score, current_best_fdr, current_best_for = best_results
        t3 = time.time()

        total_elapsed = t3 - t0

        print(f"Total time for current loop: {round(total_elapsed, 3)} s",
              f"\n\tCheckpoint #1: {round((t1-t0)/total_elapsed, 3)}",
              f"\n\tCheckpoint #2: {round((t2-t1)/total_elapsed, 2)}",
              f"\n\tCheckpoint #3 {round((t3-t2)/total_elapsed, 2)}") if verbose else None

        optimal_values.append((current_best_score, current_best_fdr, current_best_for))

    # Find the chunk index for the weights array that produces the lowest optimal FDR & FOR values (using the mean)
    optimal_values_array = np.array(optimal_values)
    best_index = fdr_for_optimizer(optimal_values_array[:,1:], allowed_rate_divergence)

    chunk_best_score_threshold, chunk_best_fdr, chunk_best_for = optimal_values_array[best_index]
    chunk_best_weights = chunk[best_index]

    # Get the matching dict of weighted matrices and use it to apply final scores to output_df
    best_weighted_matrices_dict = add_matrix_weights(chunk_best_weights, matrices_dict = matrix_arrays_dict)
    weighted_matrix_of_matrices = matrix_of_matrices * chunk_best_weights
    chunk_best_source_df = apply_motif_scores(output_df, slim_length, weighted_matrix_of_matrices, matrix_index,
                                              encoded_chemical_classes, encoded_class_count, sequences_2d, score_col,
                                              convert_phospho = convert_phospho, add_residue_cols = True,
                                              in_place = False, return_array = False)

    results_tuple = (chunk_best_fdr, chunk_best_for, chunk_best_score_threshold,
                     chunk_best_weights, best_weighted_matrices_dict, chunk_best_source_df)

    return results_tuple

def process_weights(weights_array_chunks, conditional_matrices, slim_length, source_df,
                    sequence_col, significance_col, significant_str, score_col, allowed_rate_divergence = 0.2,
                    dict_of_aa_characs = None, convert_phospho = True):
    '''
    Upper helper function for parallelization of position weight optimization; processes weights by chunking

    Args:
        weights_array_chunks (list):                list of chunks as numpy arrays for feeding to process_weights_chunk
        conditional_matrices (ConditionalMatrices): unweighted conditional position-weighted matrices
        slim_length (int):                          length of the motif being studied
        source_df (pd.DataFrame):                   source peptide-protein binding data
        sequence_col (str):                         col name in chunk where sequences are stored
        significance_col (str):                     col name in chunk where significance information is found
        significant_str (str):                      value in significance_col that denotes significance, e.g. "Yes"
        score_col (str):                            col name where motif scores are found
        allowed_rate_divergence (float):            maximum allowed difference between FDR and FOR pairs
        dict_of_aa_characs (dict):                  dict of chemical_class --> [amino acid members]
        convert_phospho (bool):                     whether to convert phospho-aa's to non-phospho before doing lookups

    Returns:
        results_list (list):     the list of results sets for all the weights arrays
    '''

    pool = multiprocessing.Pool()
    if not isinstance(dict_of_aa_characs, dict):
        dict_of_aa_characs = aa_charac_dict.copy()

    process_partial = partial(process_weights_chunk, matrix_arrays_dict = conditional_matrices.matrix_arrays_dict,
                              matrix_index = conditional_matrices.index, source_df = source_df, slim_length = slim_length,
                              sequence_col = sequence_col, significance_col = significance_col, score_col = score_col,
                              significant_str = significant_str, dict_of_aa_characs = dict_of_aa_characs,
                              convert_phospho = convert_phospho)

    results = None
    best_rate_mean = 9999

    with trange(len(weights_array_chunks), desc="Processing weights") as pbar:
        for chunk_results in pool.imap_unordered(process_partial, weights_array_chunks):
            rate_mean = (chunk_results[0] + chunk_results[1]) / 2
            rate_delta = abs(chunk_results[0] - chunk_results[1])
            if rate_mean < best_rate_mean and rate_delta <= allowed_rate_divergence:
                best_rate_mean = rate_mean
                results = chunk_results
                print(f"\tNew record: FDR={results[0]} | FOR={results[1]} | weights={results[3]}")

            pbar.update()

    pool.close()
    pool.join()

    return results

def find_optimal_weights(input_df, slim_length, conditional_matrices, sequence_col, significance_col, significant_str,
                         score_col, matrix_output_folder, output_folder, possible_weights = None,
                         dict_of_aa_characs = aa_charac_dict, convert_phospho = True, save_pickled_matrix_dict = True):
    '''
    Parent function for finding optimal position weights to generate optimally weighted matrices

    Args:
        input_df (pd.DataFrame):                     contains peptide sequences, signal values, and significances
        slim_length (int):                           length of the motif being studied
        conditional_matrices (ConditionalMatrices):  unweighted conditional matrices based on type-position rules
        sequence_col (str):                          col name where sequences are stored
        significance_col (str):                      col name where significance information is found
        significant_str (str):                       value in significance_col that denotes significance, e.g. "Yes"
        score_col (str):                             col name where motif scores are found
        possible_weights (list):                     list of numpy arrays with possible weights at each position
        dict_of_aa_characs (dict):                   dictionary of chemical_class --> [amino acid members]
        matrix_output_folder (str):                  path for saving weighted matrices
        output_folder (str):                         path for saving the scored data
        convert_phospho (bool):                      whether to convert phospho-residues to non-phospho before lookups
        save_pickled_matrix_dict (bool):             whether to save a pickled version of the dict of matrices

    Returns:
        results_tuple (tuple):  (best_fdr, best_for, best_score_threshold, best_weights, best_weighted_matrices_dict, best_dens_df)
    '''

    output_df = input_df.copy()

    # Get the permuted weights and break into chunks based on CPU count, for parallelization
    possible_weights = get_possible_weights(slim_length) if possible_weights is None else possible_weights
    weights_array = permute_weights(slim_length, possible_weights)
    weights_array_count = weights_array.shape[0]
    cpu_count = os.cpu_count()
    chunk_size = round(weights_array_count / cpu_count)
    if chunk_size > 10000:
        chunk_size = 10000
    elif chunk_size < 10:
        chunk_size = 10
    weights_array_chunks = [weights_array[i:i + chunk_size] for i in range(0, len(weights_array), chunk_size)]

    # Run the parallelized optimization process
    results = process_weights(weights_array_chunks, conditional_matrices, slim_length, output_df, sequence_col,
                              significance_col, significant_str, score_col, allowed_rate_divergence = 0.2,
                              dict_of_aa_characs = dict_of_aa_characs, convert_phospho = convert_phospho)
    best_fdr, best_for, best_score_threshold, best_weights, best_weighted_matrices_dict, best_scored_df = results
    print("\t---\n", f"\tOptimal weights of {best_weights} gave FDR = {best_fdr} and FOR = {best_for} at a SLiM score threshold > {best_score_threshold}")

    # Rebuild best_weighted_matrices_dict into a dict of dataframes rather than numpy arrays
    for key, arr in best_weighted_matrices_dict.items():
        df = pd.DataFrame(arr, index = conditional_matrices.index, columns = conditional_matrices.columns)
        best_weighted_matrices_dict[key] = df

    # Save the weighted matrices and scored data
    save_weighted_matrices(matrix_dataframes_dict, os.path.join(matrix_output_folder, "Unweighted"), save_pickled_matrix_dict)
    save_weighted_matrices(best_weighted_matrices_dict, os.path.join(matrix_output_folder, "Weighted"), save_pickled_matrix_dict)
    with open(os.path.join(matrix_output_folder, "final_weights.txt"), "w") as file:
        file.write(",".join(map(str, best_weights)))

    scored_data_filename = "pairwise_scored_data_thres" + str(best_score_threshold) + ".csv"
    save_dataframe(best_scored_df, output_folder, scored_data_filename)

    print(f"Saved weighted matrices and scored data to {output_folder}")

    return results

'''------------------------------------------------------------------------------------------------------------------
                        Define alternative method for applying predefined, unoptimized weights
   ------------------------------------------------------------------------------------------------------------------'''

def apply_predefined_weights(input_df, position_weights, matrices_dict, slim_length, sequence_col, significance_col,
                             truth_val, score_col, matrix_output_folder, output_folder, make_calls):
    '''
    Function that applies and assesses a given set of weights against matrices and source data

    Args:
        input_df (pd.DataFrame): 	the dataframe containing densitometry values for the peptides being analyzed
        position_weights (list):    list of position weights to use; length must be equal to slim_score
        matrices_dict (dict):       the dictionary of position-type rules --> unweighted matrices
        slim_length (int): 			the length of the motif being studied
        sequence_col (str):         the column in the dataframe that contains peptide sequences
        significance_col (str): 	the column in the dataframe that contains significance calls (Yes/No)
        truth_val (str):            the value to test against input_df[significance_col]
        score_col (str):            the name of the column where motif scores are found
        matrix_output_folder (str): the path to the folder where final matrices should be saved
        output_folder (str): 		the path to the folder where the output data should be saved
        make_calls (bool): 			whether to prompt the user to set a threshold for making positive/negative calls

    Returns:
        output_df (pd.DataFrame): 			the modified dataframe with scores applied
        predictive_value_df (pd.DataFrame): a dataframe containing sensitivity/specificity/PPV/NPV values for different
                                            score thresholds
    '''

    output_df = input_df.copy()

    # Get weights for positions along the motif sequence
    if position_weights is None:
        position_weights = get_position_weights(slim_length)

    # Apply the weights to the matrices
    weighted_matrices_dict = add_matrix_weights(np.array(position_weights), matrices_dict = matrices_dict)

    # Apply the motif scoring algorithm back onto the peptide sequences
    output_df = apply_motif_scores(output_df, weighted_matrices_dict, slim_length, sequence_col, score_col,
                                   add_residue_cols = True, in_place = False)

    # Use thresholding to declare true/false positives/negatives in the peptide sequences
    if make_calls:
        results = apply_threshold(output_df, sig_col = significance_col, score_col = score_col, truth_value = truth_val)
        output_df, selected_threshold, predictive_value_df = results
    else:
        selected_threshold = None
        predictive_value_df = apply_threshold(output_df, sig_col = significance_col, score_col = score_col,
                                              truth_value = truth_val, return_predictive_only = True)

    # Save the weighted matrices and scored data
    save_weighted_matrices(weighted_matrices_dict, matrix_output_folder, save_pickled_dict = True)
    scored_data_filename = "pairwise_scored_data_thres" + str(selected_threshold) + ".csv"
    save_dataframe(output_df, output_folder, scored_data_filename)

    return output_df, weighted_matrices_dict, predictive_value_df

default_general_params = {"percentiles_dict": None,
                          "motif_length": None,
                          "always_allowed_dict": None,
                          "position_weights": None,
                          "output_folder": None,
                          "make_calls": True,
                          "optimize_weights": False,
                          "possible_weights": None,
                          "aa_charac_dict": aa_charac_dict,
                          "convert_phospho": True}

def main(input_df, general_params = None, data_params = None, matrix_params = None, verbose = True):
    '''
    Main function for making pairwise position-weighted matrices

    Args:
        input_df (pd.DataFrame): 	the dataframe containing densitometry values for the peptides being analyzed
        general_params (dict):      dictionary of general parameters:
                                            --> percentiles_dict (dict): input data percentile --> mean signal value
                                            --> motif_length (int): the length of the motif being studied
                                            --> always_allowed_dict (dict): position --> always-permitted residues
                                            --> position_weights (list): predefined weight values, if not optimizing
                                            --> output_folder (str): path where the output data should be saved
                                            --> make_calls (bool): whether to set thresholds and making +/- calls
                                            --> optimize_weights (bool): whether to optimize weights to maximize FDR/FOR
                                            --> possible_weights (list): list of arrays of possible weights at each position
                                            --> aa_charac_dict (dict): dictionary of chemical_characteristic --> [AAs]
                                            --> convert_phospho (bool): whether to convert phospho-residues to unphospho
        data_params (dict):         dictionary of parameters describing the source_dataframe structure, used in matrix-building:
                                            --> bait (str): the bait to use for matrix generation; defaults to best if left blank
                                            --> bait_signal_col_marker (str): keyword that marks columns in source_dataframe that
                                                contain signal values; required only if bait is given
                                            --> best_signal_col (str): column name with best signal values; used if bait is None
                                            --> bait_pass_col (str): column name with pass/fail information
                                            --> pass_str (str): the string representing a pass in bait_pass_col, e.g. "Yes"
                                            --> seq_col (str): column name containing peptide sequences as strings
        matrix_params (dict):       dictionary of parameters that affect matrix-building behaviour, used in matrix-building:
                                            --> thresholds_points_dict (dict): dictionary where threshold_value --> points_value
                                            --> included_residues (list): the residues included for the current type-position rule
                                            --> amino_acids (tuple): the alphabet of amino acids to use when constructing the matrix
                                            --> min_members (int): the minimum number of peptides that must follow the current
                                                type-position rule for the matrix to be built
                                            --> position_for_filtering (int): the position for the type-position rule being assessed
                                            --> clear_filtering_column (bool): whether to set values in the filtering column to zero
        verbose (bool): 			whether to display user feedback and debugging information

    Returns:
        output_df (pd.DataFrame): 			the modified dataframe with scores applied
        best_weights (np.ndarray):          only returned if optimize_weights is set to True; it is the best detected
                                            weights leading to optimal FDR/FOR
        predictive_value_df (pd.DataFrame): only returned if optimize_weights is set to False; it is a dataframe 
                                            containing sensitivity/specificity/PPV/NPV values for different score thres
    '''

    if general_params is None:
        general_params = default_general_params.copy()
    if data_params is None:
        data_params = default_data_params.copy()
    if matrix_params is None:
        matrix_params = default_matrix_params.copy()

    # Declare the output folder for saving pairwise weighted matrices
    output_folder = general_params.get("output_folder")
    if output_folder is None:
        output_folder = os.getcwd()
    matrix_output_folder = os.path.join(output_folder, "Pairwise_Matrices")

    # Obtain the dictionary of matrices that have not yet been weighted
    percentiles_dict = general_params.get("percentiles_dict")
    motif_length = general_params.get("motif_length")
    aa_charac_dict = general_params.get("aa_charac_dict")
    conditional_matrices = ConditionalMatrices(motif_length, input_df, percentiles_dict, aa_charac_dict, data_params, matrix_params)

    # Apply weights to the generated matrices, or find optimal weights
    optimize_weights = general_params.get("optimize_weights")
    sequence_col = data_params.get("seq_col")
    significance_col = data_params.get("bait_pass_col")
    significant_str = data_params.get("pass_str")
    score_col = data_params.get("dest_score_col")
    output_statistics = {}

    if optimize_weights:
        # Find the optimal weights that produce the lowest FDR/FOR pair
        convert_phospho = general_params.get("convert_phospho")
        possible_weights = general_params.get("possible_weights")
        results_tuple = find_optimal_weights(input_df, motif_length, conditional_matrices, sequence_col,
                                             significance_col, significant_str, score_col, matrix_output_folder,
                                             output_folder, possible_weights, aa_charac_dict, convert_phospho,
                                             save_pickled_matrix_dict = True)
        best_fdr, best_for, best_score_threshold, best_weights, weighted_matrices_dict, scored_df = results_tuple
        position_weights = best_weights
        output_statistics["FDR"] = best_fdr
        output_statistics["FOR"] = best_for
        output_statistics["cutoff_threshold"] = best_score_threshold
        output_statistics["position_weights"] = best_weights

    else:
        # Apply predefined weights and calculate predictive values
        position_weights = general_params.get("position_weights")
        make_calls = general_params.get("make_calls")
        results = apply_predefined_weights(input_df, position_weights, matrices_dict, motif_length, sequence_col,
                                           significance_col, significant_str, score_col, matrix_output_folder,
                                           output_folder, make_calls)
        scored_df, weighted_matrices_dict, predictive_value_df = results
        output_statistics["predictive_value_df"] = predictive_value_df

    return (scored_df, position_weights, weighted_matrices_dict, output_statistics)
