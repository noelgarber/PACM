# This script contains functions used for descriptive and inferential statistical assessment of matrices

import numpy as np
import pandas as pd
from general_utils.general_utils import print_whole_df

def get_score_ranges(input_df, score_col, range_count = 100, verbose = False):
    # Helper function that gets the range of scores between the minimum and maximum observed in the dataframe

    min_score = input_df[score_col].min()
    max_score = input_df[score_col].max()
    print(f"max_score = {max_score} | min_score = {min_score}") if verbose else None
    score_range_series = np.linspace(min_score, max_score, num = range_count)

    return score_range_series

def get_rates(scores_array, passes_array, score_range_series, return_comprehensive = False, include_sens = False):
    # Vectorized helper function that gets an array with 2 columns, FDR and FOR, for each score in score_range_series

    above_thres_2d = scores_array >= score_range_series[:, np.newaxis]
    below_thres_2d = ~above_thres_2d
    passes_array_2d = np.repeat(passes_array[np.newaxis,:], len(score_range_series), axis=0)

    TP_counts = np.sum(above_thres_2d & passes_array_2d, axis=1)
    positive_calls_counts = above_thres_2d.sum(axis=1)

    fails_array = ~passes_array_2d
    TN_counts = np.sum(below_thres_2d & fails_array, axis=1)
    negative_calls_counts = below_thres_2d.sum(axis=1)

    with np.errstate(divide = "ignore", invalid = "ignore"):
        ppv_values = TP_counts / positive_calls_counts
        ppv_values[ppv_values==np.inf] = np.nan
        npv_values = TN_counts / negative_calls_counts
        npv_values[npv_values==np.inf] = np.nan

    fdr_values = 1 - ppv_values
    for_values = 1 - npv_values

    # Make an array where each row is [ppv_val, npv_val]
    if not return_comprehensive:
        rates_arr = np.stack([fdr_values, for_values], axis=1)
        return rates_arr

    # Create the comprehensive dataframe
    df = pd.DataFrame()
    df["Scores"] = score_range_series

    if include_sens:
        actual_truths_count = np.sum(passes_array) # does not vary with score threshold, therefore only 1 value
        actual_falses_count = np.sum(~passes_array)
        df["Sensitivity"] = TP_counts / actual_truths_count
        df["Specificity"] = TN_counts / actual_falses_count

    df["PPV"] = ppv_values
    df["NPV"] = npv_values
    df["FDR"] = fdr_values
    df["FOR"] = for_values

    return df

def optimize_threshold_fdr(input_df, score_range_series = None, sig_col = "One_Passes", score_col = "SLiM_Score", 
                           truth_value = "Yes", passes_bools = None, scores_array = None, range_count = 100,
                           verbose = False):
    '''
    Function to declare and apply the motif score threshold based on predictive values

    Args:
        input_df (pd.DataFrame):         the dataframe containing peptides and scores; must contain score values
        score_range_series (np.ndarray): if function is used in a loop, providing this upfront improves performance
        sig_col (str): 			         the df column containing significance information (Yes/No)
        score_col (str): 		         the df column containing the peptide scores
        truth_value (str):               the truth value found in significance_col; by default, it is the string "Yes"
        passes_bools (array-like):       the significance info as bools; if not given, it is auto-generated
        scores_array (np.ndarray):       peptide scores as a numpy array, avoiding the need to look up in dataframe
        range_count (int):               number of score values to test; default is 100
        verbose (bool):                  whether to display debugging information

    Returns:
        best_score (float):              the score threshold where FDR/FOR ratio is optimal
        best_fdr (float):                optimal false discovery rate
        best_for (float):                optimal false omission rate
    '''

    # Get required boolean arrays if not provided upfront
    passes_bools = input_df[sig_col].values == truth_value if passes_bools is None else passes_bools
    scores_array = input_df[score_col].to_numpy() if scores_array is None else scores_array

    # Make a range of SLiM scores between the minimum and maximum score values from the dataframe
    if score_range_series is None:
        score_range_series = get_score_ranges(input_df, score_col, range_count, verbose)

    # Find the row where the absolute difference between FDR and FOR is closest to 0, and use that for the FDR
    fdr_for_array = get_rates(scores_array, passes_bools, score_range_series, return_comprehensive = False)

    min_rate_vals = fdr_for_array.min(axis=1) # real values range from 0.0 to 1.0
    max_rate_vals = fdr_for_array.max(axis=1) # real values range from 0.0 to 1.0
    deltas = max_rate_vals - min_rate_vals
    deltas_clean = np.nan_to_num(deltas, nan=np.inf)

    closest_index = np.argmin(deltas_clean)

    best_fdr = fdr_for_array[closest_index, 0]
    best_for = fdr_for_array[closest_index, 1]
    best_score = score_range_series[closest_index]

    if verbose:
        print(f"\toptimize_threshold_fdr() found best score threshold ({best_score}) yielded FDR={best_fdr} and FOR={best_for}")

    return (best_score, best_fdr, best_for)

def apply_threshold(input_df, score_range_series = None, sig_col = "One_Passes", score_col = "SLiM_Score", 
                    truth_value = "Yes", range_count = 100, return_predictive_only = False, verbose = False):
    '''
    Function to declare and apply the motif score threshold based on predictive values

    Args:
        input_df (pd.DataFrame):         the dataframe containing peptides and scores; must contain score values
        score_range_series (np.ndarray): if function is used in a loop, providing this upfront improves performance
        sig_col (str): 			         the df column containing significance information (Yes/No)
        score_col (str): 		         the df column containing the peptide scores
        truth_value (str):               the truth value found in significance_col; by default, it is the string "Yes"
        range_count (int):               number of score values to test; default is 100
        return_predictive_only (bool):   whether to only return predictive_value_df
        verbose (bool):                  whether to display debugging information

    Returns:
        output_df (pd.DataFrame):           dens_df with a new column containing calls based on the selected score
        selected_threshold (float):         the selected score threshold
        predictive_value_df (pd.DataFrame): PPV/NPV/FDR/FOR values for different score thresholds
    '''

    # Make a range of SLiM scores between the minimum and maximum score values from the dataframe
    if score_range_series is None:
        score_range_series = get_score_ranges(input_df, score_col, range_count, verbose)

    # Get the predictive value dataframe with PPVs, NPVs, FDRs, and FORs, for various score thresholds
    passes_bools = input_df[sig_col]==truth_value
    scores_array = input_df[score_col].to_numpy()
    predictive_value_df = get_rates(scores_array, passes_bools, score_range_series, return_comprehensive = True)
    if return_predictive_only:
        return predictive_value_df

    print_whole_df(preceding_title = "Threshold selection information:", df = predictive_value_df)

    # Prompt the user to input the selected threshold that a SLiM score must exceed to be considered significant
    selected_threshold = input_number("Input your selected threshold for calling hits:  ", "float")

    # Apply the selected threshold to the dataframe to call hits
    output_df = input_df.copy()
    scores = output_df[score_col].to_numpy()

    above_thres_bools = scores >= selected_threshold
    calls = np.full(shape=len(scores), fill_value="-", dtype="<U10")
    calls[above_thres_bools] = "Positive"
    output_df["Call"] = calls

    passes_bools = output_df[sig_col].values == truth_value
    call_types = np.full(shape=len(scores), fill_value="-", dtype="<U2")
    call_types[above_thres_bools & passes_bools] = "TP"
    call_types[above_thres_bools & ~passes_bools] = "FP"
    call_types[~above_thres_bools & passes_bools] = "FN"
    call_types[~above_thres_bools & ~passes_bools] = "TN"
    output_df["Call_Type"] = call_types

    print("Applied hit calls based on threshold.")

    return (output_df, selected_threshold, predictive_value_df)