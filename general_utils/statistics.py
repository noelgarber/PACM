# This script contains functions used for descriptive and inferential statistical assessment of matrices

import numpy as np
import pandas as pd
from functools import partial

def diagnostic_value(score_threshold, dataframe, passes_col = "One_Passes", score_col = "SLiM_Score",
                     truth_value = "Yes", passes_bools = None, scores_array = None,
                     return_dict = True, return_fdr_for = False):
    '''
    Function to produce sensitivity, specificity, and positive and negative predictive values for a given score cutoff

    Args:
        score_threshold (float):   the score cutoff to use
        dataframe (pd.DataFrame):  the dataframe containing peptide information
        passes_col (str):          the column containing significance information
        score_col (str):           the column containing the back-calculated motif score for the peptides
        truth_value (str):         the truth value found in significance_col; by default, it is the string "Yes"
        passes_bools (np.ndarray): significance info as bools; can be passed instead of passes_col/truth_value for 
                                   performance improvement
        scores_array (np.ndarray): peptide scores as a numpy array, allowing the function to avoid dataframe[score_col]

    Returns:
        pred_val_dict (dict):     if return_dict, returns a dict of TP, FP, TN, FN, Sensitivity, Specificity, PPV, NPV, FDR, FOR
                                  --> divide_by_zero occurrences result in np.nan
    '''

    # Calculate the boolean arrays for conditions
    if scores_array is None:
        score_above_thres = dataframe[score_col].values >= score_threshold
    else:
        score_above_thres = scores_array >= score_threshold

    if passes_bools is None:
        passes_bools = dataframe[passes_col].values == truth_value

    # Count the occurrences of each call type
    TP_count = np.sum(score_above_thres & passes_bools)
    FP_count = np.sum(score_above_thres & ~passes_bools)
    FN_count = np.sum(~score_above_thres & passes_bools)
    TN_count = np.sum(~score_above_thres & ~passes_bools)

    # Calculate Sensitivity, Specificity, PPV, NPV, FDR, and FOR
    ppv = TP_count/(TP_count+FP_count) if (TP_count+FP_count) > 0 else 0
    npv = TN_count/(TN_count+FN_count) if (TN_count+FN_count) > 0 else 0
    false_discovery_rate = 1 - ppv
    false_omission_rate = 1 - npv
    if return_fdr_for: 
        return (false_discovery_rate, false_omission_rate)

    sensitivity = TP_count / (TP_count + FN_count) if (TP_count + FN_count) > 0 else 0
    specificity = TN_count / (TN_count + FP_count) if (TN_count + FP_count) > 0 else 0

    if return_dict:
        pred_val_dict = {"TP": TP_count,
                         "FP": FP_count,
                         "TN": TN_count,
                         "FN": FN_count,
                         "Sensitivity": sensitivity,
                         "Specificity": specificity,
                         "PPV": ppv,
                         "NPV": npv,
                         "FDR": false_discovery_rate,
                         "FOR": false_omission_rate}
        return pred_val_dict
    else:
        return sensitivity, specificity, ppv, npv, false_discovery_rate, false_omission_rate

def apply_threshold(input_df, score_range_series = None, sig_col = "One_Passes", score_col = "SLiM_Score", 
                    truth_value = "Yes", passes_bools = None, scores_array = None, range_count = 100,
                    return_pred_vals_only = False, return_optimized_fdr = False, verbose = False):
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
        return_pred_vals_only (bool):    whether to only return predictive values without setting and applying a threshold
        return_optimized_fdr (bool):     whether to only return best_score, best_fdr, and best_for
        verbose (bool):                  whether to display debugging information

    Returns:
        output_df (pd.DataFrame):              dens_df with a new column containing calls based on the selected score
        selected_threshold (float):            the selected score threshold
    '''

    # Make an array of bools for where significance column is equal to the truth value
    if passes_bools is None:
        passes_bools = input_df[sig_col]==truth_value
    if scores_array is None:
        scores_array = input_df[score_col].to_numpy()

    # Make a range of SLiM scores between the minimum and maximum score values from the dataframe
    if score_range_series is None:
        min_score = input_df[score_col].min()
        max_score = input_df[score_col].max()
        print(f"max_score = {max_score} | min_score = {min_score}") if verbose else None
        score_range_series = np.linspace(min_score, max_score, num = range_count)

    if return_optimized_fdr:
        # Find the row where the FDR/FOR ratio is closest to 1, and use that for the FDR
        diagnostic_value_partial = partial(diagnostic_value, dataframe = None, passes_col = None,
                                           score_col = None, truth_value = None, passes_bools = passes_bools,
                                           scores_array = scores_array, return_dict = False, return_fdr_for = True)
        fdr_for_pairs = [diagnostic_value_partial(score) for score in score_range_series]
        fdr_for_array = np.array(fdr_for_pairs)
        motif_scores = score_range_series
        with np.errstate(divide="ignore"):
            # Ignores RuntimeWarnings where divide-by-zero occurs
            ratios = np.divide(fdr_for_array[:,0], fdr_for_array[:,1], out=np.full_like(fdr_for_array[:,0], np.inf), where=(fdr_for_array[:,1] != 0))
        closest_index = np.argmin(np.abs(ratios - 1))

        best_fdr = fdr_for_array[closest_index, 0]
        best_for = fdr_for_array[closest_index, 1]
        best_score = motif_scores[closest_index]

        return best_score, best_fdr, best_for

    # Calculate positive and negative predictive values (PPV/NPV), and false discovery and omission rates (FDR/FOR)
    ppvs, npvs, fdrs, fors = [], [], [], []

    passes_bools = input_df == truth_value
    for current_score in score_range_series:
        _, _, ppv, npv, fdr_val, for_val = diagnostic_value(score_threshold = current_score, dataframe = input_df,
                                            passes_bools = passes_bools, return_fdr_for = False, return_dict = False)
        ppvs.append(ppv)
        npvs.append(npv)
        fdrs.append(fdr_val)
        fors.append(for_val)

    # Assemble a dataframe of score --> [positive predictive value, negative predictive value]
    predictive_value_df = pd.DataFrame(columns=["Score", "PPV", "NPV", "FDR", "FOR"])
    predictive_value_df["Score"] = score_range_series
    predictive_value_df["PPV"] = ppvs
    predictive_value_df["NPV"] = npvs
    predictive_value_df["FDR"] = fdrs
    predictive_value_df["FOR"] = fors

    if return_pred_vals_only:
        return predictive_value_df

    # Print the dataframe to aid in the user selecting an appropriate score cutoff for significance to be declared
    print("Threshold selection information:",)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(predictive_value_df)
    print("---")

    # Prompt the user to input the selected threshold that a SLiM score must exceed to be considered significant
    selected_threshold = input_number("Input your selected threshold for calling hits:  ", "float")

    # Apply the selected threshold to the dataframe to call hits
    output_df = input_df.copy()
    for i in np.arange(len(output_df)):
        # Get score and pass info for current row
        current_score = output_df.at[i, "SLiM_Score"]
        spot_passes = output_df.at[i, sig_col]

        # Make calls on true/false positives/negatives
        if current_score >= selected_threshold:
            output_df.at[i, "Call"] = "Positive"
            if spot_passes == "Yes":
                call_type = "TP"
            else:
                call_type = "FP"
        else:
            output_df.at[i, "Call"] = "-"
            if spot_passes == "Yes":
                call_type = "FN"
            else:
                call_type = "TN"

        # Apply the call to the dataframe at the specified row
        output_df.at[i, "Call_Type"] = call_type

    print("Applied hit calls based on threshold.")

    return output_df, selected_threshold, predictive_value_df