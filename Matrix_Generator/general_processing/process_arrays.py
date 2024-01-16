#This script standardizes and makes preliminary comparisons for SPOT peptide array results obtained by densitometry. 

import numpy as np
import pandas as pd
import math

def standardize_dataframe(df, controls_list, bait_cols_dict, probe_control_name = None):
    '''
    Function that standardizes dataframes according to a list of controls, across a set of columns

    Args:
        df (pd.DataFrame): the dataframe to standardize
        controls_list (list): list containing the control peptide names
        bait_cols_dict (dict): a dictionary where each key-value pair is a bait name and a list of columns pointing to
                               background-adjusted values for that bait
        probe_control_name (str): the name of the control probe, which will be omitted when standardizing

    Returns:
        output_df (pd.DataFrame):   a new dataframe that has been standardized according to the controls
        total_mean_control (float): the mean of peptide controls across all baits except the probe control
    '''

    if probe_control_name is None:
        probe_control_name = input("Probe control to omit from the standardization algorithm (e.g. Secondary-only): ")

    output_df = df.copy()

    # Get the mini-dataframe containing control rows
    peptide_names_list = output_df["Peptide_Name"].tolist()
    peptide_names_indexer = pd.Index(peptide_names_list)
    control_row_indices = peptide_names_indexer.get_indexer_for(controls_list)

    # Extract the bait cols that do not belong to the probe control, which is excluded from standardization
    bait_cols_list = []
    for bait, cols in bait_cols_dict.items():
        if bait != probe_control_name:
            bait_cols_list.extend(cols)

    # Calculate the mean control values across columns and overall
    control_df = output_df.iloc[control_row_indices]
    control_values = control_df[bait_cols_list].to_numpy()
    mean_control_values = control_values.mean(axis=0)
    total_mean_control = control_values.mean()

    # Loop over the columns to standardize them
    divisors = mean_control_values / total_mean_control
    current_array = output_df[bait_cols_list].to_numpy()
    adjusted_array = current_array / divisors
    output_df[bait_cols_list] = adjusted_array

    return output_df, total_mean_control

def log2fc(val1, val2, increment_avoid_inf = 0.01):
    '''
    Standard function for calculating the log2 fold change between two numbers

    Args:
        val1 (float): first number
        val2 (float): second number
        increment_avoid_inf (float): the number to add to each value to avoid indeterminate values

    Returns:
        value (float): the log2fc value
    '''
    # Handle cases where input values are negative
    if val1 < 0:
        val1 = 0
    if val2 < 0:
        val2 = 0

    value = math.log2(val1 + increment_avoid_inf) - math.log2(val2 + increment_avoid_inf)
    return value

def conditional_log2fc(input_df, bait_pair, control_signal_cols, bait1_signal_cols, bait2_signal_cols, pass_cols, control_multiplier = 5):
    '''
    Conditionally calculates log2fc based on whether it is interpretable.
    For a log2fc value to be interpretable, the function requires that:
        => at least one of the baits passes the call_index test, and
        => at least one of the baits exceeds a threshold defined as the control value * a defined multiplier

    Args:
        input_df (pd.DataFrame): the input dataframe
        bait_pair (tuple): a tuple of (bait1, bait2)
        bait1_signal_cols (list): a list of column names holding bait1 signal values
        bait2_signal_cols (list): a list of column names holding bait2 signal values
        pass_cols (dict): dictionary where bait name --> column name holding significance calls based on call_index
        control_multiplier (float): the multiple of the control value that a hit must exceed to be considered significant

    Returns:
        output_df (pd.DataFrame): a copy of the input dataframe with appended log2fc columns
    '''
    # If the bait pair elements are identical, do not perform log2fc calculation, and return the original dataframe
    if bait_pair[0] == bait_pair[1]:
        return input_df

    # Copy the input dataframe and name the log2fc column
    output_df = input_df.copy()
    log2fc_col = bait_pair[0] + "_" + bait_pair[1] + "_log2fc"

    # Get signal means and ellipsoid index values
    control_signal_means = output_df[control_signal_cols].mean(axis=1)
    bait1_signal_means = output_df[bait1_signal_cols].mean(axis=1)
    bait2_signal_means = output_df[bait2_signal_cols].mean(axis=1)

    # Determine if at least one of the baits exceeds the control by the required multiplier for each entry, and whether they pass call_index tests
    passes_control = np.logical_or(bait1_signal_means > control_multiplier * control_signal_means,
                                   bait2_signal_means > control_multiplier * control_signal_means)

    # Calculate the log2fc values
    log2fc_vals = bait1_signal_means.combine(bait2_signal_means, log2fc)

    # Use a boolean mask that requires that at least 1 bait to pass the call_index test, and also that at least 1 bait passes control
    pass_cols_pair = [pass_cols.get(bait_pair[0]), pass_cols.get(bait_pair[1])]
    mask = ((output_df[pass_cols_pair[0]] == "Pass") | (output_df[pass_cols_pair[1]] == "Pass")) & passes_control

    # Apply the log2fc values conditionally using the mask
    output_df.loc[mask, log2fc_col] = log2fc_vals

    return output_df

def one_passes(input_df, bait_cols_dict, bait_pass_cols, control_probe_name, control_multiplier, calculate_individual_passes = True):
    '''
    Function for evaluating whether each hit passes significance tests for at least one bait

    Args:
        input_df (pd.DataFrame):            the input dataframe to test
        bait_col_dict (dict):               a dictionary where each key-value pair is a bait name and a list of columns pointing to background-adjusted values for that bait
        bait_pass_cols (dict):              dictionary where bait name --> column name holding significance calls based on call_index
        control_probe_name (str):           the name of the control probe)
        control_multiplier (float):         the multiple of the control value that a hit must exceed to be considered significant
        calculate_individual_passes (bool): whether to also calculate individual pass cols for each bait

    Returns:
        output_df (pd.DataFrame): the dataframe with a "One_Passes" column added
    '''
    # Make a copy of the input dataframe to perform operations on
    output_df = input_df.copy()

    # Get signal means as a dictionary where bait_name --> pd.Series of signal mean values
    bait_signal_means_dict = {}
    control_signal_means = None # control bait probe is popped out and handled separately from the dict of baits
    for bait, signal_cols in bait_cols_dict.items():
        signal_means = output_df[signal_cols].mean(axis=1)
        if bait == control_probe_name:
            control_signal_means = signal_means
        else:
            bait_signal_means_dict[bait] = signal_means

    # Check that the control was found in the bait_cols_dict
    if control_signal_means is None:
        raise Exception(f"one_passes error: the control probe name ({control_probe_name}) was not found in the specified bait_cols_dict, but it is required for signficance testing/comparison.")

    # -------------------------------------- Determine & Assign One_Passes Column --------------------------------------

    # Determine if at least one of the baits exceeds the control by the required multiplier for each entry, and whether they pass call_index tests
    bait_signal_means_list = list(bait_signal_means_dict.values())
    bait_signal_means_stacked = np.vstack(bait_signal_means_list)
    control_signal_means_stacked = np.vstack([control_signal_means] * bait_signal_means_stacked.shape[0])
    passes_control = np.any(bait_signal_means_stacked > control_multiplier * control_signal_means_stacked, axis=0)

    # Use a boolean mask that requires that at least 1 bait to pass the call_index test
    control_pass_conditions = []
    bait_pass_conditions = []
    for bait, pass_col in bait_pass_cols.items():
        if bait == control_probe_name:
            control_pass_conditions.append(output_df[pass_col] == "Pass")
        else:
            bait_pass_conditions.append(output_df[pass_col] == "Pass")
    call_index_mask = np.logical_or.reduce(bait_pass_conditions)

    # Combine the two conditions with an 'and' operator
    mask = np.logical_and(passes_control, call_index_mask)

    # Apply the log2fc values conditionally using the mask
    output_df.loc[mask, "One_Passes"] = "Yes"
    output_df.loc[~mask, "One_Passes"] = ""

    # ----------------------------------- Determine & Assign Individual Pass Columns -----------------------------------
    if calculate_individual_passes:
        for bait, bait_signal_means in bait_signal_means_dict.items():
            # Test if bait signal mean values pass the bait probe control by a sufficient margin
            passes_control = np.any(bait_signal_means > control_multiplier * control_signal_means)

            # Test if the bait passes the ellipsoid index test
            bait_pass_col = bait_pass_cols.get(bait)
            call_index_mask = output_df[bait_pass_col] == "Pass"

            # Combine the testing conditions and apply
            mask = np.logical_and(passes_control, call_index_mask)
            sig_col = bait + "_Passes"
            col_idx = output_df.columns.get_loc(bait_pass_col) + 1  # Get the index of the column after bait_pass_col
            output_df.insert(col_idx, sig_col, "")  # Insert the new column after bait_pass_col
            output_df.loc[mask, sig_col] = "Yes"
            output_df.loc[mask, sig_col] = ""

    return output_df

def find_max_bait_signal(input_df, bait_cols_dict, control_probe_name, subtract_control = False, control_multiplier = 1,
                         max_bait_mean_col = "Max_Bait_Background-Adjusted_Mean", return_percentiles_dict = True):
    '''
    Function for finding the max bait signal, averaged accross replicates, of any of the baits (excluding control)

    Args:
        input_df (pd.DataFrame):        the input dataframe to test
        bait_col_dict (dict):           a dictionary where each key-value pair is a bait name and a list of columns pointing
                                        to background-adjusted values for that bait
        control_probe_name (str):       the name of the control probe
        subtract_control (bool):        whether to subtract the control from the max signal values
        control_multiplier (int|float): multiplier for control values before they are used for max signal adjustment
        max_bait_mean_col (str):        the destination column name for assigning max bait signal values
        return_percentiles_dict (bool): whether to return a dict of signal value percentiles

    Returns:
        output_df (pd.DataFrame):       the dataframe with the "Max_Bait_Background-Adjusted_Mean" column added
        percentiles_dict (dict):        optional dict containing calculated signal percentiles from 1st to 99th
    '''

    # Make a copy of the input dataframe
    output_df = input_df.copy()

    # Get signal means
    bait_signal_means_dict = {}
    for bait, signal_cols in bait_cols_dict.items():
        signal_means = output_df[signal_cols].mean(axis=1)
        if bait != control_probe_name:
            bait_signal_means_dict[bait] = signal_means

    # Apply control correction
    if subtract_control:
        control_signal_cols = bait_cols_dict[control_probe_name]
        control_signal_means = output_df[control_signal_cols].mean(axis=1) * control_multiplier
        for bait, signal_means in bait_signal_means_dict.items():
            corrected_means = signal_means - control_signal_means
            corrected_means[corrected_means < 0] = 0
            bait_signal_means_dict[bait] = corrected_means

    # Get a series containing the max mean signal at each index
    series_list = list(bait_signal_means_dict.values())
    series_concatenated_df = pd.concat(series_list, axis=1)
    max_series = series_concatenated_df.max(axis=1)

    # Append the max series to the dataframe
    output_df[max_bait_mean_col] = max_series

    if return_percentiles_dict:
        percentiles_dict = {}
        max_vals = np.array(list(max_series))
        for i in np.arange(1, 100):
            percentiles_dict[i] = np.percentile(max_vals, i)

        return output_df, percentiles_dict

    else:
        return output_df

def get_bait_pairs(list_of_baits):
    '''
    Function to get a permuted list of bait pairs for performing log2fc calculations

    Args:
        list_of_baits (list): list of bait names

    Returns:
        bait_pairs (list): list of tuples of bait pairs
    '''
    bait_pairs = []
    for bait1 in list_of_baits:
        for bait2 in list_of_baits:
            if bait1 != bait2:
                bait_pair = (bait1, bait2)
                bait_pairs.append(bait_pair)

    return bait_pairs

def apply_log2fc(data_df, bait_cols_dict, bait_pass_cols, control_probe_name, control_multiplier):
    '''
    Function for applying the conditional_log2fc() function to a dataframe

    Args:
         data_df (pd.DataFrame):     the dataframe to apply log2fc to
         bait_cols_dict (dict):      a dictionary where each key-value pair is a bait name and a list of columns
                                     pointing to background-adjusted values for that bait
         bait_pass_cols (dict):      a dictionary where bait name --> bait_pass_col, the name of the column holding the
                                     significance calls col for each bait
         control_probe_name (str):   the name of the control probe)
         control_multiplier (float): optional, but recommended when this script will be run on multiple datasets; it is
                                     the multiplier for testing if hits are above this multiple of the control values

    Returns:
        data_df (pd.DataFrame): a dataframe with the log2fc columns added for comparing each bait-bait pair
    '''

    list_of_baits = []
    for key in bait_cols_dict.keys():
        if key != control_probe_name:
            list_of_baits.append(key)

    bait_pairs = get_bait_pairs(list_of_baits = list_of_baits)
    for bait_pair in bait_pairs:
        control_signal_cols = bait_cols_dict.get(control_probe_name)
        bait1_signal_cols, bait2_signal_cols = bait_cols_dict.get(bait_pair[0]), bait_cols_dict.get(bait_pair[1])
        data_df = conditional_log2fc(input_df = data_df, bait_pair = bait_pair, control_signal_cols = control_signal_cols,
                                     bait1_signal_cols = bait1_signal_cols, bait2_signal_cols = bait2_signal_cols,
                                     pass_cols = bait_pass_cols, control_multiplier = control_multiplier)
    return data_df

def main_processing(data_df, controls_list, bait_cols_dict, bait_pass_cols, control_probe_name, control_multiplier,
                    max_bait_mean_col = "Max_Bait_Background-Adjusted_Mean", df_standardization = True,
                    return_percentiles = False):
    '''
    Main function for processing array data

    Args:
        data_df (pd.DataFrame):     the dataframe to standardize
        controls_list (list):       list containing the control peptide names
        bait_cols_dict (dict):      dict where each key-value pair is a bait name and a list of columns pointing
                                    to background-adjusted values for that bait
        bait_pass_cols (dict):      bait name --> column name holding significance calls based on call_index
        control_probe_name (str):   the name of the control probe)
        control_multiplier (float): multiplier of control values for significance testing
        max_bait_mean_col (str):    the destination column name for assigning max bait signal values
        df_standardization (bool):  whether to standardize the dataframe to control(s)
        return_percentiles (bool):  whether to return a dict of signal value percentiles

    Returns:
        output_df (pd.DataFrame): the intra-dataset standardized dataframe with log2fc and significance columns
        percentiles_dict (dict): optional dict containing calculated signal percentiles from 1st to 99th
    '''

    output_df = data_df.copy()

    # Standardize the input dataframe
    if df_standardization:
        output_df, _ = standardize_dataframe(output_df, controls_list, bait_cols_dict, control_probe_name)

    # Calculate log2fc conditionally for each bait pair, if >1 bait passes the call_index test and exceeds control
    output_df = apply_log2fc(output_df, bait_cols_dict, bait_pass_cols, control_probe_name, control_multiplier)

    # Check if each hit passes significance for at least one bait
    output_df = one_passes(output_df, bait_cols_dict, bait_pass_cols, control_probe_name, control_multiplier)

    # Create a column containing the maximum value across baits for the mean background-adjusted signal
    output_df, percentiles_dict = find_max_bait_signal(output_df, bait_cols_dict, control_probe_name, max_bait_mean_col,
                                                       return_percentiles_dict = True)

    if return_percentiles:
        return output_df, percentiles_dict
    else:
        return output_df