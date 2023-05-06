#This script standardizes and makes preliminary comparisons for SPOT peptide array results obtained by densitometry. 

import numpy as np
import pandas as pd
import math
from general_utils.general_utils import dict_value_append, list_inputter, input_number

def standardize_dataframe(df, controls_list, bait_col_dict):
    '''
    Define a function that standardizes dataframes according to a list of controls, across a set of columns

    Args:
        df (pd.DataFrame): the dataframe to standardize
        controls_list (list): list containing the control peptide names
        bait_col_dict (dict): a dictionary where each key-value pair is a bait name and a list of columns pointing to background-adjusted values for that bait

    Returns:
        standardized_df (pd.DataFrame): a new dataframe that has been standardized according to the controls
    '''
    output_df = df.copy()

    # Make a dict where control_name => corresponding row index in the dataframe
    controls_indices = {}
    for control in controls_list:
        control_indices = output_df.index[output_df["Peptide_Name"]==control].tolist() # Gets the list of indices matching the current control name
        control_index = control_indices[0] # Takes the first instance
        controls_indices[control] = control_index

    # Make a dict where bait_name => mean of controls in all corresponding columns
    bait_controls_means = {}
    for bait, cols in bait_col_dict.items():
        bait_controls_values = []
        for control_index in controls_indices.values():
            bait_control_values = [output_df.at[control_index, col] for col in cols]
            bait_controls_values.extend(bait_control_values)
        bait_control_mean = np.array(bait_controls_values).mean()
        bait_controls_means[bait] = bait_control_mean

    # Get mean of controls for all baits
    controls_supermean = np.array(list(bait_controls_means.values())).mean()

    # Standardize dataframe
    for bait, cols in bait_col_dict.items():
        for col in cols:
            control_values = []
            for control_index in controls_indices.values():
                control_value = output_df.at[control_index, col]
                control_values.append(control_value)
            control_values = np.array(control_values)
            controls_mean_in_col = control_values.mean()
            multiplier = controls_supermean / controls_mean_in_col
            output_df[col] = output_df[col] * multiplier

    return output_df, controls_supermean

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
    value = math.log2(val1 + increment_avoid_inf) - math.log2(val2 + increment_avoid_inf)
    return value

def conditional_log2fc(input_df, bait_pair, control_signal_cols, bait1_signal_cols, bait2_signal_cols, pass_cols, control_multiplier = 5):
    '''
    Conditionally calculates log2fc based on whether it is interpretable.
    For a log2fc value to be interpretable, the function requires that:
        => at least one of the baits passes the ellipsoid_index test, and
        => at least one of the baits exceeds a threshold defined as the control value * a defined multiplier

    Args:
        input_df (pd.DataFrame): the input dataframe
        bait_pair (tuple): a tuple of (bait1, bait2)
        bait1_signal_cols (list): a list of column names holding bait1 signal values
        bait2_signal_cols (list): a list of column names holding bait2 signal values
        pass_cols (tuple): tuple of (bait1_pass_col, bait2_pass_col) holding the significance calls col for each bait
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

    # Determine if at least one of the baits exceeds the control by the required multiplier for each entry, and whether they pass ellipsoid_index tests
    passes_control = np.logical_or(bait1_signal_means > control_multiplier * control_signal_means,
                                   bait2_signal_means > control_multiplier * control_signal_means)

    # Calculate the log2fc values
    log2fc_value = log2fc(bait1_signal_means, bait2_signal_means)

    # Use a boolean mask that requires that at least 1 bait to pass the ellipsoid_index test, and also that at least 1 bait passes control
    mask = ((output_df[pass_cols[0]] == "Pass") | (output_df[pass_cols[1]] == "Pass")) & passes_control

    # Apply the log2fc values conditionally using the mask
    output_df.loc[mask, log2fc_col] = log2fc_value
    output_df.loc[~mask, log2fc_col] = "NaN"

    return output_df

def one_passes(input_df, bait_cols_dict, bait_pass_cols, control_probe_name, control_multiplier):
    '''
    Function for evaluating whether each hit passes significance tests for at least one bait

    Args:
        input_df (pd.DataFrame): the input dataframe to test
        bait_col_dict (dict): a dictionary where each key-value pair is a bait name and a list of columns pointing to background-adjusted values for that bait
        bait_pass_cols (tuple): n-tuple of (bait1_pass_col, bait2_pass_col, ...) holding the significance calls col for each bait
        control_probe_name (str): the name of the control probe)
        control_multiplier (float): the multiple of the control value that a hit must exceed to be considered significant

    Returns:
        output_df (pd.DataFrame): the dataframe with a "One_Passes" column added
    '''
    output_df = input_df.copy()

    # Get signal means
    bait_signal_means_dict = {}
    control_signal_means = None
    for bait, signal_cols in bait_cols_dict.items():
        signal_means = output_df[signal_cols].mean(axis=1)
        if bait == control_probe_name:
            control_signal_means = signal_means
        else:
            bait_signal_means_dict[bait] = signal_means

    if control_signal_means is None:
        raise Exception(f"one_passes error: the control probe name ({control_probe_name}) was not found in the specified bait_cols_dict, but it is required for signficance testing/comparison.")

    # Determine if at least one of the baits exceeds the control by the required multiplier for each entry, and whether they pass ellipsoid_index tests
    bait_signal_means_list = list(bait_signal_means_dict.values())
    bait_signal_means_stacked = np.vstack(bait_signal_means_list)
    passes_control = np.any(bait_signal_means_stacked > control_multiplier * control_signal_means, axis=0)

    # Use a boolean mask that requires that at least 1 bait to pass the ellipsoid_index test
    pass_conditions = []
    for pass_col in bait_pass_cols:
        pass_conditions.append(output_df[pass_col] == "Pass")
    ellipsoid_index_mask = np.logical_or.reduce(pass_conditions)

    # Combine the two conditions with an 'and' operator
    mask = np.logical_and(passes_control, ellipsoid_index_mask)

    # Apply the log2fc values conditionally using the mask
    output_df.loc[mask, "One_Passes"] = "Yes"
    output_df.loc[~mask, "One_Passes"] = ""

    return output_df

def find_max_bait_signal(input_df, bait_cols_dict, control_probe_name, return_percetiles_dict = True):
    '''
    Function for finding the max bait signal, averaged accross replicates, of any of the baits (excluding control)

    Args:
        input_df (pd.DataFrame): the input dataframe to test
        bait_col_dict (dict): a dictionary where each key-value pair is a bait name and a list of columns pointing to background-adjusted values for that bait
        control_probe_name (str): the name of the control probe)

    Returns:
        output_df (pd.DataFrame): the dataframe with the "Max_Bait_Background-Adjusted_Mean" column added
        percentiles_dict (dict): optional dict containing calculated signal percentiles from 1st to 99th
    '''
    # Make a copy of the input dataframe
    output_df = input_df.copy()

    # Get signal means
    bait_signal_means_dict = {}
    for bait, signal_cols in bait_cols_dict.items():
        signal_means = output_df[signal_cols].mean(axis=1)
        if bait != control_probe_name:
            bait_signal_means_dict[bait] = signal_means

    # Get a series containing the max mean signal at each index
    series_list = list(bait_signal_means_dict.values())
    series_concatenated_df = pd.concat(series_list, axis=1)
    max_series = series_concatenated_df.max(axis=1)

    # Append the max series to the dataframe
    output_df["Max_Bait_Background-Adjusted_Mean"] = max_series

    if return_percetiles_dict:
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

def apply_log2fc(data_df, bait_col_dict, bait_pass_cols, control_probe_name):
    '''
    Function for applying the conditional_log2fc() function to a dataframe

    Args:
         data_df (pd.DataFrame): the dataframe to apply log2fc to
         bait_col_dict (dict): a dictionary where each key-value pair is a bait name and a list of columns pointing to background-adjusted values for that bait
         bait_pass_cols (tuple): tuple of (bait1_pass_col, bait2_pass_col) holding the significance calls col for each bait
         control_probe_name (str): the name of the control probe)

    Returns:
        data_df (pd.DataFrame): a dataframe with the log2fc columns added for comparing each bait-bait pair
    '''
    control_multiplier = input_number(prompt = "Enter a control multiplier for testing if hits are above this multiple (recommended between 2 and 5):  ", mode = "float")
    bait_pairs = get_bait_pairs(list_of_baits = list(bait_col_dict.keys()))
    for bait_pair in bait_pairs:
        control_signal_cols = bait_col_dict.get(control_probe_name)
        bait1_signal_cols, bait2_signal_cols = bait_col_dict.get(bait_pair[0]), bait_col_dict.get(bait_pair[1])
        data_df = conditional_log2fc(input_df = data_df, bait_pair = bait_pair, control_signal_cols = control_signal_cols,
                                     bait1_signal_cols = bait1_signal_cols, bait2_signal_cols = bait2_signal_cols,
                                     pass_cols = bait_pass_cols, control_multiplier = control_multiplier)
    return data_df, control_multiplier

def main_processing(data_df, controls_list, bait_col_dict, bait_pass_cols, control_probe_name):
    '''
    Main function for processing array data

    Args:
        data_df (pd.DataFrame): the dataframe to standardize
        controls_list (list): list containing the control peptide names
        bait_col_dict (dict): a dictionary where each key-value pair is a bait name and a list of columns pointing to background-adjusted values for that bait
        bait_pass_cols (tuple): tuple of (bait1_pass_col, bait2_pass_col) holding the significance calls col for each bait
        control_probe_name (str): the name of the control probe)

    Returns:
        output_df (pd.DataFrame): the intra-dataset standardized dataframe with log2fc and significance columns
        percentiles_dict (dict): optional dict containing calculated signal percentiles from 1st to 99th
    '''
    output_df = data_df.copy()

    # Standardize the input dataframe
    output_df, _ = standardize_dataframe(df = output_df, controls_list = controls_list, bait_col_dict = bait_col_dict)

    # Calculate log2fc conditionally for each pair of baits, if at least one bait passes the ellipsoid_index test and exceeds the control
    output_df, control_multiplier = apply_log2fc(data_df = output_df, bait_col_dict = bait_col_dict,
                                                 bait_pass_cols = bait_pass_cols, control_probe_name = control_probe_name)

    # Check if each hit passes significance for at least one bait
    output_df = one_passes(input_df = output_df, bait_cols_dict = bait_col_dict, bait_pass_cols = bait_pass_cols,
                           control_probe_name = control_probe_name, control_multiplier = control_multiplier)

    # Create a column containing the maximum value across baits for the mean background-adjusted signal
    output_df, percentiles_dict = find_max_bait_signal(input_df = output_df, bait_cols_dict = bait_col_dict,
                                                       control_probe_name = control_probe_name, return_percetiles_dict = True)

    return output_df, percentiles_dict