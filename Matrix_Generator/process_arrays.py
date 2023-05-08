#This script standardizes and makes preliminary comparisons for SPOT peptide array results obtained by densitometry. 

import numpy as np
import pandas as pd
import math
from general_utils.general_utils import input_number

def standardize_dataframe(df, controls_list, bait_cols_dict, probe_control_name = None):
    '''
    Define a function that standardizes dataframes according to a list of controls, across a set of columns

    Args:
        df (pd.DataFrame): the dataframe to standardize
        controls_list (list): list containing the control peptide names
        bait_cols_dict (dict): a dictionary where each key-value pair is a bait name and a list of columns pointing to background-adjusted values for that bait
        probe_control_name (str): the name of the control probe, which will be omitted when standardizing

    Returns:
        standardized_df (pd.DataFrame): a new dataframe that has been standardized according to the controls
    '''
    if probe_control_name is None:
        probe_control_name = input("Please enter the name of the probe control to omit from the standardization algorithm (e.g. Secondary-only): ")

    output_df = df.copy()

    # Make a dict where control_name => corresponding row index in the dataframe
    controls_indices = {}
    for control in controls_list:
        control_indices = output_df.index[output_df["Peptide_Name"]==control].tolist() # Gets the list of indices matching the current control name
        control_index = control_indices[0] # Takes the first instance
        controls_indices[control] = control_index

    # Make a dict where bait_name => mean of controls in all corresponding columns
    bait_controls_means = {}
    for bait, cols in bait_cols_dict.items():
        if bait != probe_control_name:
            bait_controls_values = []
            for control_index in controls_indices.values():
                bait_control_values = [output_df.at[control_index, col] for col in cols]
                bait_controls_values.extend(bait_control_values)
            bait_control_mean = np.array(bait_controls_values).mean()
            bait_controls_means[bait] = bait_control_mean

    # Get mean of controls for all baits, excluding the control bait
    peptide_controls_supermean = np.array(list(bait_controls_means.values())).mean()

    # Standardize dataframe
    for bait, cols in bait_cols_dict.items():
        # Check that the bait is an actual bait and not the control bait
        if bait != probe_control_name:
            # Iterate over the columns belonging to the current bait and standardize each one
            for col in cols:
                # Define the control values for this column
                peptide_control_values = []
                for control_index in controls_indices.values():
                    peptide_control_value = output_df.at[control_index, col]
                    peptide_control_values.append(peptide_control_value)
                peptide_control_values = np.array(peptide_control_values)

                # Find the mean of the control values in the column
                peptide_controls_mean_in_col = peptide_control_values.mean()

                # Calculate the multiplier to standardize, which will result in the control value equalling the control supermean
                multiplier = peptide_controls_supermean / peptide_controls_mean_in_col

                # Apply the standardization using the multiplier
                output_df[col] = output_df[col] * multiplier

    return output_df, peptide_controls_supermean

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
        => at least one of the baits passes the ellipsoid_index test, and
        => at least one of the baits exceeds a threshold defined as the control value * a defined multiplier

    Args:
        input_df (pd.DataFrame): the input dataframe
        bait_pair (tuple): a tuple of (bait1, bait2)
        bait1_signal_cols (list): a list of column names holding bait1 signal values
        bait2_signal_cols (list): a list of column names holding bait2 signal values
        pass_cols (dict): dictionary where bait name --> column name holding significance calls based on ellipsoid_index
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
    log2fc_vals = bait1_signal_means.combine(bait2_signal_means, log2fc)

    # Use a boolean mask that requires that at least 1 bait to pass the ellipsoid_index test, and also that at least 1 bait passes control
    pass_cols_pair = [pass_cols.get(bait_pair[0]), pass_cols.get(bait_pair[1])]
    mask = ((output_df[pass_cols_pair[0]] == "Pass") | (output_df[pass_cols_pair[1]] == "Pass")) & passes_control

    # Apply the log2fc values conditionally using the mask
    output_df.loc[mask, log2fc_col] = log2fc_vals

    return output_df

def one_passes(input_df, bait_cols_dict, bait_pass_cols, control_probe_name, control_multiplier):
    '''
    Function for evaluating whether each hit passes significance tests for at least one bait

    Args:
        input_df (pd.DataFrame): the input dataframe to test
        bait_col_dict (dict): a dictionary where each key-value pair is a bait name and a list of columns pointing to background-adjusted values for that bait
        bait_pass_cols (dict): dictionary where bait name --> column name holding significance calls based on ellipsoid_index
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
    control_signal_means_stacked = np.vstack([control_signal_means] * bait_signal_means_stacked.shape[0])
    passes_control = np.any(bait_signal_means_stacked > control_multiplier * control_signal_means_stacked, axis=0)

    # Use a boolean mask that requires that at least 1 bait to pass the ellipsoid_index test
    pass_conditions = []
    for pass_col in bait_pass_cols.values():
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

def apply_log2fc(data_df, bait_cols_dict, bait_pass_cols, control_probe_name, control_multiplier = None):
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
    if control_multiplier is None:
        control_multiplier = input_number(prompt = "\tEnter a control multiplier for testing if hits are above this multiple (recommended between 2 and 5):  ", mode = "float")

    bait_pairs = get_bait_pairs(list_of_baits = list(bait_cols_dict.keys()))
    for bait_pair in bait_pairs:
        control_signal_cols = bait_cols_dict.get(control_probe_name)
        bait1_signal_cols, bait2_signal_cols = bait_cols_dict.get(bait_pair[0]), bait_cols_dict.get(bait_pair[1])
        data_df = conditional_log2fc(input_df = data_df, bait_pair = bait_pair, control_signal_cols = control_signal_cols,
                                     bait1_signal_cols = bait1_signal_cols, bait2_signal_cols = bait2_signal_cols,
                                     pass_cols = bait_pass_cols, control_multiplier = control_multiplier)
    return data_df, control_multiplier

def main_processing(data_df, controls_list, bait_cols_dict, bait_pass_cols, control_probe_name, control_multiplier = None, df_standardization = True):
    '''
    Main function for processing array data

    Args:
        data_df (pd.DataFrame): the dataframe to standardize
        controls_list (list): list containing the control peptide names
        bait_cols_dict (dict): a dictionary where each key-value pair is a bait name and a list of columns pointing to background-adjusted values for that bait
        bait_pass_cols (dict): dictionary where bait name --> column name holding significance calls based on ellipsoid_index
        control_probe_name (str): the name of the control probe)

    Returns:
        output_df (pd.DataFrame): the intra-dataset standardized dataframe with log2fc and significance columns
        percentiles_dict (dict): optional dict containing calculated signal percentiles from 1st to 99th
    '''
    output_df = data_df.copy()

    # Standardize the input dataframe
    if df_standardization:
        output_df, _ = standardize_dataframe(df = output_df, controls_list = controls_list, bait_cols_dict = bait_cols_dict, probe_control_name = control_probe_name)

    # Calculate log2fc conditionally for each pair of baits, if at least one bait passes the ellipsoid_index test and exceeds the control
    if control_multiplier is None:
        output_df, control_multiplier = apply_log2fc(data_df = output_df, bait_cols_dict = bait_cols_dict,
                                                     bait_pass_cols = bait_pass_cols, control_probe_name = control_probe_name)

    # Check if each hit passes significance for at least one bait
    output_df = one_passes(input_df = output_df, bait_cols_dict = bait_cols_dict, bait_pass_cols = bait_pass_cols,
                           control_probe_name = control_probe_name, control_multiplier = control_multiplier)

    # Create a column containing the maximum value across baits for the mean background-adjusted signal
    output_df, percentiles_dict = find_max_bait_signal(input_df = output_df, bait_cols_dict = bait_cols_dict,
                                                       control_probe_name = control_probe_name, return_percetiles_dict = True)

    return output_df, percentiles_dict