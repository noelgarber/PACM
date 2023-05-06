import numpy as np
import pandas as pd
import os
from general_utils.general_utils import list_inputter, dict_value_append
import Matrix_Generator.image_prep.main as preprocess_images
import Matrix_Generator.process_arrays.main_processing as process_arrays

def preprocess_list():
    # Function for preprocessing sets of spot array images
    no_more_sets = False
    df_list = []
    while not no_more_sets:
        df = preprocess_images(verbose = False)
        df_list.append(df)
        print("----------------------------------------------------------------")
        add_another_df = input("Would you like to process another set of images? (Y/N)  ")
        if add_another_df != "Y" and add_another_df != "y":
            no_more_sets = True
    return df_list

def get_bait_cols(df_list):
    '''
    Function to get a dictionary of baits and their corresponding lists of column names holding signal values

    Args:
        df_list (list): list of dataframes

    Returns:
        bait_cols_dict (dict): a dictionary of bait names --> lists of column names holding background-adjusted signals
        bait_calls_cols_dict (dict): a dictionary of bait names --> column names holding significance calls
    '''
    bait_cols_dict = {} # dict containing column names with signal values
    bait_calls_cols_dict = {} # dict containing column names with significance calls
    for df in df_list:
        all_cols = list(df.columns)
        for col in all_cols:
            if "Background-Adjusted_Signal" in col:
                bait_name = col.split("_")[0]
                dict_value_append(bait_cols_dict, bait_name, col)
            elif "call" in col:
                bait_name = col.split("_")[0]
                bait_calls_cols_dict[bait_name] = col

    return bait_cols_dict, bait_calls_cols_dict

def standardize_within_dataset(df_list, control_probe_name):
    standardize_within = input("Standardize within datasets using control peptides? (Y/N)  ")

    output_df_list = []
    percentiles_dict_list = []
    bait_cols_dict, bait_calls_cols_dict = get_bait_cols(df_list=df_list)

    if standardize_within == "Y" or standardize_within == "y":
        controls_list = list_inputter("Enter next control: ")
        for df in df_list:
            output_df, percentiles_dict = process_arrays(data_df = df, controls_list = controls_list, bait_cols_dict = bait_cols_dict,
                                                         bait_pass_cols = bait_calls_cols_dict, control_probe_name = control_probe_name,
                                                         df_standardization = True)
            output_df_list.append(output_df)
            percentiles_dict_list.append(percentiles_dict_list)
    else:
        for df in df_list:
            output_df, percentiles_dict = process_arrays(data_df = df, controls_list = None, bait_cols_dict = bait_cols_dict,
                                                         bait_pass_cols = bait_calls_cols_dict, control_probe_name = control_probe_name,
                                                         df_standardization = False)
            output_df_list.append(output_df)
            percentiles_dict_list.append(percentiles_dict_list)

    return output_df_list, percentiles_dict_list

def standardize_by_control(df_list):
    # Standardize the signal across sets
    standardize = input("Would you like to standardize the data between sets using a common control? (Y/N)  ")
    if standardize = "Y" or standardize = "y":
        control_name = input("\tEnter the name of the control:  ")

        # Define a dictionary for the mean control values in each dataframe
        control_means = {}
        for i, df in enumerate(df_list):
            # Get a list of background-adjusted signal controls
            cols_list = list(df.columns)
            bas_cols_list = []
            for col in cols_list:
                if "Background-Adjusted_Signal" in col:
                    bas_cols_list.append(col)

            # Get mean control value
            df_control_values = []
            for j in np.arange(len(df)):
                if df.at[i, "Peptide_Name"] == control_name:
                    row_control_values = df.loc[j, bas_cols_list].tolist()
                    for k, value in enumerate(row_control_values):
                        if value < 0:
                            # Replace negative values with 0
                            row_control_values[k] = 0
                    df_control_values.extend(row_control_values)
            df_control_values = np.array(df_control_values)
            df_mean_control = df_control_values.mean()
            control_means[i] = df_mean_control

        # Get the mean of mean control values and enforce it across dataframes
        control_means_values = np.array(list(control_means.values()))
        control_super_mean = control_means_values.mean()
        for i, df in enumerate(df_list):
            df_mean_control = control_means.get(i)
            multiplier = control_super_mean / df_mean_control
            for bas_col in bas_cols_list:
                bas_col_loc = df.columns.get_loc(bas_col)
                elements = bas_col.split("Background-Adjusted_Signal")
                bas_std_col = elements[0] + "Background-Adjusted_Standardized_Signal"
                df[bas_std_col] = df[bas_col] * multiplier
                vals = df.pop(bas_std_col)
                df.insert(bas_col_loc+1, bas_std_col, vals)
            df_list[i] = df
    return df_list

# Perform image pre-processing, quantified data processing, standardization, and concatenation
def main_workflow():
    # Preprocess the sets of images
    df_list = preprocess_list()

    # Intra-Group Standardization to enforce consistent controls between baits, along with comparative processing
    control_probe_name = input("For comparative processing, enter the name of the probe control (e.g. \"Secondary-only\"): ")
    output_df_list, percentiles_dict_list = standardize_within_dataset(df_list = df_list, control_probe_name = control_probe_name)

    # Inter-Group Standardization to enforce a consistent shared control between datasets
    standardized_df_list = standardize_by_control(df_list = output_df_list)

    # Concatenate the dataframes together
    concatenated_df = pd.concat(standardized_df_list, axis=0)

    return concatenated_df

# If the script is executed directly, invoke the main workflow and save the dataframe
if __name__ == "__main__":
    df = main_workflow()
    output_path = input("Please enter a path where the concatenated dataframe should be saved: ")
    output_destination = os.path.join(output_path, "concatenated_standardized_df.csv")
    df.to_csv(output_destination)