import numpy as np
import pandas as pd
import os
from general_utils.general_utils import list_inputter, dict_value_append, input_number
from Matrix_Generator.image_prep import main_preprocessing as preprocess_images
from Matrix_Generator.image_prep import get_grid_dimensions, declare_output_dirs
from Matrix_Generator.general_processing.process_arrays import main_processing as process_arrays
from Matrix_Generator.general_processing.process_arrays import find_max_bait_signal

def get_image_dirs_dims():
    # Declare probe order
    print("For ordering the resulting data from each inputted directory, please enter the probe order; omit probes you wish to drop.")
    probe_order = list_inputter("Next probe: ")

    print("For each directory containing a set of TIFF images, please provide the following information.", "\n---")
    no_more_entries = False
    output_dict = {}
    while not no_more_entries:
        # Declare source image directory and grid dimensions
        image_directory = input("Directory where TIFF images are stored: ")
        if image_directory != "":
            # Get grid dimensions (number of spots in width, number of spots in height)
            grid_dims = get_grid_dimensions(verbose = False)

            # Declare path to peptide names
            peptide_names_path = input("Directory where (coordinates --> peptide names) are stored as a CSV file (hit enter when done): ")

            # Declare output directory
            parent_dir = input("Directory where outlined images for this dataset should be deposited: ")
            output_dirs = declare_output_dirs(parent_directory=parent_dir)

            # Declare significance threshold for ellipsoid index
            ei_sig_thres = input_number("Enter the ellipsoid index significance threshold (e.g. 1.5): ", mode = "float")

            # Get the last valid alphanumeric coordinate for the set
            last_coord = input("Enter the last valid alphanumeric spot coordinate after which data should be dropped (leave blank to skip):  ")
            if last_coord == "":
                last_coord = None

            # Assign to dict
            value_tuple = (grid_dims, output_dirs, peptide_names_path, ei_sig_thres, probe_order, last_coord)
            output_dict[image_directory] = value_tuple

            print("---")
        else:
            print("Done inputting dataset info", "\n---")
            no_more_entries = True

    return output_dict

def preprocess_list(image_directory_dims_dict = None, add_peptide_seqs = False, peptide_seq_cols = None,
                    ending_coord = None, arbitrary_coords_to_drop = None, buffer_width = None):
    '''
    Function to preprocess a list of images

    Args:
        image_directory_dims_dict (dict): dictionary where source_directory -->
                                          (grid_dims, output_dirs, pep_names_path, ei_thres, probe_order, last_coord)

    Returns:
        df_list (list): list of preprocessed dataframes
    '''
    if image_directory_dims_dict is None:
        no_more_sets = False
        df_list = []
        while not no_more_sets:
            df = preprocess_images(multiline_cols = False, add_peptide_seqs = add_peptide_seqs,
                                   peptide_seq_cols = peptide_seq_cols, ending_coord = ending_coord,
                                   arbitrary_coords_to_drop = arbitrary_coords_to_drop, buffer_width = buffer_width,
                                   verbose = False)
            df_list.append(df)
            print("----------------------------------------------------------------")
            add_another_df = input("Would you like to process another set of images? (Y/N)  ")
            if add_another_df != "Y" and add_another_df != "y":
                no_more_sets = True
    else:
        df_list = []
        for i, (image_directory, value_tuple) in enumerate(image_directory_dims_dict.items()):
            print(f"Processing data in {image_directory}")
            grid_dims, output_dirs_dict, names_dir, ei_thres, probe_order, last_coord = value_tuple
            df = preprocess_images(image_directory = image_directory, spot_grid_dimensions = grid_dims,
                                   output_dirs = output_dirs_dict, peptide_names_path = names_dir,
                                   ellipsoid_index_thres = ei_thres, probes_ordered = probe_order,
                                   multiline_cols = False, add_peptide_seqs = add_peptide_seqs,
                                   peptide_seq_cols = peptide_seq_cols, ending_coord = last_coord,
                                   arbitrary_coords_to_drop = arbitrary_coords_to_drop, verbose = False)
            df = df.rename(index = lambda x: str(i+1) + "-" + x)
            df_list.append(df)

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

def standardize_within_dataset(df_list, control_probe_name, control_multiplier = None):
    standardize_within = input("Standardize within datasets using control peptides? (Y/N)  ")

    if control_multiplier is None:
        control_multiplier = input_number(prompt = "\tEnter a control multiplier for testing if hits are above this multiple (recommended between 2 and 5):  ", mode = "float")

    output_df_list = []
    percentiles_dict_list = []
    bait_cols_dict, bait_calls_cols_dict = get_bait_cols(df_list=df_list)

    if standardize_within == "Y" or standardize_within == "y":
        controls_list = list_inputter("\tEnter next control: ")
        for df in df_list:
            output_df, percentiles_dict = process_arrays(data_df = df, controls_list = controls_list, bait_cols_dict = bait_cols_dict,
                                                         bait_pass_cols = bait_calls_cols_dict, control_probe_name = control_probe_name,
                                                         control_multiplier = control_multiplier, df_standardization = True)
            output_df_list.append(output_df)
            percentiles_dict_list.append(percentiles_dict)
    else:
        for df in df_list:
            output_df, percentiles_dict = process_arrays(data_df = df, controls_list = None, bait_cols_dict = bait_cols_dict,
                                                         bait_pass_cols = bait_calls_cols_dict, control_probe_name = control_probe_name,
                                                         control_multiplier = control_multiplier, df_standardization = False)
            output_df_list.append(output_df)
            percentiles_dict_list.append(percentiles_dict)

    return output_df_list, percentiles_dict_list, bait_cols_dict

def standardize_by_control(df_list):
    # Standardize the signal across sets
    standardize = input("Would you like to standardize the data between sets using a common control? (Y/N)  ")
    if standardize == "Y" or standardize == "y":
        control_name = input("\tEnter the name of the control:  ")

        # Define a dictionary for the mean control values in each dataframe
        control_means = {}
        bas_cols_list_dict = {}
        for i, df in enumerate(df_list):
            # Get a list of background-adjusted signal controls
            cols_list = list(df.columns)
            bas_cols_list = []
            for col in cols_list:
                if "Background-Adjusted_Signal" in col:
                    bas_cols_list.append(col)
            bas_cols_list_dict[i] = bas_cols_list

            # Get mean control value
            df_control_values = []
            for j in np.arange(len(df)):
                index_value = str(df.index[j])
                row_at_index = df.loc[index_value].to_dict()
                current_peptide = row_at_index.get("Peptide_Name")
                if current_peptide == control_name:
                    # Get list of control values in row, which is a dict
                    row_control_values = [row_at_index.get(bas_col) for bas_col in bas_cols_list]

                    # Replace negative values with 0
                    for k, value in enumerate(row_control_values):
                        if value < 0:
                            row_control_values[k] = 0

                    # Add this list to the larger control values list
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
            bas_cols_list = bas_cols_list_dict.get(i)
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
def main_workflow(predefined_batch = True, add_peptide_seqs = False, peptide_seq_cols = None, buffer_width = None):
    # Preprocess the sets of images
    if predefined_batch:
        image_dirs_dims_dict = get_image_dirs_dims()
        df_list = preprocess_list(image_directory_dims_dict = image_dirs_dims_dict, add_peptide_seqs = add_peptide_seqs,
                                  peptide_seq_cols = peptide_seq_cols, buffer_width = buffer_width)
    else:
        df_list = preprocess_list(add_peptide_seqs = add_peptide_seqs, peptide_seq_cols = peptide_seq_cols,
                                  buffer_width = buffer_width)

    # Intra-Group Standardization to enforce consistent controls between baits, along with comparative processing
    control_probe_name = input("For comparative processing, enter the name of the probe control (e.g. \"Secondary-only\"): ")
    control_multiplier = input_number(prompt = "\tEnter a control multiplier for testing if hits are above this multiple (recommended between 2 and 5):  ", mode = "float")
    intraset_standardized_df_list, percentiles_dict_list, bait_cols_dict = standardize_within_dataset(df_list = df_list, control_probe_name = control_probe_name,
                                                                                                      control_multiplier = control_multiplier)

    # Inter-Group Standardization to enforce a consistent shared control between datasets
    standardized_df_list = standardize_by_control(df_list = intraset_standardized_df_list)

    # Concatenate the dataframes together
    concatenated_df = pd.concat(standardized_df_list, axis=0)
    
    # Find the percentiles of the concatenated df
    _, concatenated_percentiles_dict = find_max_bait_signal(input_df = concatenated_df, bait_cols_dict = bait_cols_dict,
                                                            control_probe_name = control_probe_name, max_bait_mean_col = "Max_Bait_Standardized_Mean",
                                                            return_percentiles_dict = True)

    return concatenated_df, concatenated_percentiles_dict

# If the script is executed directly, invoke the main workflow and save the dataframe
if __name__ == "__main__":
    df, _ = main_workflow()
    output_path = input("Please enter a path where the concatenated dataframe should be saved: ")
    output_destination = os.path.join(output_path, "concatenated_standardized_df.csv")
    df.to_csv(output_destination)