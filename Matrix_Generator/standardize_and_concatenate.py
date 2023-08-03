import numpy as np
import pandas as pd
import os
from general_utils.general_utils import dict_value_append
from Matrix_Generator.image_prep import image_preprocessing as preprocess_images
from Matrix_Generator.general_processing.process_arrays import main_processing as process_arrays
from Matrix_Generator.general_processing.process_arrays import find_max_bait_signal
try:
    from Matrix_Generator.config_local import image_params
except:
    from Matrix_Generator.config import image_params

def preprocess_list(image_params = image_params, arbitrary_coords_to_drop = None):
    '''
    Function to preprocess a list of images

    Args:
        image_params (dict):             image processing parameters as described in config.py
        arbitrary_coords_to_drop (list): alphanumeric spot coords to ignore

    Returns:
        df_list (list): list of preprocessed dataframes
    '''

    # Get zippable lists of params for sets of images and zip them into an interable object
    input_image_paths = image_params["tiff_paths"]
    input_grid_dims = image_params["grid_dimensions"]
    output_paths = image_params["processed_image_paths"]
    peptide_names_paths = image_params["peptide_names_paths"]
    last_valid_coords = image_params["last_valid_coords"]

    zipped_sets = zip(input_image_paths, input_grid_dims, output_paths, peptide_names_paths, last_valid_coords)

    # Iterate through the zipped image quantification params and construct a list of dataframes, one for each set
    df_list = []
    for i, (input_path, grid_dims, output_path, peptide_names_path, ending_coord) in enumerate(zipped_sets):
        df = preprocess_images(input_path, output_path, image_params, grid_dims, peptide_names_path, ending_coord,
                               arbitrary_coords_to_drop, verbose = False)
        df = df.rename(index = lambda x: str(i + 1) + "-" + x)
        df_list.append(df)

    return df_list

def get_bait_cols(df_list, signal_keyword = "Background-Adjusted_Signal", call_keyword = "call"):
    '''
    Function to get a dictionary of baits and their corresponding lists of column names holding signal values

    Args:
        df_list (list): list of dataframes

    Returns:
        bait_signal_cols (dict): bait names --> lists of column names holding background-adjusted signals
        bait_pass_cols (dict):   bait names --> column names holding significance calls
    '''

    bait_signal_cols = {} # dict containing column names with signal values
    bait_pass_cols = {} # dict containing column names with significance calls
    for df in df_list:
        all_cols = list(df.columns)
        for col in all_cols:
            if signal_keyword in col:
                bait_name = col.split("_")[0]
                dict_value_append(bait_signal_cols, bait_name, col, duplicates_allowed = False)
            elif call_keyword in col:
                bait_name = col.split("_")[0]
                bait_pass_cols[bait_name] = col

    return bait_signal_cols, bait_pass_cols

def standardize_within_dataset(df_list, image_params = image_params):
    '''
    Function to internally standardize each dataframe to defined positive control(s), and also applies bait-bait log2fc
    values and significance calls afterwards

    Args:
        df_list (list):          list of dataframes containing quantified image data
        image_params (dict):     image quantification parameters as defined in config.py

    Returns:
        output_df_list (list):   internally standardized list of dataframes
        bait_signal_cols (dict): dictionary of bait --> [signal column names]
    '''

    standardize_within = image_params["standardize_within_datasets"]
    control_multiplier = image_params["control_multiplier"]
    control_probe_name = image_params["control_probe_name"]

    bait_signal_cols, bait_pass_cols = get_bait_cols(df_list = df_list)

    controls_list = image_params["intra_dataset_controls"]
    max_bait_mean_col = image_params["max_bait_mean_col"]

    output_df_list = []
    for df in df_list:
        output_df = process_arrays(df, controls_list, bait_signal_cols, bait_pass_cols, control_probe_name,
                                   control_multiplier, max_bait_mean_col, standardize_within, return_percentiles=False)
        output_df_list.append(output_df)

    return output_df_list, bait_signal_cols

def standardize_between_datasets(df_list, image_params):
    '''
    Function to standardize dataframes to each other using a defined positive control

    Args:
        df_list (list):          list of dataframes containing quantified image data
        image_params (dict):     image quantification parameters as defined in config.py

    Returns:
        df_list (list):          the inter-dataset standardized dataframes
    '''

    # Standardize the signal across sets
    standardize_between = image_params["standardize_between_datasets"]

    if standardize_between:
        control_name = image_params["inter_dataset_control"]

        # Define a dictionary for the mean control values in each dataframe
        control_means = {}
        bas_cols_list_dict = {} # bas = background-adjusted signal

        for i, df in enumerate(df_list):
            # Get a list of background-adjusted signal columns
            cols_list = list(df.columns)
            bas_cols_list = []
            for col in cols_list:
                if "Background-Adjusted_Signal" in col:
                    bas_cols_list.append(col)
            bas_cols_list_dict[i] = bas_cols_list

            # Get the background-adjusted, intra-dataset standardized control values for all control rows in the df
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

            # Get mean control value
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
def main_workflow(image_params = image_params):
    # Preprocess the sets of images
    df_list = preprocess_list(image_params)

    # Intra-Group Standardization to enforce consistent controls between baits, along with comparative processing
    intraset_standardized_df_list, bait_cols_dict = standardize_within_dataset(df_list, image_params)

    # Inter-Group Standardization to enforce a consistent shared control between datasets
    standardized_df_list = standardize_between_datasets(intraset_standardized_df_list, image_params)

    # Concatenate the dataframes together
    concatenated_df = pd.concat(standardized_df_list, axis=0)

    # Find the percentiles of the concatenated df
    max_bait_mean_col = image_params["max_bait_mean_col"]
    control_probe_name = image_params["control_probe_name"]
    _, concatenated_percentiles_dict = find_max_bait_signal(concatenated_df, bait_cols_dict, control_probe_name,
                                                            max_bait_mean_col, return_percentiles_dict = True)

    return concatenated_df, concatenated_percentiles_dict

# If the script is executed directly, invoke the main workflow and save the dataframe
if __name__ == "__main__":
    df, _ = main_workflow()
    output_path = input("Please enter a path where the concatenated dataframe should be saved: ")
    output_destination = os.path.join(output_path, "concatenated_standardized_df.csv")
    df.to_csv(output_destination)