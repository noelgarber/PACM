'''
This script performs image quantification of arrays of spots.
It was developed for SPOT peptide arrays, but it can be used for any image containing a grid of spots with rows and columns.
Note:   images inputted into this algorithm should be pre-straightened and cropped to the borders of the actual
        grid of spots. Leave no surrounding black space.
The data generated by this algorithm is compatible with the rest of the PACM pipeline.
'''

import numpy as np
import pandas as pd
import os
from general_utils.general_utils import input_number, csv_to_dict, dict_value_append
from Matrix_Generator.image_processing.SpotArray import SpotArray
from Matrix_Generator.config import image_params
from tifffile import imwrite

def get_grid_dimensions(verbose = True):
    print("Please enter the dimensions of the array (number of spots in width x number of spots in height).")
    spot_grid_width = input_number(prompt = "Width (number of spots):  ", mode = "int")
    spot_grid_height = input_number(prompt = "Height (number of spots):  ", mode = "int")
    spot_grid_dimensions = (spot_grid_width, spot_grid_height)
    print("-----------------------") if verbose else None
    return spot_grid_dimensions

def load_spot_arrays(filenames_list, image_folder, spot_grid_dimensions, pixel_log_base = 1, ending_coord = None,
                     arbitrary_coords_to_drop = None, buffer_width = 0, verbose = True):
    '''
    Function to load a set of SpotArray objects and return them as a list

    Args:
        filenames_list (list):           list of filenames containing spot array images
        image_folder (str):           the directory where the filenames are stored
        spot_grid_dimensions (tuple):    the tuple of (number of spots wide, number of spots tall)
        pixel_log_base (int):            the base for linearizing the pixel encoding, if a logarithmic encoding was used
        ending_coord (str):              the last alphanumeric coordinate that represents a sample peptide,
                                         if trailing ones are blank
        arbitrary_coords_to_drop (list): if given, it is the list of coords to be dropped from the loaded spot data
        buffer_width (int):              a positive integer used for defining a buffer between inner and outer pixels
                                         for each spot during the background adjustment process
        verbose (bool):                  whether to display separator lines

    Returns:
        spot_arrays (list): a list of SpotArray objects
    '''
    spot_arrays = []
    for filename in filenames_list:
        print("\tLoading", filename) if verbose else None
        file_path = os.path.join(image_folder, filename)

        # Extract metadata from filename
        filename_elements = filename.split("_")
        metadata_tuple = (filename_elements[0], filename_elements[1][4:], filename_elements[2][4:])

        spot_array = SpotArray(tiff_path = file_path, spot_dimensions = spot_grid_dimensions, metadata = metadata_tuple,
                               show_sliced_image = False, show_outlined_image = False, suppress_warnings = False,
                               pixel_log_base = pixel_log_base, ending_coord = ending_coord,
                               arbitrary_coords_to_drop = arbitrary_coords_to_drop, buffer_width = buffer_width,
                               verbose = verbose)
        spot_arrays.append(spot_array)
    print("-----------------------") if verbose else None
    return spot_arrays

# Declare output directories and ensure that they exist
def declare_output_dirs(parent_directory = None):
    '''
    Function for declaring output directories for preprocessed data and image files

    Args:
        parent_directory (str): folder into which subfolders containing output data should be placed

    Returns:
        output_dirs (dict): dictionary of output directories for "output" (quantified data),
                            "rlt_output" (linear grayscale images), and "outlined_output" (outlined images)
    '''

    # If no parent directory was given, declare parent directory as current working directory
    if parent_directory is None:
        # Warning: if this script is run for multiple datasets, data will be overwritten when no parent_directory is given
        parent_directory = os.getcwd()

    # Declare output directories
    output_dirs = {
        "output": parent_directory,
        "rlt_output": os.path.join(parent_directory, "linear_images"),
        "outlined_output": os.path.join(parent_directory, "outlined_spot_images")
    }

    # Check if the directories exist, and if not, make them
    for name, path in output_dirs.items():
        if not os.path.exists(path):
            os.makedirs(path)

    return output_dirs

def assign_data_values(data_df, spot_arrays, multiline_cols = True):
    uas_cols_dict = {} # Dictionary of lists of unadjusted signal column names, where the key is the probe name
    bas_cols_dict = {} # Dictionary of lists of background-adjusted signal column names, where the key is the probe name
    ei_cols_dict = {}  # Dictionary of lists of ellipsoid index column names, where the key is the probe name
    new_cols_dict = {} # Dictionary that includes both of the above, along with the copy and scan numbers, in the form of (copy, scan, bas_col, ei_col)

    for spot_array in spot_arrays:
        if multiline_cols:
            col_prefix = spot_array.probe_name + "\nCopy " + str(spot_array.copy_number) + "\nScan " + str(spot_array.scan_number)
            uas_col = col_prefix + "\nRaw_Spot_Signal"
            bas_col = col_prefix + "\nBackground-Adjusted_Signal"
            ei_col = col_prefix + "\nEllipsoid_Index"
        else:
            col_prefix = spot_array.probe_name + "_Copy-" + str(spot_array.copy_number)
            uas_col = col_prefix + "_Raw_Spot_Signal"
            bas_col = col_prefix + "_Background-Adjusted_Signal"
            ei_col = col_prefix + "_Ellipsoid_Index"

        #Assign column names to dict by probe name
        dict_value_append(uas_cols_dict, spot_array.probe_name, uas_col)
        dict_value_append(bas_cols_dict, spot_array.probe_name, bas_col)
        dict_value_append(ei_cols_dict, spot_array.probe_name, ei_col)
        dict_value_append(new_cols_dict, spot_array.probe_name, (spot_array.copy_number, spot_array.scan_number, uas_col, bas_col, ei_col))

        #Assign dataframe values
        for spot_coord, signal_tuple in spot_array.spot_info_dict.items():
            unadjusted_signal, background_adjusted_signal, ellipsoid_index, _, _ = signal_tuple

            data_df.at[spot_coord, uas_col] = unadjusted_signal
            data_df.at[spot_coord, bas_col] = background_adjusted_signal
            data_df.at[spot_coord, ei_col] = ellipsoid_index

    # Return dicts of column names
    return uas_cols_dict, bas_cols_dict, ei_cols_dict, new_cols_dict

def write_images(parent_output_path, spot_arrays):
    # Simple function to save spot images to output folders

    linear_folder = os.path.join(parent_output_path, "linear_images")
    if not os.path.exists(linear_folder):
        os.makedirs(linear_folder)

    outlined_folder = os.path.join(parent_output_path, "outlined_spot_images")
    if not os.path.exists(outlined_folder):
        os.makedirs(outlined_folder)

    for spot_array in spot_arrays:
        #Save modified image
        linear_directory = os.path.join(linear_folder, "Copy" + str(spot_array.copy_number) + "_" + spot_array.probe_name + "_linear.tif")
        imwrite(linear_directory, spot_array.linear_array)

        outlined_directory = os.path.join(outlined_folder, "Copy" + str(spot_array.copy_number) + "_" + spot_array.probe_name + "_outlined.tif")
        imwrite(outlined_directory, spot_array.outlined_image)

def get_probe_order(probes_list):
    probes_ordered = []
    input_probe_order = input("Would you like to specify the order of probes for sorting columns and/or drop some probes? (Y/N)  ")
    if input_probe_order == "Y":
        print("\tThe probes in this dataset are:", probes_list)
        print("\tEnter the probes in the order you wish them to appear. If a probe is omitted, it is dropped. Hit enter when done.")
        no_more_probes = False
        while not no_more_probes:
            next_probe = input("Probe name:  ")
            if next_probe != "":
                probes_ordered.append(next_probe)
            else:
                no_more_probes = True
    else:
        probes_ordered = probes_list
    return probes_ordered

def prepare_sorted_cols(data_df, probes_ordered, cols_dict, seqs_cols):
    '''
    Function to sort the dataframe by bait (probe) protein

    Args:
        data_df (pd.DataFrame): the dataframe to sort
        probes_ordered (list):  ordered list of probes
        cols_dict (dict):       dict of cols by probe
        seqs_cols (list):       list of column names for peptide seqs

    Returns:

    '''

    sorted_cols = ["Peptide_Name"] # Adds a column to receive peptide names later
    sorted_cols.extend(seqs_cols)

    for current_probe in probes_ordered:
        col_tuples = cols_dict.get(current_probe)
        col_tuples = sorted(col_tuples, key = lambda x: x[0]) #Sorts by copy number
        cols_dict[current_probe] = col_tuples
        for col_tuple in col_tuples:
            sorted_cols.append(col_tuple[2]) # Appends unadjusted signal column name
            sorted_cols.append(col_tuple[3]) # Appends background_adjusted_signal column name
            sorted_cols.append(col_tuple[4]) # Appends ellipsoid_index column name
        data_df.insert(1, current_probe + "_call", "")
        sorted_cols.append(current_probe + "_call")

    # Sort dataframe
    data_df = data_df[sorted_cols]

    return data_df

def call_positives(data_df, bas_cols_dict, ei_cols_dict, image_params = image_params):
    '''
    Function that assigns passes in-place in the dataframe based on the ellipsoid index and signal values

    Args:
        data_df (pd.DataFrame):  input dataframe with quantified spot image data
        bas_cols_dict (dict):    dictionary of probe --> [background-adjusted signal columns]
        ei_cols_dict (dict):     dictionary of probe --> [ellipsoid index columns]
        image_params (dict):     image params as defined in config.py

    Returns:
        None; operation is performed in-place
    '''

    # Get arguments out of image params
    enforce_control_multiple = image_params["enforce_positive_control_multiple"]
    positive_control = image_params["positive_control"]
    positive_control_multiple = image_params["positive_control_multiple"]

    ei_sig_thres = image_params["circle_index_threshold"]

    bait_control_name = image_params["control_probe_name"]
    bait_control_multiplier = image_params["control_multiplier"]

    probes_ordered = image_params["ordered_probe_names"]

    # Get positive control row index if enforcing a multiple of the control
    if enforce_control_multiple and positive_control is None:
        positive_control_index = data_df.index[0]
    elif enforce_control_multiple:
        peptide_names = data_df["Peptide_Name"].to_numpy()
        positive_control_index = np.nanargmax(peptide_names == positive_control)
    else:
        positive_control_index = None

    # Loop through probes and test for whether they pass as positives
    for current_probe in probes_ordered:
        call_col = current_probe + "_call"
        bas_cols = bas_cols_dict[current_probe]
        ei_cols = ei_cols_dict[current_probe]

        if enforce_control_multiple:
            # Test signals against a positive control peptide along with testing ellipsoid index
            positive_control_dict = data_df.iloc[positive_control_index].to_dict()
            positive_control_bas_values = []
            for bas_col in bas_cols:
                positive_control_bas_values.append(positive_control_dict.get(bas_col))
            positive_control_bas_values = np.array(positive_control_bas_values)
            positive_control_bas_mean = positive_control_bas_values.mean()
            minimum_signal = positive_control_multiple * positive_control_bas_mean
            test = lambda x: "Pass" if np.logical_and(np.greater_equal(x[bas_cols].values, minimum_signal).all(),
                                                      np.greater_equal(x[ei_cols].values, ei_sig_thres).all()) else ""
        else:
            # Test only ellipsoid index
            test = lambda x: "Pass" if np.greater_equal(x[ei_cols].values, ei_sig_thres).all() else ""

        data_df[call_col] = data_df.apply(test, axis = 1)

    # Also test whether the signal values are above a defined multiple of the bait control (e.g. Secondary-only)
    bait_control_bas_cols = bas_cols_dict[bait_control_name]
    bait_control_bas_means = data_df[bait_control_bas_cols].to_numpy().mean(axis=1)
    minimum_bas_means = bait_control_bas_means * bait_control_multiplier

    for current_probe in probes_ordered:
        exceeds_control_col = current_probe + "_exceeds_" + str(bait_control_multiplier) + "x_control"
        current_bas_cols = bas_cols_dict[current_probe]

        current_bas_means = data_df[current_bas_cols].to_numpy().mean(axis=1)
        bas_exceeds_control_bait = np.greater(current_bas_means, minimum_bas_means)

        exceeds_control_bait_calls = np.full(shape=len(bas_exceeds_control_bait), fill_value="", dtype="<U4")
        exceeds_control_bait_calls[bas_exceeds_control_bait] = "Yes"

        data_df[exceeds_control_col] = exceeds_control_bait_calls

def add_peptide_names(data_df, names_path = None, include_seqs = False, cols_list = None):
    '''
    Function for adding peptide names and, optionally, their sequences to the dataframe

    Args:
        data_df (pd.DataFrame): the input dataframe with peptide coords
        names_path (str):       the path to the CSV file containing coordinate-->value pairs
        include_seqs (bool):    whether to also assign sequence values
        cols_list (list):       if inlcude_seqs is True, then this is the list of sequence cols starting from column C
    '''

    data_df.insert(0, "Peptide_Name", "")

    if names_path is None:
        names_path = input("\tEnter the path containing the CSV with coordinate-name pairs:  ")

    names_dict = csv_to_dict(names_path, list_mode = include_seqs)
    if include_seqs:
        # Make cols for sequences to reside in
        peptide_col_index = data_df.columns.get_loc("Peptide_Name")
        for i, col in enumerate(cols_list):
            data_df.insert(peptide_col_index + i + 1, col, "")

        # Add sequence elements
        for i, row in data_df.iterrows():
            values_list = names_dict.get(i)
            data_df.at[i, "Peptide_Name"] = values_list[0]
            for j, col in enumerate(cols_list):
                data_df.at[i, col] = values_list[j+1]
    else:
        for i, row in data_df.iterrows():
            pep_name = names_dict.get(i)
            data_df.at[i, "Peptide_Name"] = pep_name

def image_preprocessing(image_folder, output_folder, image_params = image_params, spot_grid_dimensions = None,
                        names_path = None, ending_coord = None, arbitrary_coords_to_drop = None, verbose = True):
    '''
    Main image preprocessing function

    Args:
        image_folder (str):                 directory where TIFF image files are stored
        output_folder (str):                parent directory to output processed images and data to
        image_params (dict):                image_params as described in config.py
        spot_grid_dimensions (tuple):       tuple of (number of spots in width, number of spots in height)
        names_path (str):                   path to CSV containing peptide names
        ending_coord (str):                 last alphanumeric coordinate considered valid in the array
        arbitrary_coords_to_drop (list):    list of arbitrary coordinates to drop/disregard
        verbose (bool):                     whether to display verbose messages

    Returns:
        data_df (pd.DataFrame):             a dataframe containing the quantified spot image data
    '''

    # User prompting for missing arguments
    spot_grid_dimensions = get_grid_dimensions(verbose) if spot_grid_dimensions is None else spot_grid_dimensions
    image_folder = input("Enter the folder containing TIFF images: ") if image_folder is None else image_folder
    filenames_list = os.listdir(image_folder)

    # Load images as SpotArray objects
    print("Loading and processing files as SpotArray objects...") if verbose else None
    buffer_width = image_params["buffer_width"]
    pixel_log_base = image_params["pixel_encoding_base"]
    spot_arrays = load_spot_arrays(filenames_list, image_folder, spot_grid_dimensions, pixel_log_base,
                                   ending_coord, arbitrary_coords_to_drop, buffer_width, verbose)

    # Assemble a dataframe containing results values
    print("Assembling dataframe and saving images...") if verbose else None
    data_df = pd.DataFrame()
    multiline_cols = image_params["multiline_cols"]
    uas_cols_dict, bas_cols_dict, ei_cols_dict, new_cols_dict = assign_data_values(data_df, spot_arrays, multiline_cols)

    # Write output images to destination directories
    write_images(output_folder, spot_arrays)

    # Add peptide names/sequences
    add_names = image_params["add_peptide_names"]
    if add_names:
        print("Applying peptide names to the dataframe...") if verbose else None
        add_peptide_seqs = image_params["add_peptide_seqs"]
        peptide_seq_cols = image_params["peptide_seq_cols"]
        add_peptide_names(data_df, names_path, add_peptide_seqs, peptide_seq_cols)

    # Sorting dataframe and testing significance of hits
    print("Organizing dataframe and testing hit significance...") if verbose else None
    probes_ordered = image_params["ordered_probe_names"]
    seqs_cols = image_params["peptide_seq_cols"]
    data_df = prepare_sorted_cols(data_df, probes_ordered, new_cols_dict, seqs_cols)
    call_positives(data_df, bas_cols_dict, ei_cols_dict, image_params)

    # Save completed dataframe
    data_df.to_csv(os.path.join(output_folder, "preprocessed_data.csv"))

    print("Done!") if verbose else None

    # Return dataframe, which can be optionally assigned when main() is invoked
    return data_df

if __name__ == "__main__":
    main_preprocessing()