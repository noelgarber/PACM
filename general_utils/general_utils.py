import numpy as np
import pandas as pd
import csv
import os
import pickle

def input_number(prompt, mode = "float"):
    '''
    Similar to the built-in input() function, but returns a number instead of a string.
    Prompts the user again if an invalid input is given.

    Args:
        prompt (str): user prompt for inputting the number; same usage as input()
        mode (str): type of number to return; must be "float" or "int"

    Returns:
        output_value: a float or int depending on the mode, based on what the user inputs
    '''
    if mode not in ["int", "float"]:
        raise Exception(f"input_number error: mode was set to {mode}, but either \"int\" or \"float\" was expected")

    input_finished = False
    output_value = None

    while not input_finished:
        input_value = input(prompt)
        if mode == "int":
            try:
                output_value = int(input_value)
                input_finished = True
            except:
                print("\tinput value was not an integer; please try again.")
        elif mode == "float":
            try:
                output_value = float(input_value)
                input_finished = True
            except:
                print("\tinput value was not a float; please try again.")

    return output_value

def csv_to_dict(filepath, list_mode = False):
    '''
    Simple function for converting a 2-column CSV file into a dictionary.

    Args:
        filepath (str): the path to the CSV file, which must contain 2 columns without titles; keys go in first column
        list_mode (bool): whether to read additional cols as a list, e.g. col1 --> [col2, col3, col4, ...]

    Returns:
        result (dict): a dictionary of keys (from the first column) matching values (from the second column)
    '''
    result = {}
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            key = row[0]
            if list_mode:
                value = row[1:]
            else:
                value = row[1]
            result[key] = value
    return result

def dict_value_append(input_dict, key, element_to_append):
    '''
    Simple function for appending elements to lists held as dictionary values for a given key in an existing dictionary.

    Args:
        input_dict (dict): dict of (key --> value_list) pairs, where an element will be appended to value_list for key
        key: the key to use for accessing value_list
        element_to_append: the value to append to the value_list associated with key

    Returns:
        None; the operation is performed in-place
    '''
    if input_dict.get(key) == None:
        input_dict[key] = [element_to_append]
    else:
        value_list = input_dict.get(key)
        value_list.append(element_to_append)
        input_dict[key] = value_list

def list_inputter(item_prompt):
    '''
    Simple function to sequentially input elements into a list

    Args:
        item_prompt (str): the prompt to display for adding each list item

    Returns:
        output_list (list): a list containing all the inputted values
    '''
    no_more_items = False
    output_list = []
    while not no_more_items:
        item = input(item_prompt)
        if item != "":
            output_list.append(item)
        else:
            no_more_items = True
    return output_list

def get_delimited_list(prompt, enforced_list_length = None, delimiter = ",", dtype = float):
    '''
    Simple function to get a list as a delimited string and prompt the user if it is not correct

    Args:
        prompt (str):               the prompt to display to the user
        enforced_list_length (int): if given, this list length is enforced by reprompting the user if incorrect length
        delimiter (str):            the delimiter for the list, e.g. ","
        dtype (type):               the dtype to enforce on the list

    Returns:
        delimited_list (list):      the delimited list
    '''

    valid_list_given = False
    while not valid_list_given:
        delimited_list_str = input(prompt)
        delimited_list = delimited_list_str.split(delimiter)
        try:
            delimited_list = [dtype(x) for x in delimited_list]
            if not enforced_list_length:
                valid_list_given = True
            elif len(delimited_list) == enforced_list_length:
                valid_list_given = True
            else:
                print(f"Incorrect list length (expected {enforced_list_length}, got {len(delimited_list)}); please try again.")
        except:
            print(f"Error casting delimited list to dtype {dtype}; please try again.")

    return delimited_list

def mean_at_index(df, row_index, col_names):
    '''
    Simple function to find the mean of values in a dataframe row across a specified set of columns by name

    Args:
        df (pd.DataFrame):  the dataframe to use for lookup
        row_index (int):    the index of the row to find the mean at
        col_names (list):   list of column names for calculating a mean value

    Returns:
        mean_value (float): the mean value at row_index in df for cols defined in col_names
    '''
    values = np.empty(0)
    for col in col_names:
        values = np.append(values, df.at[row_index, col])
    mean_value = values.mean()
    return mean_value


def save_dict(dictionary, output_path, filename):
    # Basic function to save key-value pairs as lines in a text file

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_path = os.path.join(output_path, filename)

    with open(file_path, 'w') as file:
        for key, value in dictionary.items():
            file.write(f'{key}: {value}\n')

def save_dataframe(dataframe, output_directory, output_filename):
    '''
    Simple function to save a dataframe to a destination folder under a specific name

    Args:
        dataframe (pd.DataFrame): 	the dataframe to save
        output_directory (str): 	the output directory
        output_filename (str):     the name of the file

    Returns:
        output_file_path (str): 	the output path where the file was saved
    '''

    if ".csv" not in output_filename:
        output_filename = output_filename + ".csv"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_file_path = os.path.join(output_directory, output_filename)
    dataframe.to_csv(output_file_path)

    return output_file_path

def save_weighted_matrices(weighted_matrices_dict, matrix_directory = None, save_pickled_dict = True):
    '''
    Simple function to save the weighted matrices to disk

    Args:
        weighted_matrices_dict (dict): the dictionary of type-position rule --> corresponding weighted matrix
        matrix_directory (str): directory to save matrices into; defaults to a subfolder called Pairwise_Matrices
    '''
    if matrix_directory is None:
        matrix_directory = os.path.join(os.getcwd(), "Pairwise_Matrices")

    # If the matrix directory does not exist, make it
    if not os.path.exists(matrix_directory):
        os.makedirs(matrix_directory)

    # Save matrices by key name as CSV files
    for key, df in weighted_matrices_dict.items():
        df.to_csv(os.path.join(matrix_directory, key + ".csv"))

    if save_pickled_dict:
        pickled_dict_path = os.path.join(matrix_directory, "weighted_matrices_dict.pkl")
        with open(pickled_dict_path, "wb") as f:
            pickle.dump(weighted_matrices_dict, f)
        print(f"Saved {len(weighted_matrices_dict)} matrices and pickled weighted_matrices_dict to {matrix_directory}")
    else:
        print(f"Saved {len(weighted_matrices_dict)} matrices to {matrix_directory}")

def get_log2fc_cols(comparator_set_1, comparator_set_2):
    # Define the columns containing log2fc values
    log2fc_cols = []
    for bait1 in comparator_set_1:
        for bait2 in comparator_set_2:
            if bait1 != bait2:
                log2fc_cols.append(bait1 + "_" + bait2 + "_log2fc")

    return log2fc_cols

def least_different(input_df, comparator_set_1 = None, comparator_set_2 = None, log2fc_cols = None, return_df = False,
                    return_array = True, in_place = True):
    '''
    Simple function to determine the least different log2fc between permutations of the sets of comparators

    Args:
        input_df (pd.DataFrame):   the dataframe to operate on
        comparator_set_1 (list):   the first set of comparator baits
        comparator_set_2 (list):   the second set of comparator baits
        log2fc_cols (list):        list of columns; if not given, it will be auto-generated
        return_df (bool):          whether to return the modified dataframe
        return_array (bool):       whether to return the least different value array and least different baits array
        in_place (bool):           whether to add "least_different_log2fc" and "least_different_baits" cols in-place

    Returns:
        least_different_values (np.ndarray): log2fc values for the least different pair of baits
        least_different_baits (np.ndarray):  tuples of (bait1, bait2) corresponding to least_diff_log2fc
        output_df (pd.DataFrame):            returned instead of the series if return_df is True; modified df with series
    '''

    if comparator_set_1 is None or comparator_set_2 is None:
        comparator_set_1, comparator_set_2 = get_comparator_baits()

    # Define the columns containing log2fc values
    if log2fc_cols is None:
        log2fc_cols = get_log2fc_cols(comparator_set_1, comparator_set_2)

    # Get least different values and bait pairs
    log2fc_array = input_df[log2fc_cols].to_numpy()
    log2fc_array[np.isnan(log2fc_array)] = np.inf
    least_different_indices = np.nanargmin(log2fc_array, axis=1)
    least_different_values = np.min(log2fc_array, axis=1)
    least_different_cols = np.array(log2fc_cols)[least_different_indices]
    least_different_baits = [col.rsplit("_", 1)[0] for col in least_different_cols]

    # Assign columns if specified
    if in_place or return_df:
        output_df = input_df if in_place else input_df.copy()
        output_df["least_different_log2fc"] = least_different_values
        output_df["least_different_baits"] = least_different_baits

    # Return appropriate values
    if return_array and not return_df:
        return least_different_values, least_different_baits
    elif return_array and return_df:
        return least_different_values, least_different_baits, output_df
    elif return_df and not return_series:
        return output_df


def print_whole_df(df, preceding_title = None):
    # Simple helper function for displaying the score threshold selection dataframe

    print("---")
    print(preceding_title) if preceding_title else None
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(df)
    print("---")

def unravel_seqs(sequences_array, motif_length, convert_phospho = True):
    # Simple function to convert an array of sequences of equal length to a 2D array of amino acid residues

    # Check that all the sequences are an equal, correct length, and then unravel them into a 2D array of residues
    sequence_lengths = np.vectorize(len)(sequences_array)
    matches_slim_length = sequence_lengths == motif_length
    if not matches_slim_length.all():
        wrong_lengths_count = np.sum(~matches_slim_length)
        raise ValueError(f"score_seqs error: {wrong_lengths_count} seqs (of {len(sequences)}) are not the expected length ({slim_length})")

    sequences_array = sequences_array.astype("<U")
    sequences_unravelled = sequences_array.view("U1")
    sequences_2d = np.reshape(sequences_unravelled, (-1, motif_length))
    if convert_phospho:
        sequences_2d[sequences_2d=="B"] = "S"
        sequences_2d[sequences_2d=="J"] = "T"
        sequences_2d[sequences_2d=="O"] = "Y"

    return sequences_2d

def check_seq_lengths(sequences, expected_length, raise_error = True):
    # Helper function to check that a pandas series or array of peptide sequences is all the same length
    
    if isinstance(sequences, np.ndarray): 
        sequences = pd.Series(sequences)

    sequence_lengths = sequences.str.len()
    unique_lengths_count = sequence_lengths.nunique()
    first_length_matches = sequence_lengths[0] == expected_length
    
    correct_uniform_length = True
    if unique_lengths_count > 1:
        if raise_error: 
            raise ValueError(f"weighted_matrix error: source_dataframe sequences in \"{seq_col}\" vary in length from {sequence_lengths.min()} to {sequence_lengths.max()}, but must be equal.")
        else: 
            correct_uniform_length = False
    elif not first_length_matches:
        if raise_error: 
            raise ValueError(f"weighted_matrix error: source_dataframe sequences are {sequence_lengths[0]} amino acids long, but motif_length is set to {motif_length}")
        else: 
            correct_uniform_length = False
    
    if not raise_error: 
        return correct_uniform_length