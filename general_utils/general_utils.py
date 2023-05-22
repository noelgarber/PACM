import csv

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