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
        try:
            if mode == "int":
                output_value = int(input_value)
                input_finished = True
            elif mode == "float":
                output_value = float(input_float)
                input_finished = True
        except:
            print(f"\tinput value was not {mode}; please try again.")

    return output_value

def csv_to_dict(filepath):
    '''
    Simple function for converting a 2-column CSV file into a dictionary.

    Args:
        filepath (str): the path to the CSV file, which must contain 2 columns without titles; keys go in first column

    Returns:
        result (dict): a dictionary of keys (from the first column) matching values (from the second column)
    '''
    result = {}
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            key = row[0]
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