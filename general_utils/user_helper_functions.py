from general_utils.general_utils import input_number

def get_min_members(display_message = True):
    '''
    Simple function to prompt the user to provide the minimum number of member peptides for a given category

    Args:
        display_message (bool): whether to display the informational message on why this is necessary

    Returns:
        minimum_members (int): an integer representing the minimum number of peptides in a given group that must be
                               present before it is used for matrix-building
    '''
    if display_message:
        print("Weighted matrices are calculated relative to a reference position being in a particular chemical class (e.g. acidic, basic, hydrophobic, etc.).",
              "\n    --> They are based on studying all the peptides following this position-type rule. ",
              "\n    --> We advise setting a minimum number of peptides here to prevent overfitting.",
              "\nHow many peptides are required before a matrix defaults to using the total list rather than the type-position rule-following subset?")
        minimum_members = input_number(prompt = "Input an integer: ", mode = "int")
    else:
        minimum_members = input_number(prompt = "Please enter the minimum peptide count for a given group to be included in matrix-building: ", mode = "int")

    return minimum_members

def get_always_allowed(slim_length):
    '''
    Simple function to get a user-inputted dict of position # --> list of residues that are always permitted at that position

    Args:
        slim_length (int): the length of the motif being studied

    Returns:
        always_allowed_dict (dict): a dictionary of position number (int) --> always-permitted residues at that position (list)
    '''
    input_always_allowed = input("Would you like to input residues always allowed at certain positions, rather than auto-generating? (Y/N)  ")

    always_allowed_dict = {}

    for i in np.arange(1, slim_length + 1):
        position = "#" + str(i)

        allowed_list = []
        if input_always_allowed == "Y":
            allowed_str = input(f"Enter comma-delimited residues always allowed at position {position} (e.g. \"D,E\"): ")
            if len(allowed_str) > 0:
                allowed_list = allowed_str.split(",")

        always_allowed_dict[position] = allowed_list

    return always_allowed_dict

def get_thresholds(percentiles_dict = None, use_percentiles = True, show_guidance = True):
    '''
    Simple function to prompt the user to define thresholds and corresponding points for the point assignment system

    Args:
        percentiles_dict (dict): dictionary of percentile numbers --> signal values
        use_percentiles (bool):  whether to display percentiles from a dict and use percentiles for setting thresholds
        show_guidance (bool): 	 whether to display guidance for setting thresholds

    Returns:
        thres_tuple (tuple):  tuple of 3 thresholds for extreme, high, and mid-level signal values
        points_tuple (tuple): tuple of 4 point values corresponding to extreme, high, mid-level, and below-threshold signal values
    '''

    print("Setting thresholds for scoring.",
          "\n\tUse more points for HIGH signal hits to produce a model that correlates strongly with signal intensity.",
          "\n\tUse more points for LOW signal hits to produce a model that is better at finding weak positives.",
          "\n---") if show_guidance else None

    # Set threshold values
    no_more_thresholds = False
    thresholds_dict = {}
    while not no_more_thresholds:
        if use_percentiles:
            current_threshold = input("Enter the next threshold as an integer percentile:  ")
            current_threshold = percentiles_dict.get(current_threshold)
        else:
            current_threshold = input("Enter the next signal threshold:  ")

        if current_threshold != "":
            current_threshold = float(current_threshold)
            current_points = input_number(f"\tPoints for hits >= {current_threshold}:  ", "float")
            thresholds_dict[current_threshold] = current_points
        else:
            no_more_thresholds = True

    return thresholds_dict

def get_position_weights(slim_length):
    '''
    Simple function to prompt the user for weights to apply to each position of the motif sequence

    Args:
        slim_length (int): the length of the motif being assessed

    Returns:
        position_weights (dict): a dictionary of position (int) --> weight (float)
    '''
    print("Enter numerical weights for each position based on their expected structural importance. If unknown, use 1.")
    position_weights = {}
    for position in np.arange(1, slim_length + 1):
        weight = input_number(f"\tEnter weight for position {position}:  ", "float")
        position_weights[position] = weight

    return position_weights