# This script conducts residue-residue pairwise analysis to generate position-aware SLiM matrices and back-calculated scores.

#Import required functions and packages

import numpy as np
import pandas as pd
import os
import pickle
from itertools import product
from general_utils.general_utils import input_number

# Declare the sorted list of amino acids
amino_acids = ("D", "E", "R", "H", "K", "S", "T", "N", "Q", "C", "G", "P", "A", "V", "I", "L", "M", "F", "Y", "W")
amino_acids_phos = ("D", "E", "R", "H", "K", "S", "T", "N", "Q", "C", "G", "P", "A", "V", "I", "L", "M", "F", "Y", "W", "B", "J", "O") # B=pSer, J=pThr, Y=pTyr

# Declare the amino acid chemical characteristics dictionary
aa_charac_dict = {
    "Acidic": ["D", "E"],
    "Basic": ["K", "R", "H"],
    "ST": ["S", "T"],
    "Aromatic": ["F", "Y"],
    "Aliphatic": ["A", "V", "I", "L"],
    "Other_Hydrophobic": ["W", "M"],
    "Polar_Uncharged": ["N", "Q", "C"],
    "Special_Cases": ["P", "G"]
}

def get_min_cat_members(display_message = True):
    '''
    Simple function to prompt the user to provide the minimum number of member peptides for a given category

    Args:
        None

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

#DEFINE GENERALIZED MATRIX FUNCTION

def weighted_matrix(bait, motif_length, source_dataframe, min_members, amino_acid_list, thres_tuple, points_tuple,
                    position_for_filtering = None, residues_included_at_filter_position = amino_acids_phos,
                    bjo_seq_col = "BJO_Sequence", signal_col_suffix = "Background-Adjusted_Standardized_Signal"):
    '''
    Generalized weighted matrix function

    Args:
        bait (str):										the bait to use for matrix generation
                                                        --> can be set to "Best" to use the top bait for each peptide
        motif_length (int): 							the length of the motif being assessed
        source_dataframe (pd.DataFrame):				the dataframe containing the peptide binding data
        min_members (int):								the minimum number of peptides that must be present in a group
                                                        to be used for matrix generation
        amino_acid_list (list): 						the amino acid alphabet to be used in matrix construction
                                                        --> default is standard 20 plus phospho: pSer=B, pThr=J, pTyr=O
        thres_tuple (tuple): 							tuple of (thres_extreme, thres_high, thres_mid) signal thres
        points_tuple (tuple): 							tuple of (points_extreme, points_high, points_mid, points_low)
                                                        representing points associated with peptides above thresholds
        position_for_filtering (int):					the position to omit during matrix generation
        residues_included_at_filter_position (list):	list of residues in the chemical class being assessed
        bjo_seq_col (str): 								the name of the column in the source_dataframe that contains the
                                                        sequences, where phospho-residues are represented as [B,J,O]
        signal_col_suffix (str): 						the suffix of columns containing signal values

    Returns:
        generic_matrix_df (pd.DataFrame):			standardized matrix for the given type-position rule
    '''
    index_for_filtering = position_for_filtering - 1

    # Declare thresholds and point values
    thres_extreme, thres_high, thres_mid = thres_tuple
    points_extreme, points_high, points_mid, points_low = points_tuple

    # Create a list to contain numbered positions across the motif length, to use as column headers in weighted matrices
    list_pos = []
    for i in range(1, int(motif_length) + 1):
        list_pos.append("#" + str(i))

    # Initialize a dataframe where the index is the list of amino acids and the columns are the positions (e.g. #1)
    generic_matrix_df = pd.DataFrame(index = amino_acid_list, columns = list_pos)
    generic_matrix_df = generic_matrix_df.fillna(0)

    # Delete any position indices that are declared as being filtered out
    position_indices = np.arange(0, motif_length)
    if position_for_filtering != None:
        position_indices_filtered = np.delete(position_indices, position_for_filtering - 1)
    else:
        position_indices_filtered = position_indices

    # Default to no filtering if the number of members is below the minimum.
    num_qualifying_entries = 0
    for i in np.arange(len(source_dataframe)):
        seq = source_dataframe.at[i, bjo_seq_col]
        if position_for_filtering > len(seq):
            raise IndexError(f"Position {position_for_filtering} (index {index_for_filtering}) is out of bounds for sequence {seq}")
        else:
            aa_at_filter_index = seq[index_for_filtering]
            if aa_at_filter_index in residues_included_at_filter_position:
                num_qualifying_entries += 1
    if num_qualifying_entries < min_members:
        residues_included_at_filter_position = amino_acid_list

    # Calculate the points and assign to the matrix.
    for i in np.arange(len(source_dataframe)):
        seq = source_dataframe.at[i, bjo_seq_col]
        seq = list(seq) #converts to aa list

        # Check if the current bait passes as a positive result for the given peptide sequence
        if bait == "Best":
            passes = source_dataframe.at[i, "One_Passes"]
        elif source_dataframe.at[i, bait + "_Passes"] == "Yes":
            passes = "Yes"
        else:
            passes = "No"

        # Find the mean signal value
        if bait == "Best":
            value = source_dataframe.at[i, "Max_Bait_Background-Adjusted_Mean"]
        else:
            # Calculate the mean of signal values for the bait being assessed
            value_list = []
            for col in source_dataframe.columns:
                if signal_col_suffix in col and bait in col:
                    value_list.append(source_dataframe.at[i, col])
            value_list = np.array(value_list)
            value = value_list.mean()

        # Check if the residue at the filter position belongs to the residues list for the chemical class of interest
        if seq[position_for_filtering - 1] in residues_included_at_filter_position:
            # Iterate over the filtered position indices to apply points to the matrix dataframe
            for n in position_indices_filtered:
                m = n + 1 # value for column number
                seq_at_position = seq[n]
                for aa in amino_acid_list:
                    if aa == seq_at_position:
                        # Calculation of points:
                        if value > thres_extreme:
                            points = points_extreme
                        elif value > thres_high:
                            points = points_high
                        elif value > thres_mid:
                            points = points_mid
                        else:
                            points = points_low

                        # Pass-conditional point assignment:
                        if passes == "Yes":
                            generic_matrix_df.at[aa, "#" + str(m)] += points

    # Convert matrix to floating point values
    generic_matrix_df = generic_matrix_df.astype("float32")

    return generic_matrix_df

def get_thresholds(percentiles_dict = None, use_percentiles = True, show_guidance = True, display_points_system = False):
    '''
    Simple function to define thresholds and corresponding points for the point assignment system

    Args:
        percentiles_dict (): 	dictionary of percentile numbers --> signal values
        use_percentiles (bool): whether to display percentiles from a dict and use percentiles for setting thresholds
        show_guidance (bool): 	whether to display guidance for setting thresholds

    Returns:
        thres_tuple (tuple): 	tuple of 3 thresholds for extreme, high, and mid-level signal values
        points_tuple (tuple): 	tuple of 4 point values corresponding to extreme, high, mid-level, and below-threshold signal values
    '''
    if show_guidance:
        print("Setting thresholds for scoring.",
              "\n\tUse more points for HIGH signal hits to produce a model that correlates strongly with signal intensity.",
              "\n\tUse more points for LOW signal hits to produce a model that is better at finding weak positives.",
              "\nWe suggest the 90th, 80th, and 70th percentiles as thresholds, but this may vary depending on the number of hits expected.",
              "\n---")

    # Set threshold values
    if use_percentiles:
        thres_extreme = int(input("Enter the upper percentile threshold (1 of 3):  "))
        thres_extreme = percentiles_dict.get(thres_extreme)

        thres_high = int(input("Enter the upper-middle percentile threshold (2 of 3):  "))
        thres_high = percentiles_dict.get(thres_high)

        thres_mid = int(input("Enter the lower-middle percentile threshold (3 of 3):  "))
        thres_mid = percentiles_dict.get(thres_mid)
    else:
        thres_extreme = int(input("Enter the upper signal threshold (1 of 3):  "))
        thres_high = int(input("Enter the upper-middle signal threshold (2 of 3):  "))
        thres_mid = int(input("Enter the lower-middle signal threshold (3 of 3):  "))

    thres_tuple = (thres_extreme, thres_high, thres_mid)

    # Set number of points for each threshold
    points_extreme = float(input("How many points for values greater than " + str(thres_extreme) + "? Input:  "))
    points_high = float(input("How many points for values greater than " + str(thres_high) + "? Input:  "))
    points_mid = float(input("How many points for values greater than " + str(thres_mid) + "? Input:  "))
    points_low = float(input("Some hits are marked significant but fall below the signal threshold. How many points for these lower hits? Input:  "))
    points_tuple = (points_extreme, points_high, points_mid, points_low)

    if display_points_system:
        print("----------------------",
              "\nInputted point System: ",
              "\nIf max_bait >", thres_extreme, "and passes, points =", points_extreme,
              "\nIf max_bait >", thres_high, "and passes, points =", points_high,
              "\nIf max_bait >", thres_mid, "and passes, points =", points_mid,
              "\nIf max_bait > 0 and passes, points =", points_low,
              "\n----------------------")

    return thres_tuple, points_tuple

def make_weighted_matrices(slim_length, aa_charac_dict, dens_df, minimum_members, list_aa, thres_tuple, points_tuple,
                           sequence_col = "BJO_Sequence", signal_col_suffix = "Background-Adjusted_Standardized_Signal"):
    '''
    Function for generating weighted matrices corresponding to each type/position rule (e.g. position #1 = Acidic)

    Args:
        slim_length (int): 		the length of the motif being studied
        aa_charac_dict (dict): 	the dictionary of amino acid characteristics and their constituent amino acids
        dens_df (pd.DataFrame): the dataframe containing peptide spot intensity data
        minimum_members (int): 	the minimum number of peptides that must be present in a group to be used for matrix generation
        list_aa (tuple):		the list of amino acids to use for weighted matrix rows
        thres_tuple (tuple):	a tuple of (thres_extreme, thres_high, thres_mid)
        points_tuple (tuple):	a tuple of (points_extreme, points_high, points_mid, points_low)

    Returns:
        dictionary_of_matrices (dict): a dictionary of standardized matrices
    '''
    # Declare dict where keys are position-type rules (e.g. "#1=Acidic") and values are corresponding weighted matrices
    dictionary_of_matrices = {}

    # Iterate over columns for the weighted matrix (position numbers)
    for col_num in range(1, slim_length + 1):
        # Iterate over dict of chemical characteristic --> list of member amino acids (e.g. "Acidic" --> ["D","E"]
        for charac, mem_list in aa_charac_dict.items():
            # Generate the weighted matrix
            weighted_matrix_containing_charac = weighted_matrix(bait = "Best", motif_length = slim_length, source_dataframe = dens_df,
                                                                min_members = minimum_members, amino_acid_list = list_aa,
                                                                thres_tuple = thres_tuple, points_tuple = points_tuple,
                                                                position_for_filtering = col_num, residues_included_at_filter_position = mem_list,
                                                                bjo_seq_col = sequence_col, signal_col_suffix = signal_col_suffix)

            # Standardize the weighted matrix so that the max value is 1
            for n in np.arange(1, slim_length + 1):
                # Declare the column name index
                col_name = "#" + str(n)

                # Find the max value in the weighted matrix, expressed in number of assigned points
                max_value = weighted_matrix_containing_charac[col_name].max()
                if max_value == 0:
                    max_value = 1 # required to avoid divide-by-zero error

                # Iterate over the rows of the weighted matrix and standardize each value to be relative to the max value
                for i, row in weighted_matrix_containing_charac.iterrows():
                    weighted_matrix_containing_charac.at[i, col_name] = weighted_matrix_containing_charac.at[i, col_name] / max_value

            # Assign the weighted matrix to the dictionary
            dict_key_name = "#" + str(col_num) + "=" + charac
            dictionary_of_matrices[dict_key_name] = weighted_matrix_containing_charac

    return dictionary_of_matrices

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
            allowed_str = input("Enter comma-delimited residues always allowed at position " + position + " (e.g. \"D,E\"): ")
            if len(allowed_str) > 0:
                allowed_list = allowed_str.split(",")

        always_allowed_dict[position] = allowed_list

    return always_allowed_dict


print("---")

def collapse_phospho(matrices_dict, slim_length):
    '''
    Function to collapse the matrix rows for B,J,O (pSer, pThr, pTyr) into S,T,Y respectively, since phosphorylation
    status is not usually known when scoring a de novo sequence.

    Args:
        matrices_dict (dict): a dictionary of weighted matrices containing rows for B, J, O
        slim_length (int): the length of the motif represented by these matrices

    Returns:
        matrices_dict (dict): updated dictionary with rows collapsed per the above description
    '''
    for key, df in matrices_dict.items():
        for n in np.arange(1, slim_length + 1):
            df.at["S", "#" + str(n)] = df.at["S", "#" + str(n)] + df.at["B", "#" + str(n)]
            df.at["T", "#" + str(n)] = df.at["T", "#" + str(n)] + df.at["J", "#" + str(n)]
            df.at["Y", "#" + str(n)] = df.at["Y", "#" + str(n)] + df.at["O", "#" + str(n)]
        df.drop(labels=["B", "J", "O"], axis=0, inplace=True)
        matrices_dict[key] = df

    return matrices_dict

def apply_always_allowed(matrices_dict, slim_length, always_allowed_dict):
    '''
    Function to apply the override always-allowed residues specified by the user in the matrices

    Args:
        matrices_dict (dict): a dictionary of weighted matrices
        slim_length (int): the length of the motif represented by these matrices

    Returns:
        matrices_dict (dict): updated dictionary where always-allowed residues at each position are set to the max score
    '''
    for key, df in matrices_dict.items():
        for i in np.arange(1, slim_length + 1):
            position = "#" + str(i)
            always_allowed_residues = always_allowed_dict.get(position)
            for residue in always_allowed_residues:
                df.at[residue, position] = 1

        matrices_dict[key] = df

    return matrices_dict

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

def add_matrix_weights(matrices_dict, position_weights):
    '''
    Function to apply the matrix weights by position to the generated matrices

    Args:
        matrices_dict (dict):    dictionary of type-position rule --> unadjusted position-weighted matrix
        position_weights (list): list of position weights; length is equal to slim_length

    Returns:
        weighted_matrices_dict (dict): same as matrices_dict, but with the weights applied to the matrix values
    '''
    weighted_matrices_dict = {}

    # Backwards compatibility for when position_weights is a dict of positions
    if isinstance(position_weights, dict):
        position_weights = list(position_weights.values())

    for key, df in matrices_dict.items():
        output_df = df.copy()
        for i, weight in enumerate(position_weights):
            position = "#" + str(i+1)
            output_df[position] = output_df[position] * weight
        weighted_matrices_dict[key] = output_df

    return weighted_matrices_dict


def save_weighted_matrices(weighted_matrices_dict, matrix_directory = None, save_pickled_dict = True):
    '''
    Simple function to save the weighted matrices to disk

    Args:
        weighted_matrices_dict (dict): the dictionary of type-position rule --> corresponding weighted matrix
        matrix_directory (str): directory to save matrices into; defaults to a subfolder called Pairwise_Matrices

    Returns:
        None
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

def aa_chemical_class(amino_acid, dict_of_aa_characs = aa_charac_dict):
    '''
    Simple function to check if a particular amino acid is a member of a chemical class in the dictionary

    Args:
        amino_acid (str): the amino acid as a single letter code
        dict_of_aa_characs (dict): a dictionary of chemical_characteristic --> [amino acid members]

    Returns:
        charac_result (str): the chemical class that the amino acid belongs to
    '''
    charac_result = None
    for charac, mem_list in dict_of_aa_characs.items():
        if amino_acid in mem_list:
            charac_result = charac

    return charac_result

def score_aa_seq(sequence, weighted_matrices, slim_length, dens_df = None, df_row_index = None, add_residue_cols = False):
    '''
    Function to score amino acid sequences based on the dictionary of context-aware weighted matrices

    Args:
        sequence (str): 		  the amino acid sequence; must be the same length as the motif described by the weighted matrices
        weighted_matrices (dict): dictionary of type-position rule --> position-weighted matrix
        slim_length (int): 		  the length of the motif being studied
        dens_df (pd.DataFrame):   if add_residue_cols is True, must be the dataframe to add residue col values to
        df_row_index (int):       if add_residue_cols is True, must be the row of the dataframe to assign residue col values
        add_residue_cols (bool):  whether to add columns containing individual residue letters, for sorting, in a df

    Returns:
        output_total_score (float): the total motif score for the input sequence
    '''

    # Check that input arguments are correct
    if not isinstance(sequence, str):
        raise ValueError(f"score_aa_seq input sequence is {type(sequence)}, but string was expected")
    elif not isinstance(slim_length, int):
        raise ValueError(f"score_aa_seq slim_length type is {type(slim_length)}, but must be int")
    elif not isinstance(weighted_matrices, dict):
        raise ValueError(f"score_aa_seq weighted_matrices type is {type(weighted_matrices)}, but must be a dict of dataframes")
    elif len(sequence) != slim_length:
        raise ValueError(f"score_aa_seq input sequence length was {len(sequence)}, but must be equal to slim_length ({slim_length})")

    # Get sequence as numpy array
    sequence_array = np.array(list(sequence))

    # Define residues of interest; assume slim_length is equal to sequence length
    current_residues = sequence_array.copy()

    # Define residues flanking either side of the residues of interest; for out-of-bounds cases, use only the other side
    flanking_left = np.concatenate((current_residues[0:1], current_residues[0:-1]), axis=0)
    flanking_right = np.concatenate((current_residues[1:], current_residues[-1:]), axis=0)

    # Get chemical classes
    flanking_left_classes = np.array([aa_chemical_class(left_residue) for left_residue in flanking_left])
    flanking_right_classes = np.array([aa_chemical_class(right_residue) for right_residue in flanking_right])

    # Get positions, indexed from 1, for residues of interest and the flanking residues on either side
    positions = np.arange(1, slim_length + 1)
    left_positions = positions - 1
    right_positions = positions + 1
    left_positions[0] = right_positions[0]
    right_positions[-1] = left_positions[-1]

    # Get keys for weighted matrices
    flanking_left_keys = np.char.add("#", left_positions.astype(str))
    flanking_left_keys = np.char.add(flanking_left_keys, "=")
    flanking_left_keys = np.char.add(flanking_left_keys, flanking_left_classes)

    flanking_right_keys = np.char.add("#", right_positions.astype(str))
    flanking_right_keys = np.char.add(flanking_right_keys, "=")
    flanking_right_keys = np.char.add(flanking_right_keys, flanking_right_classes)

    # Get weighted matrices from keys
    flanking_left_weighted_matrices = [weighted_matrices.get(key) for key in flanking_left_keys]
    flanking_right_weighted_matrices = [weighted_matrices.get(key) for key in flanking_right_keys]

    # Get column names for weighted matrix dataframes
    matrix_columns = ["#" + str(position) for position in positions]

    # Compute score values
    flanking_left_scores = np.array([flanking_left_weighted_matrices[index].at[current_residues[index], matrix_columns[index]] for index in np.arange(slim_length)])
    flanking_right_scores = np.array([flanking_right_weighted_matrices[index].at[current_residues[index], matrix_columns[index]] for index in np.arange(slim_length)])
    total_position_scores = flanking_left_scores + flanking_right_scores

    # Get the total score
    output_total_score = np.sum(total_position_scores)

    # Optionally add columns in-place for each residue in the sequence, for easy searching in the output
    if add_residue_cols:
        residue_col_names = np.char.add("Residue_", np.arange(1, slim_length + 1).astype(str)).tolist()
        dens_df.loc[df_row_index, residue_col_names] = list(sequence)

    return output_total_score

def apply_motif_scores(dens_df, weighted_matrices, slim_length, seq_col = "No_Phos_Sequence", score_col = "SLiM_Score", add_residue_cols = False):
    '''
    Function to apply the score_aa_seq() function to all sequences in the source dataframe

    Args:
        dens_df (pd.DataFrame):   dataframe containing the motif sequences to back-apply motif scores onto, that were originally used to generate the scoring system
        weighted_matrices (dict): dictionary of type-position rule --> position-weighted matrix
        slim_length (int): 		  the length of the motif being studied
        seq_col (str): 			  the column in dens_df that contains the peptide sequence to score (unphosphorylated, if model phospho-residues were collapsed to non-phospho during building)
        score_col (str): 		  the column in dens_df that will contain the score values
        add_residue_cols (bool):  whether to add columns containing individual residue letters, for sorting, in a df

    Returns:
        output_df (pd.DataFrame): dens_df with scores added
    '''
    output_df = dens_df.copy()

    sequences = output_df[seq_col].values.tolist()
    scores = []
    for seq in sequences:
        total_score = score_aa_seq(sequence = seq, weighted_matrices = weighted_matrices, slim_length = slim_length)
        scores.append(total_score)

    output_df[score_col] = scores

    # Organize residue cols
    if add_residue_cols:
        cols_to_move = []
        for i in np.arange(1, slim_length + 1):
            current_col = "Residue_" + str(i)
            cols_to_move.append(current_col)

        columns = output_df.columns.tolist()
        for col in cols_to_move:
            columns.remove(col)

        insert_index = columns.index(seq_col) + 1
        for col in cols_to_move[::-1]:
            columns.insert(insert_index, col)

        output_df = output_df[columns]

    return output_df

def diagnostic_value(score_threshold, dataframe, significance_col = "One_Passes", score_col = "SLiM_Score",
                     truth_value = "Yes", return_dict = True, return_fdr_for = False):
    '''
    Function to produce sensitivity, specificity, and positive and negative predictive values for a given score cutoff

    Args:
        score_threshold (float):  the score cutoff to use
        dataframe (pd.DataFrame): the dataframe containing peptide information
        significance_col (str):   the column containing significance information
        score_col (str):          the column containing the back-calculated motif score for the peptides
        truth_value (str):        the truth value found in significance_col; by default, it is the string "Yes"

    Returns:
        pred_val_dict (dict):     if return_dict, returns a dict of TP, FP, TN, FN, Sensitivity, Specificity, PPV, NPV, FDR, FOR
                                  --> divide_by_zero occurrences result in np.nan
    '''

    # Calculate the boolean arrays for conditions
    score_above_thres = dataframe[score_col] >= score_threshold
    sig_truth = dataframe[significance_col] == truth_value

    # Count the occurrences of each call type
    TP_count = np.sum(score_above_thres & sig_truth)
    FP_count = np.sum(score_above_thres & ~sig_truth)
    FN_count = np.sum(~score_above_thres & sig_truth)
    TN_count = np.sum(~score_above_thres & ~sig_truth)

    # Calculate Sensitivity, Specificity, PPV, NPV, FDR, and FOR
    sensitivity = TP_count/(TP_count+FN_count) if (TP_count+FN_count) > 0 else 0
    specificity = TN_count/(TN_count+FP_count) if (TN_count+FP_count) > 0 else 0
    ppv = TP_count/(TP_count+FP_count) if (TP_count+FP_count) > 0 else 0
    npv = TN_count/(TN_count+FN_count) if (TN_count+FN_count) > 0 else 0
    false_discovery_rate = 1 - ppv
    false_omission_rate = 1 - npv

    if return_dict:
        pred_val_dict = {"TP": TP_count,
                         "FP": FP_count,
                         "TN": TN_count,
                         "FN": FN_count,
                         "Sensitivity": sensitivity,
                         "Specificity": specificity,
                         "PPV": ppv,
                         "NPV": npv,
                         "FDR": false_discovery_rate,
                         "FOR": false_omission_rate}
        return pred_val_dict
    elif return_fdr_for:
        return false_discovery_rate, false_omission_rate
    else:
        return sensitivity, specificity, ppv, npv, false_discovery_rate, false_omission_rate

def apply_threshold(input_df, score_range_series = None, sig_col = "One_Passes", score_col = "SLiM_Score", range_count = 100,
                    return_pred_vals_only = False, return_optimized_fdr = False, verbose = False):
    '''
    Function to declare and apply the motif score threshold based on predictive values

    Args:
        input_df (pd.DataFrame):         the dataframe containing peptides and scores; must contain score values
        score_range_series (np.ndarray): if function is used in a loop, providing this upfront improves performance
        sig_col (str): 			         the df column containing significance information (Yes/No)
        score_col (str): 		         the df column containing the peptide scores
        range_count (int):               number of score values to test; default is 100
        return_pred_vals_only (bool):    whether to only return predictive values without setting and applying a threshold
        return_optimized_fdr (bool):     whether to only return best_score, best_fdr, and best_for
        verbose (bool):                  whether to display debugging information

    Returns:
        output_df (pd.DataFrame):              dens_df with a new column containing calls based on the selected score
        selected_threshold (float):            the selected score threshold
    '''

    # Make a range of SLiM scores between the minimum and maximum score values from the dataframe
    if score_range_series is None:
        min_score = input_df[score_col].min()
        max_score = input_df[score_col].max()
        print(f"max_score = {max_score} | min_score = {min_score}") if verbose else None
        score_range_series = np.linspace(min_score, max_score, num = range_count)

    # Calculate positive and negative predictive values (PPV/NPV), and false discovery and omission rates (FDR/FOR)
    ppvs, npvs, fdrs, fors = [], [], [], []

    for current_score in score_range_series:
        _, _, ppv, npv, fdr_val, for_val = diagnostic_value(score_threshold = current_score, dataframe = input_df,
                                                            significance_col = "One_Passes", score_col = score_col,
                                                            return_dict = False, return_fdr_for = False)
        ppvs.append(ppv)
        npvs.append(npv)
        fdrs.append(fdr_val)
        fors.append(for_val)

    if return_optimized_fdr:
        # Find the row where the FDR/FOR ratio is closest to 1, and use that for the FDR
        motif_scores = score_range_series
        with np.errstate(divide="ignore"):
            # Ignores RuntimeWarnings where divide-by-zero occurs
            ratios = np.divide(fdrs, fors, out=np.full_like(fdrs, np.inf), where=(fors != 0))
        closest_index = np.argmin(np.abs(ratios - 1))

        best_fdr = fdrs[closest_index]
        best_for = fors[closest_index]
        best_score = motif_scores[closest_index]

        return best_score, best_fdr, best_for

    # Assemble a dataframe of score --> [positive predictive value, negative predictive value]
    predictive_value_df = pd.DataFrame(columns=["Score", "PPV", "NPV", "FDR", "FOR"])
    predictive_value_df["Score"] = score_range_series
    predictive_value_df["PPV"] = ppvs
    predictive_value_df["NPV"] = npvs
    predictive_value_df["FDR"] = fdrs
    predictive_value_df["FOR"] = fors

    if return_pred_vals_only:
        return predictive_value_df

    # Print the dataframe to aid in the user selecting an appropriate score cutoff for significance to be declared
    print("Threshold selection information:",)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(predictive_value_df)
    print("---")

    # Prompt the user to input the selected threshold that a SLiM score must exceed to be considered significant
    selected_threshold = input_number("Input your selected threshold for calling hits:  ", "float")

    # Apply the selected threshold to the dataframe to call hits
    output_df = input_df.copy()
    for i in np.arange(len(output_df)):
        # Get score and pass info for current row
        current_score = output_df.at[i, "SLiM_Score"]
        spot_passes = output_df.at[i, sig_col]

        # Make calls on true/false positives/negatives
        if current_score >= selected_threshold:
            output_df.at[i, "Call"] = "Positive"
            if spot_passes == "Yes":
                call_type = "TP"
            else:
                call_type = "FP"
        else:
            output_df.at[i, "Call"] = "-"
            if spot_passes == "Yes":
                call_type = "FN"
            else:
                call_type = "TN"

        # Apply the call to the dataframe at the specified row
        output_df.at[i, "Call_Type"] = call_type

    print("Applied hit calls based on threshold.")

    return output_df, selected_threshold, predictive_value_df

def save_scored_data(scored_df, selected_threshold, output_directory, verbose = True):
    '''
    Simple function to save original data with SLiM scores

    Args:
        scored_df (pd.DataFrame): 	the dataframe to save
        selected_threshold (float): the selected thresholding value for making calls; may be set to None to omit
        output_directory (str): 	the output directory
        verbose (bool): 			whether to display user feedback

    Returns:
        output_file_path (str): 	the output path where the file was saved
    '''

    if selected_threshold is None:
        output_filename = "pairwise_scored_data.csv"
    else:
        output_filename = "pairwise_scored_data_thres" + str(selected_threshold) + ".csv"
    parent_directory = os.path.join(output_directory, "Array_Output")
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
    output_file_path = os.path.join(parent_directory, output_filename)
    scored_df.to_csv(output_file_path)

    print("Saved! Filename:", output_file_path) if verbose else None

    return output_file_path

def permute_weights(slim_length, position_copies):
    '''
    Simple function to generate permutations of weights from 0-3 for scoring a peptide motif of defined length

    Args:
        slim_length (int): 	    the length of the motif for which weights are being generated
        position_copies (dict): a dictionary of position --> copy_num, where the sum of dict values equals slim_length

    Returns:
        expanded_weights_array (np.ndarray): an array of shape (permutations_number, slim_length)
    '''
    # Check that the dictionary of position copies is the correct length
    if slim_length != sum(position_copies.values()):
        raise ValueError(f"permute_weights error: slim_length ({slim_length}) is not equal to position_copies dict values sum ({sum(position_copies.values())})")

    # Get permutations of possible weights at each position
    permutations_length = slim_length
    for position, total_copies in position_copies.items():
        permutations_length = permutations_length - (total_copies - 1)
    permuted_weights = np.array(np.meshgrid(*([[3, 2, 1, 0]] * permutations_length))).T.reshape(-1, permutations_length)

    # Expand permutations to copied columns
    expanded_weights_list = []
    for position, total_copies in position_copies.items():
        current_column = permuted_weights[:, position:position + 1]
        if total_copies == 1:
            expanded_weights_list.append(current_column)
        else:
            repeated_column = np.repeat(current_column, total_copies, axis=1)
            expanded_weights_list.append(repeated_column)
    expanded_weights_array = np.hstack(expanded_weights_list)

    print(f"Shape of expanded_weights_array: {expanded_weights_array.shape}")

    return expanded_weights_array

def make_pairwise_matrices(dens_df, percentiles_dict = None, slim_length = None, minimum_members = None,
                           thres_tuple = None, points_tuple = None, always_allowed_dict = None, position_weights = None,
                           output_folder = None, sequence_col = "BJO_Sequence", significance_col = "One_Passes",
                           make_calls = True, optimize_weights = False, position_copies = None, verbose = True):
    '''
    Main function for making pairwise position-weighted matrices

    Args:
        dens_df (pd.DataFrame): 	the dataframe containing densitometry values for the peptides being analyzed
        percentiles_dict (dict): 	dictionary of percentile --> mean signal value, for use in thesholding
        slim_length (int): 			the length of the motif being studied
        minimum_members (int): 		the minimum number ofo peptides that must belong to a chemically classified group
                                    in order for them to be used in pairwise matrix-building
        thres_tuple (tuple): 		tuple of (thres_extreme, thres_high, thres_mid) signal thres
        points_tuple (tuple): 		tuple of (points_extreme, points_high, points_mid, points_low)
                                    representing points associated with peptides above thresholds
        always_allowed_dict (dict): a dictionary of position number (int) --> always-permitted residues at that position (list)
        position_weights (dict): 	a dictionary of position number (int) --> weight value (int or float)
        output_folder (str): 		the path to the folder where the output data should be saved
        sequence_col (str): 		the column in the dataframe that contains unphosphorylated peptide sequences
        significance_col (str): 	the column in the dataframe that contains significance calls (Yes/No)
        make_calls (bool): 			whether to prompt the user to set a threshold for making positive/negative calls
        optimize_weights (bool): 	whether to optimize position weights along the motif sequence to maximize FDR & FOR
        position_copies (dict): 	dict of permuted_weight_index --> copy_number, where the sum of all the dict values
                                    is equal to slim_length; only required when optimize_weights is set to True
        verbose (bool): 			whether to display user feedback and debugging information

    Returns:
        dens_df (pd.DataFrame): 			the modified dataframe with scores applied
        predictive_value_df (pd.DataFrame): a dataframe containing sensitivity/specificity/PPV/NPV values for different
                                            score thresholds
    '''
    print("Starting pairwise matrices generation process...") if verbose else None

    # Get the minimum number of peptides that must belong to a classified group for them to be used in matrix-building
    if minimum_members is None:
        minimum_members = get_min_cat_members()

    # Define the length of the short linear motif (SLiM) being studied
    if slim_length is None:
        slim_length = int(input("Enter the length of your SLiM of interest as an integer (e.g. 15):  "))

    # Get threshold and point values
    if thres_tuple is None or points_tuple is None:
        thres_tuple, points_tuple = get_thresholds(percentiles_dict = percentiles_dict, use_percentiles = True,
                                                   show_guidance = True, display_points_system = True)

    # Make the dictionary of weighted matrices based on amino acid composition across positions
    print("Generating matrices...")
    matrices_dict = make_weighted_matrices(slim_length = slim_length, aa_charac_dict = aa_charac_dict,
                                           dens_df = dens_df, minimum_members = minimum_members, list_aa = amino_acids_phos,
                                           thres_tuple = thres_tuple, points_tuple = points_tuple,
                                           sequence_col = sequence_col,
                                           signal_col_suffix = "Background-Adjusted_Standardized_Signal")

    # Get list of always-allowed residues (overrides algorithm for those positions)
    print("Collapsing phospho-residues to their non-phospho counterparts and applying always-allowed residues...") if verbose else None
    if always_allowed_dict is None:
        always_allowed_dict = get_always_allowed(slim_length = slim_length)

    # Collapse phospho-residues into non-phospho counterparts and apply always-allowed residues
    matrices_dict = collapse_phospho(matrices_dict = matrices_dict, slim_length = slim_length)
    matrices_dict = apply_always_allowed(matrices_dict = matrices_dict, slim_length = slim_length,
                                         always_allowed_dict = always_allowed_dict)

    # Declare the output folder for saving pairwise weighted matrices
    if output_folder is None:
        output_folder = os.getcwd()
        output_folder = os.getcwd()
    matrix_output_folder = os.path.join(output_folder, "Pairwise_Matrices")

    # Apply weights to the generated matrices, or find optimal weights
    if optimize_weights:
        print("Starting automatic position weight optimization...") if verbose else None

        # Get permutations of possible weights at each position, as an array of shape (permutations_count, slim_length)
        print(f"\tComputing permutations of weights from 0 to 3 along {slim_length} positions...")
        expanded_weights_array = permute_weights(slim_length = slim_length, position_copies = position_copies)

        # Add each set of matrix weights and determine the optimal set
        print("\tIterating over permutations of weights...") if verbose else None
        best_fdr, best_for, best_score_threshold = 9999, 9999, 0
        best_weights, best_weighted_matrices_dict, predictive_value_df = None, None, None
        for weights_array in expanded_weights_array:
            weights_list = weights_array.tolist()
            print("\t\tCurrent weights:", weights_list) if verbose else None

            # Add the matrix weights
            #print("\t\t---\n", f"\t\tApplying matrix weights ({weights_list}) to the matrices...") if verbose else None
            current_weighted_matrices_dict = add_matrix_weights(matrices_dict, weights_list)

            # Apply motif scoring back onto the original sequences
            #print("\t\tBack-calculating SLiM scores on source data...")
            # TODO There is an error here! Extremely high motif scores are being generated, and it doesn't really make sense. What's going on?
            current_dens_df = apply_motif_scores(dens_df = dens_df, weighted_matrices = current_weighted_matrices_dict,
                                                 slim_length = slim_length, seq_col = sequence_col, score_col = "SLiM_Score",
                                                 add_residue_cols = False)

            # Obtain the optimal FDR, FOR, and corresponding SLiM Score
            #print("\t\tComputing optimal balance of FDR and FOR where values are similar, and matching SLiM score...")
            #print("---\nCurrent dataframe:")
            #print(current_dens_df)
            #print(matrices_dict.get("#1=Acidic"))
            #print("---")
            score_range_series = np.linspace(current_dens_df["SLiM_Score"].min(), current_dens_df["SLiM_Score"].max(), num=100)
            current_best_score, current_best_fdr, current_best_for = apply_threshold(current_dens_df, score_range_series = score_range_series,
                                                                                     sig_col = significance_col, score_col = "SLiM_Score",
                                                                                     return_optimized_fdr = True)

            # If the current best FDR is better than the previous record, update the best FDR value and assign best_weights
            if not np.isnan(current_best_fdr) and not np.isnan(current_best_for):
                if current_best_fdr < best_fdr:
                    best_fdr = current_best_fdr
                    best_for = current_best_for
                    best_score_threshold = current_best_score
                    best_weights = weights_list
                    best_weighted_matrices_dict = current_weighted_matrices_dict
                    best_dens_df = current_dens_df
                    print(f"\t\t\tNew record set for FDR={best_fdr} and FOR={best_for}                 <------------------ ***") if verbose else None
                else:
                    print(f"\t\t\tCurrent valid inferior FDR={current_best_fdr} and FOR={current_best_for}")

        if best_weights is not None:
            print("\t---\n", f"\tOptimal weights for pairwise matrices: {best_weights}")
            print(f"\t\t--> corresponding optimal FDR={best_fdr} and FOR={best_for} at a SLiM score threshold > {best_score_threshold}")
        else:
            raise Exception("make_pairwise_matrices error: failed to define best_weights")

        # Save the weighted matrices and scored data
        print("\tSaving weighted matrices and scored data...")
        save_weighted_matrices(weighted_matrices_dict = best_weighted_matrices_dict, matrix_directory = matrix_output_folder,
                               save_pickled_dict = True)
        save_scored_data(scored_df = dens_df, selected_threshold = best_score_threshold, output_directory = output_folder)

        return best_dens_df, predictive_value_df

    else:
        print("Starting manual application of position weights...") if verbose else None

        # Get weights for positions along the motif sequence
        if position_weights is None:
            position_weights = get_position_weights(slim_length = slim_length)

        # Apply the weights to the matrices
        print(f"\tApplying matrix weights of {position_weights}...") if verbose else None
        weighted_matrices_dict = add_matrix_weights(matrices_dict = matrices_dict,
                                                    position_weights = position_weights)

        # Apply the motif scoring algorithm back onto the peptide sequences
        print(f"\tBack-calculating SLiM scores on source data...")
        dens_df = apply_motif_scores(dens_df = dens_df, weighted_matrices = weighted_matrices_dict,
                                     slim_length = slim_length, seq_col = sequence_col, score_col = "SLiM_Score",
                                     add_residue_cols = True)

        # Use thresholding to declare true/false positives/negatives in the peptide sequences
        if make_calls:
            print(f"\tApplying thresholding and making positive/negative calls on source data...")
            dens_df, selected_threshold, predictive_value_df = apply_threshold(dens_df, sig_col = significance_col,
                                                                               score_col = "SLiM_Score")
        else:
            print(f"\tGetting predictive value dataframe for different score thresholds...")
            selected_threshold = None
            predictive_value_df = apply_threshold(dens_df, sig_col = significance_col, score_col = "SLiM_Score",
                                                  return_pred_vals_only = True)

        # Save the weighted matrices and scored data
        print("Saving weighted matrices and scored data...")
        save_weighted_matrices(weighted_matrices_dict = weighted_matrices_dict, matrix_directory = matrix_output_folder,
                               save_pickled_dict = True)
        save_scored_data(scored_df = dens_df, selected_threshold = selected_threshold, output_directory = output_folder)

        return dens_df, predictive_value_df