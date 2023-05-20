# This script conducts residue-residue pairwise analysis to generate position-aware SLiM matrices and back-calculated scores.

#Import required functions and packages

import numpy as np
import pandas as pd
import os
import pickle
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
					bjo_seq_col = "BJO_sequence", signal_col_suffix = "Background-Adjusted_Standardized_Signal"):
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
		seq = list(seq) # converts to aa list
		if seq[position_for_filtering - 1] in residues_included_at_filter_position: 
			num_qualifying_entries += 1
	if num_qualifying_entries < min_members: 
		residues_included_at_filter_position = amino_acid_list

	# Calculate the points and assign to the matrix.
	for i in np.arange(len(source_dataframe)): 
		seq = source_dataframe.at[i, "BJO_Sequence"]
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
				for aa in amino_acid_list: 
					if aa == seq[n]: 
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
						   sequence_col = "BJO_sequence", signal_col_suffix = "Background-Adjusted_Standardized_Signal"):
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

def get_always_allowed():
	'''
	Simple function to get a user-inputted dict of position # --> list of residues that are always permitted at that position

	Args:
		None

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

def add_matrix_weights(matrices_dict, position_weights, amino_acids = amino_acids):
	'''
	Function to apply the matrix weights by position to the generated matrices

	Args:
		matrices_dict (dict): dictionary of type-position rule --> unadjusted position-weighted matrix

	Returns:
		weighted_matrices_dict (dict): same as matrices_dict, but with the weights applied to the matrix values
	'''
	weighted_matrices_dict = {}

	for key, df in matrices_dict.items():
		for i in np.arange(1, slim_length + 1):
			position = "#" + str(i)
			position_weight = position_weights.get(i)
			for aa in amino_acids:
				df.at[aa, position] = df.at[aa, position] * position_weight
		weighted_matrices_dict[key] = df

	return weighted_matrices_dict


def save_weighted_matrices(weighted_matrices_dict, matrix_directory = None):
	'''
	Simple function to save the weighted matrices to disk

	Args:
		weighted_matrices_dict (dict): the dictionary of type-position rule --> corresponding weighted matrix
		matrix_directory (str): directory to save matrices into; defaults to a subfolder called Pairwise_Matrices

	Returns:
		None
	'''
	if matrix_directory = None:
		matrix_directory = os.path.join(os.getcwd(), "Pairwise_Matrices")

	# If the matrix directory does not exist, make it
	if not os.path.exists(matrix_directory):
		os.makedirs(matrix_directory)

	# Save matrices by key name as CSV files
	for key, df in weighted_matrices_dict.items():
		df.to_csv(os.path.join(matrix_directory, key + ".csv"))

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
	output_total_score = 0
	for j in np.arange(1, slim_length + 1): 
		res = sequence[j-1:j]
		res_previous = sequence[j-2:j-1]
		res_subsequent = sequence[j:j+1]
		
		res_previous_position = j - 1
		res_subsequent_position = j + 1

		if res_previous not in amino_acids:
			res_previous = res_subsequent #If there is no previous residue (i.e. start of seq), use the subsequent residue twice
			res_previous_position = j + 1

		if res_subsequent not in amino_acids:
			res_subsequent = res_previous #If there is no subsequent residue (i.e. end of seq), use the preevious residue twice
			res_subsequent_position = j - 1

		res_previous_charac = aa_chemical_class(res_previous)
		res_subsequent_charac = aa_chemical_class(res_subsequent)

		res_previous_weighted_matrix_key = "#" + str(res_previous_position) + "=" + res_previous_charac
		res_subsequent_weighted_matrix_key = "#" + str(res_subsequent_position) + "=" + res_subsequent_charac

		res_previous_weighted_matrix = weighted_matrices.get(res_previous_weighted_matrix_key)
		res_subsequent_weighted_matrix = weighted_matrices.get(res_subsequent_weighted_matrix_key)

		if add_residue_cols:
			dens_df.at[df_row_index, "Residue_" + str(j)] = res

		score_previous = res_previous_weighted_matrix.at[res, "#" + str(j)]
		score_subsequent = res_subsequent_weighted_matrix.at[res, "#" + str(j)]

		score_current_position = score_previous + score_subsequent

		output_total_score = output_total_score + score_current_position

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
	for i in np.arange(len(output_df)):
		seq = dens_scored_df.at[i, seq_col]
		total_score = score_aa_seq(sequence = seq, weighted_matrices = weighted_matrices, slim_length = slim_length,
								   dens_df = output_df, df_row_index = i, add_residue_cols = add_residue_cols)
		output_df.at[i, score_col] = total_score

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

def divide_inf(numerator, denominator, infinity_value = 999):
	'''
	Simple division function that substitutes a specified infinity value when divide-by-zero errors are encountered

	Args:
		numerator (float): 		the numerator for division
		denominator (float): 	the denominator for division
		infinity_value (float): the code value to substitute when divide-by-zero errors are encountered; default is 999

	Returns:
		value (float): 			the result of the division
	'''
	if denominator == 0: 
		value = infinity_value
	else: 
		value = numerator / denominator

	return value

def diagnostic_value(score_threshold, dataframe, significance_col = "One_Passes", score_col = "SLiM_Score"):
	'''
	Function to produce sensitivity, specificity, and positive and negative predictive values for a given score cutoff

	Args:
		score_threshold (float):  the score cutoff to use
		dataframe (pd.DataFrame): the dataframe containing peptide information
		significance_col (str):   the column containing significance information
		score_col (str):          the column containing the back-calculated motif score for the peptides

	Returns:
		pred_val_dict (dict):     dictionary with values for keys of "Sensitivity", "Specificity", "PPV", and "NPV"
	'''
	pred_val_dict = {
		"TP": 0,
		"FP": 0,
		"TN": 0,
		"FN": 0,
	}

	for i in np.arange(len(dataframe)): 
		sig_truth = dataframe.at[i, significance_col]
		score = dataframe.at[i, score_col]
		if score >= score_threshold: 
			score_over_n = "Yes"
		else: 
			score_over_n = "No"

		if score_over_n == "Yes" and sig_truth == "Yes": 
			pred_val_dict["TP"] = pred_val_dict["TP"] + 1
		elif score_over_n == "Yes" and sig_truth == "No": 
			pred_val_dict["FP"] = pred_val_dict["FP"] + 1
		elif score_over_n != "Yes" and sig_truth == "Yes": 
			pred_val_dict["FN"] = pred_val_dict["FN"] + 1
		elif score_over_n != "Yes" and sig_truth == "No": 
			pred_val_dict["TN"] = pred_val_dict["TN"] + 1

	pred_val_dict["Sensitivity"] = round(divide_inf(numerator = pred_val_dict.get("TP"), denominator = pred_val_dict.get("TP") + pred_val_dict.get("FN")), 3)
	pred_val_dict["Specificity"] = round(divide_inf(numerator = pred_val_dict.get("TN"), denominator = pred_val_dict.get("TN") + pred_val_dict.get("FP")), 3)
	pred_val_dict["PPV"] = round(divide_inf(numerator = pred_val_dict.get("TP"), denominator = pred_val_dict.get("TP") + pred_val_dict.get("FP")), 3)
	pred_val_dict["NPV"] = round(divide_inf(numerator = pred_val_dict.get("TN"), denominator = pred_val_dict.get("TN") + pred_val_dict.get("FN")), 3)

	return pred_val_dict

def apply_threshold(input_df, sig_col = "One_Passes", score_col = "SLiM_Score", range_count = 100):
	'''
	Function to declare and apply the motif score threshold based on predictive values

	Args:
		input_df (pd.DataFrame):   the dataframe containing peptides and scores
		sig_col (str): 			  the df column containing significance information (Yes/No)
		score_col (str): 		  the df column containing the peptide scores

	Returns:
		output_df (pd.DataFrame):              dens_df with a new column containing calls based on the selected score
		selected_threshold (float):            the selected score threshold
	'''
	output_df = input_df.copy()

	# Make a range of SLiM scores between the minimum and maximum score values from the dataframe
	min_score = output_df[score_col].min()
	max_score = output_df[score_col].max()
	score_range_series = np.linspace(min_score, max_score, num = range_count)

	# Assemble a dictionary of score --> [positive predictive value, negative predictive value]
	threshold_selection_dict = {}
	for i in score_range_series:
		i_rounded = round(i, 1)
		pv_dict = diagnostic_value(score_threshold = i, dataframe = output_df,
								   significance_col = "One_Passes", score_col = score_col)
		ppv_npv_list = [pv_dict.get("PPV"), pv_dict.get("NPV")]
		threshold_selection_dict[i_rounded] = ppv_npv_list

	# Print the dictionary to aid in the user selecting an appropriate score cutoff for significance to be declared
	print("Threshold selection information:", "\n---")
	for key, value in threshold_selection_dict.items():
		print(key, ":", "PPV =", value[0], "\tNPV =", value[1], "\tFDR =", round(1 - value[0], 3), "\tFOR =", round(1 - value[1], 3))
	print("---")

	# Prompt the user to input the selected threshold that a SLiM score must exceed to be considered significant
	selected_threshold = input_number("Input your selected threshold for calling hits:  ", "float")

	# Apply the selected threshold to the dataframe to call hits
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

	return output_df, selected_threshold

def save_scored_data(selected_threshold, output_directory, verbose = True):
	output_filename = "pairwise_scored_data" + "_thres" + str(selected_threshold) + ".csv"
	output_file_path = os.path.join(output_directory, "Array_Output", output_filename)
	dens_scored_df.to_csv(output_file_path)

	print("Saved! Filename:", output_file_path) if verbose else None

	return output_file_path


def make_pairwise_matrices(dens_df, percentiles_dict = None, slim_length = None, minimum_members = None,
						   thres_tuple = None, points_tuple = None, always_allowed_dict = None, position_weights = None,
						   output_folder = None, sequence_col = "No_Phos_Sequence", significance_col = "One_Passes"):
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

	Returns:

	'''

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
	matrices_dict = make_weighted_matrices(slim_length = slim_length, aa_charac_dict = aa_charac_dict,
										   dens_df = dens_df, minimum_members = minimum_members, list_aa = amino_acids_phos,
										   thres_tuple = thres_tuple, points_tuple = points_tuple,
										   sequence_col = "BJO_sequence",
										   signal_col_suffix = "Background-Adjusted_Standardized_Signal")

	# Get list of always-allowed residues (overrides algorithm for those positions)
	if always_allowed_dict is None:
		always_allowed_dict = get_always_allowed()

	# Collapse phospho-residues into non-phospho counterparts and apply always-allowed residues
	matrices_dict = collapse_phospho(matrices_dict = matrices_dict, slim_length = slim_length)
	matrices_dict = apply_always_allowed(matrices_dict = matrices_dict, slim_length = slim_length,
										 always_allowed_dict = always_allowed_dict)

	# Get weights for positions along the motif sequence
	if position_weights is None:
		position_weights = get_position_weights(slim_length = slim_length)

	# Apply weights and save the weighted matrices
	weighted_matrices_dict = add_matrix_weights(matrices_dict = matrices_dict, position_weights = position_weights,
												amino_acids = amino_acids)
	if output_folder is None:
		output_folder = os.getcwd()
	matrix_output_folder = os.path.join(output_folder, "Pairwise_Matrices")
	save_weighted_matrices(weighted_matrices_dict = weighted_matrices_dict, matrix_directory = matrix_output_folder)

	# Apply the motif scoring algorithm back onto the peptide sequences
	dens_df = apply_motif_scores(dens_df = dens_df, weighted_matrices = weighted_matrices_dict,
								 slim_length = slim_length, seq_col = sequence_col, score_col = "SLiM_Score",
					   			 add_residue_cols = True)

	# Use thresholding to declare true/false positives/negatives in the peptide sequences
	dens_df, selected_threshold = apply_threshold(dens_df, sig_col = significance_col, score_col = "SLiM_Score")

	# Save the results
	save_scored_data(selected_threshold = selected_threshold, output_directory = output_folder)

	return dens_df