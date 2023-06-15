#This script constructs a singular weighted matrix to predict bait-bait specificity in SLiM sequences.

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from general_utils.general_utils import input_number, list_inputter, permute_weights, save_dict

#Declare general variables
amino_acids = ("D", "E", "R", "H", "K", "S", "T", "N", "Q", "C", "G", "P", "A", "V", "I", "L", "M", "F", "Y", "W")
amino_acids_phos = ("D", "E", "R", "H", "K", "S", "T", "N", "Q", "C", "G", "P", "A", "V", "I", "L", "M", "F", "Y", "W", "B", "J", "O") # B=pSer, J=pThr, Y=pTyr

def get_comparator_baits():
	'''
	Function to prompt the user for the sets of baits to compare; allows pooling of baits in the binary comparison

	Returns:
		comparator_set_1 (list): the first set of comparator baits
		comparator_set_2 (list): the second set of comparator baits
	'''

	comparator_set_1 = list_inputter("For comparator bait set #1, input the baits one at a time:")
	comparator_set_2 = list_inputter("For comparator bait set #2, input the baits one at a time:")

	return comparator_set_1, comparator_set_2

#print("There are two options for scoring method:")
#print("    Option 1 --> Score to predict hits specific to Comparator 1 vs. hits specific to Comparator 2")
#print("    Option 2 --> Score to predict hits specific to Comparator 1 vs. all hits that bind Comparator 2, either equally or specifically")
valid_input = False
while not valid_input: 
	points_handling = NumInput("Which option? Select 1 or 2:")
	if points_handling == 1 or points_handling == 2: 
		valid_input = True

#--------------------------------------------------------------------------

def least_different(index, lookup_df, comparator_set_1 = None, comparator_set_2 = None):
	'''
	Simple function to determine the least different log2fc between permutations of the sets of comparators

	Args:
		index (int): 			   the row index for the lookup dataframe
		lookup_df (pd.DataFrame):  the dataframe to look up values within
		comparator_set_1 (list):   the first set of comparator baits
		comparator_set_2 (list):   the second set of comparator baits

	Returns:
		least_diff_log2fc (float): the log2fc value for the least different pair of baits, one from each comparator set
		least_diff_baits (tuple):  a tuple of (bait1, bait2) corresponding to least_diff_log2fc
	'''

	if comparator_set_1 is None or comparator_set_2 is None:
		comparator_set_1, comparator_set_2 = get_comparator_baits()

	least_diff_log2fc = 9999
	least_diff_baits = ()

	for bait1 in comparator_set_1:
		for bait2 in comparator_set_2:
			if bait1 != bait2:
				b1_b2_log2fc = lookup_df.at[index, bait1 + "_" + bait2 + "_log2fc"]
				if abs(b1_b2_log2fc) < abs(least_diff_log2fc): 
					least_diff_log2fc = b1_b2_log2fc
					least_diff_baits = (bait1, bait2)

	return least_diff_log2fc, least_diff_baits

def bias_ratio(source_df, thresholds = (1.0,, -1.0), passes_col = "Significant", pass_str = "Yes",
			   comparator_set_1 = None, comparator_set_2 = None):
	'''
	Function for finding the ratio of entries in the dataframe specific to one bait set vs. the other bait set

	Args:
		source_df (pd.DataFrame):  the source dataframe containing sequences, log2fc data, and significance calls
		thresholds (tuple): 	   tuple of floats as (positive_thres, negative_thres)
		passes_col (str): 		   the column in source_df containing pass/fail information (significance calls)
		pass_str (str): 		   the string that indicates a pass, e.g. "Yes"
		comparator_set_1 (list):   the first set of comparator baits
		comparator_set_2 (list):   the second set of comparator baits

	Returns:
		ratio (float):			   the ratio of entries above pos_thres to entries below neg_thres
	'''

	positive_thres, negative_thres = thresholds

	above_thres_count = 0
	below_neg_thres_count = 0

	for i in np.arange(len(source_df)): 
		passes = source_df.at[i, passes_col]

		bait1_bait2_log2fc, bait1_bait2_list = least_different(i, source_df, comparator_set_1, comparator_set_2)

		if passes == pass_str and bait1_bait2_log2fc >= positive_thres:
			above_thres_count += 1
		elif passes == pass_str and bait1_bait2_log2fc <= negative_thres:
			below_neg_thres_count += 1

	if below_neg_thres_count == 0:
		# Avoids divide-by-zero errors by increasing both values by 1
		below_neg_thres_count += 1
		above_thres_count += 1

	ratio = above_thres_count / below_neg_thres_count

	return ratio

def assign_points(matrix_df, sequence, seq_log2fc, significant, bias_ratio = 1, pass_str = "Yes", thresholds,
				  mode = "discrete", include_phospho = False):
	'''
	Function for assigning points based on sequences and their log2fc values

		matrix_df:				destination dataframe for applying points in-place
		sequence (str):			the current amino acid sequence to determine points for
		seq_log2fc (float):		the log2fc value for the current sequence
		significant (str):		a string representing whether an entry is significant
		bias_ratio (float):		ratio of data specific to the first set of comparators to those specific to the second set
		pass_str (str): 		the string value denoting significance, e.g. "Yes"
		thresholds (tuple): 	(upper_positive_thres, middle_positive_thres, middle_negative_thres, upper_negative_thres)
		mode (str): 			"discrete" (uses thresholds) or "continuous" (uses log2fc values directly)
		include_phospho (bool):	whether to include phospho-residues when applying points; if False, will pool with non-phospho counterparts

	Returns:
		None; operations are performed in-place
	'''

	upper_positive_thres, middle_positive_thres, middle_negative_thres, upper_negative_thres = thresholds

	row_amino_acids = amino_acids_phos
	phos_amino_acids = ["B", "J", "O"]

	if significant == pass_str:
		for i, aa in enumerate(sequence):
			# Assignment of points
			if mode == "continuous":
				points = seq_log2fc
			elif mode == "discrete":
				if seq_log2fc >= upper_positive_thres:
					points = 2
				elif seq_log2fc >= middle_positive_thres:
					points = 1
				elif seq_log2fc <= middle_negative_thres:
					points = -1 * bias_ratio
				elif seq_log2fc <= upper_negative_thres:
					points = -2 * bias_ratio
				else:
					points = 0
			else:
				raise Exception(f"assign_points mode was set to {mode}, but must be either \"continuous\" or \"discrete\"")

			# Declare whether the aa is valid and whether it is phosphorylated
			valid_aa = aa in row_amino_acids
			phos_aa = aa in phos_amino_acids

			# Assignment of points
			if valid_aa and include_phospho:
				matrix_df.at[aa, "#"+str(i+1)] += points
			elif valid_aa and not phos_aa:
				matrix_df.at[aa, "#"+str(i+1)] += points
			elif valid_aa:
				if aa == "B":
					matrix_df.at["S", "#"+str(i+1)] += points
				elif aa == "J":
					matrix_df.at["T", "#"+str(i+1)] += points
				elif aa == "O":
					matrix_df.at["Y", "#"+str(i+1)] += points

def get_specificity_scores(sequences, log2fc_values, specificity_matrix_df, position_weights = None):
	'''
	Function to back-calculate specificity scores on peptide sequences based on the generated specificity matrix

	Args:
		sequences (array-like): 				the sequences to score, containing strings as their values
		log2fc_values (np.ndarray): 			the log2fc values for the sequences
		specificity_matrix_df (pd.DataFrame): 	the generated specificity matrix dataframe for getting residue values
		position_weights (np.ndarray): 			an array of weights for the columns in the matrix dataframe

	Returns:
		score_values (np.ndarray): 					the specificity scores for the given sequences
		weighted_specificity_matrix (pd.DataFrame):	the specificity matrix with weights applied
		r2 (float): 								the r-squared value for the linear association between log2fc values and scores
	'''

	# Get position weights
	if position_weights is None:
		column_count = len(specificity_matrix_df.columns)
		position_weights = np.ones(column_count, dtype=int)

	# Apply position weights
	if len(position_weights) == len(specificity_matrix_df.columns):
		weighted_specificity_matrix = specificity_matrix_df.mul(position_weights, axis=1)
	else:
		raise ValueError(f"get_specificity_scores received position_weights of length {len(position_weights)}, which does not match the number of matrix columns ({len(specificity_matrix_df.columns)})")

	# Back-calculate specificity scores on the source data
	score_values = []
	for i in np.arange(len(sequences)):
		seq = sequences[i]

		specificity_score = 0
		for j, aa in enumerate(seq):
			matrix_column = "#" + str(j + 1)
			residue_score = weighted_specificity_matrix.at[aa, matrix_column]
			specificity_score += residue_score

		score_values.append(specificity_score)

	score_values = np.array(score_values)

	# Perform linear regression between log2fc values and scores
	model = LinearRegression()
	x_actual = score_values.reshape(-1, 1)
	y_actual = log2fc_values.reshape(-1, 1)
	model.fit(x_actual, y_actual)
	y_pred = model.predict(x_actual)
	r2 = r2_score(y_actual, y_pred)

	return score_values, weighted_specificity_matrix, r2

def get_position_copies():
	# Simple function to get a dict where the sum of the values is equal to the length of the motif being studied

	print("Please enter the list of copies for each position when permuting weights; sum must be equal to motif length")
	copies_list = list_inputter("Next value:  ")

	copies_dict = {}
	for i, copies_value in enumerate(copies_list):
		copies_dict[i]: copies_value

	return copies_dict

def specificity_weighted_matrix(source_dataframe, pass_str, comparator_set_1, comparator_set_2, thresholds,
								sequence_col = "BJO_Sequence", significance_col = "One_Passes", mode = "discrete",
								include_phospho = False, position_weights = None, motif_length = None, 
								position_copies = None, optimize_weights = False, verbose = True):
	'''
	Main function for building the specificity position-weighted matrix and back-calculating scores onto source data

	Args:
		source_dataframe (pd.DataFrame): the dataframe containing entries with log2fc values and peptide sequences
		pass_str (str): 				 a string representing significance, e.g. "Yes" or "Significant"
		comparator_set_1 (list): 		 the first set of comparator baits
		comparator_set_2 (list): 		 the second set of comparator baits
		thresholds (tuple): 			 a tuple of log2fc thresholds as (upper_positive_thres, middle_positive_thres,
										 middle_negative_thres, upper_negative_thres)
		sequence_col (str): 			 the name of the column containing peptide sequences
		significance_col (str): 		 the name of the column containing significance information
		mode (str): 					 "discrete" (uses thresholds) or "continuous" (uses log2fc values directly)
		include_phospho (bool):			 whether to include phospho-residues when applying points; if False, will pool with non-phospho counterparts
		position_weights (np.ndarray): 	 an array of weights for the columns in the matrix dataframe
		motif_length (int): 			 the length of the motif being studied
        position_copies (dict): 		 a dictionary of position --> copy_num, where the sum of dict values equals slim_length
		optimize_weights (bool): 		 whether to find optimal weights for the weighted matrix, compared to supplying them upfront
		verbose (bool): 				 whether to display additional information

	Returns:
		specificity_matrix_df (pd.DataFrame): 	the specificity position-weighted matrix
		output_df (pd.DataFrame): 				the updated source dataframe containing least different log2fc values
	'''

	# If optimizing weights, get position copies for permuting weights if none were given
	if position_copies is None and optimize_weights:
		position_copies = get_position_copies()

	output_df = source_dataframe.copy()

	# Extract threshold values; note that you can compare hits specific to comparator_set_1 against hits binding both comparator sets by setting negative thresholds to zero, and vice versa
	upper_positive_thres, middle_positive_thres, middle_negative_thres, upper_negative_thres = thresholds

	# Make an empty matrix that will become the specificity position-weighted matrix
	unweighted_specificity_matrix = pd.DataFrame()

	# Calculate an adjustment ratio to correct for inequalities between counts of hits specific to each comparator set
	bias_thresholds = (middle_positive_thres, middle_negative_thres)
	bias_ratio_value = bias_ratio(output_df, bias_thresholds, significance_col, pass_str, comparator_set_1, comparator_set_2)
	print(f"Bias ratio: {bias_ratio_value}") if verbose else None

	#Score the sequences
	sequences = output_df[sequence_col].values
	sig_calls = output_df[significance_col].values
	log2fc_values = []
	least_different_pairs = []
	for i in np.arange(len(sequences)):
		# Get sequence and significance values
		seq = sequences[i]
		passes = sig_calls[i]

		# Get log2fc values for the least different pair of baits, one from each comparator set
		least_diff_log2fc, least_diff_baits = least_different(i, output_df, comparator_set_1, comparator_set_2)
		log2fc_values.append(least_diff_log2fc)

		pair_separator = ", "
		least_different_pairs.append(least_diff_baits[0] + pair_separator + least_different_baits[1])

		# Apply points in-place to relevant rows and columns in specificity_matrix_df
		assign_points(unweighted_specificity_matrix, seq, least_diff_log2fc, passes, bias_ratio_value, pass_str, thresholds, mode, include_phospho)

	# Assign the log2fc values and source baits to the dataframe
	output_df["Least_Different_Log2fc"] = log2fc_values
	output_df["Least_Different_Baits"] = least_different_pairs

	# Fill NaN matrix values with 0 and convert to float32
	unweighted_specificity_matrix = unweighted_specificity_matrix.fillna(0)
	unweighted_specificity_matrix = unweighted_specificity_matrix.astype("float32")

	# Back-calculate specificity scores on the source data, and get R2 for the linear association with log2fc values
	log2fc_values = np.array(log2fc_values)

	if optimize_weights:
		permuted_weights = permute_weights(motif_length, position_copies)
		best_r2 = 0
		best_weights = None
		for weights in permuted_weights: 
			_, _, r2 = get_specificity_scores(sequences, log2fc_values, unweighted_specificity_matrix, weights)
			if r2 > best_r2: 
				best_r2 = r2
				best_weights = weights
		specificity_scores, weighted_specificity_matrix, r2 = get_specificity_scores(sequences, log2fc_values, unweighted_specificity_matrix, best_weights)
		final_weights = best_weights
	else: 
		specificity_scores, weighted_specificity_matrix, r2 = get_specificity_scores(sequences, log2fc_values, unweighted_specificity_matrix, position_weights)
		final_weights = position_weights

	output_df["Specificity_Score"] = specificity_scores

	return (unweighted_specificity_matrix, weighted_specificity_matrix, output_df, final_weights, r2)

# ---------------------------------------------------------------------------------------------------------------------

def main(source_dataframe, pass_str = "Yes", sequence_col = "BJO_Sequence", significance_col = "One_Passes",
		 points_mode = "discrete", comparator_set_1 = None, comparator_set_2 = None, thresholds = None,
		 include_phospho = False, position_weights = None, motif_length = None, position_copies = None,
		 optimize_weights = False, save_data = True, output_folder = None, verbose = True):

	# Define groups of baits (or single baits) to compare
	if comparator_set_1 is None or comparator_set_2 is None:
		comparator_set_1, comparator_set_2 = get_comparator_baits()

	# Define the thresholds to use for making calls about log2fc values
	if thresholds is None:
		upper_positive_thres = input_number("Enter the upper positive threshold (e.g. 1.0):  ", "float")
		middle_positive_thres = input_number("Enter the middle positive threshold (e.g. 0.5):  ", "float")
		middle_negative_thres = input_number("Enter the middle negative threshold (e.g. -0.5):  ", "float")
		upper_negative_thres = input_number("Enter the upper negative threshold (e.g. -1.0):  ", "float")
		thresholds = (upper_positive_thres, middle_positive_thres, middle_negative_thres, upper_negative_thres)

	# If optimizing weights, get permutation parameters
	if position_copies is None and optimize_weights:
		position_copies = get_position_copies()

	# Make the weighted matrix, calculate scores, and calculate r2 for the linear correlation between scores vs. log2fc
	matrix_results_tuple = specificity_weighted_matrix(source_dataframe, pass_str, comparator_set_1, comparator_set_2,
													   thresholds, sequence_col, significance_col, points_mode,
													   include_phospho, position_weights, motif_length, position_copies,
													   optimize_weights, verbose)
	unweighted_specificity_matrix, weighted_specificity_matrix, output_df, final_weights, r2 = matrix_results_tuple

	# Save the data
	if output_folder is None:
		output_folder = os.getcwd()

	if save_data:
		unweighted_specificity_matrix.to_csv(os.path.join(output_folder, "unweighted_specificity_matrix.csv"))
		weighted_specificity_matrix.to_csv(os.path.join(output_folder, "weighted_specificity_matrix.csv"))
		output_df.to_csv(os.path.join(output_folder, "specificity_scored_data.csv"))

		output_dict = {"R2": r2, "Position weights": final_weights}
		save_dict(output_dict, output_folder, "specificity_matrix_info.txt")

	# Return the results tuple, which can be optionally assigned if this module is imported to another script

	return matrix_results_tuple

if __name__ == "__main__":
	source_dataframe_path = input("Please enter the path to the source data as a csv file:  ")
	source_dataframe = pd.read_csv(source_dataframe_path)

	sequence_col = input("Enter the sequence column name, or leave blank for default:  ")
	if sequence_col == "":
		sequence_col = None

	significance_col = input("Enter the significance column name, or leave blank for default:  ")
	if significance_col == "":
		significance_col = None

	valid_mode = False
	while not valid_mode:
		points_mode = input("Use discrete or continuous application of points? ")
		if points_mode == "discrete" or points_mode == "continuous":
			valid_mode = True
		else:
			print("\tInvalid entry, please enter either \"discrete\" or \"continuous\"")

	main(source_dataframe, sequence_col = sequence_col, significance_col = significance_col, points_mode = points_mode)