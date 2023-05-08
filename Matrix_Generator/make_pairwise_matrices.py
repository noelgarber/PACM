# This script conducts residue-residue pairwise analysis to generate position-aware SLiM matrices and back-calculated scores.

#Import required functions and packages

import numpy as np
import pandas as pd
import os
import pickle

# Declare the sorted list of amino acids
list_aa_no_phos = ["D", "E", "R", "H", "K", "S", "T", "N", "Q", "C", "G", "P", "A", "V", "I", "L", "M", "F", "Y", "W"]
list_aa = ["D", "E", "R", "H", "K", "S", "T", "N", "Q", "C", "G", "P", "A", "V", "I", "L", "M", "F", "Y", "W", "B", "J", "O"] # B=pSer, J=pThr, Y=pTyr

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

#--------------------------------------------------------------------------

#Set minimum members in a category

print("Weighted matrices are calculated relative to a reference position being in a particular chemical class (e.g. acidic, basic, hydrophobic, etc.).")
print("    --> They are based on studying all the peptides following this position-type rule. ")
print("    --> We advise setting a minimum number of peptides here to prevent overfitting.")
print("How many peptides are required before a matrix defaults to using the total list rather than the type-position rule-following subset?")
minimum_members_str = input("Input an integer: ")
minimum_members = int(minimum_members_str)
print("----------------")

#--------------------------------------------------------------------------

#DEFINE GENERALIZED MATRIX FUNCTION

def WeightedMatrix(bait, motif_length, source_dataframe, min_members, amino_acid_list, extreme_thres, high_thres, mid_thres, 
	points_for_extreme, points_for_high, points_for_mid, points_for_low, position_for_filtering = None, residues_included_at_filter_position = ["D", "E", "R", "H", "K", "S", "T", "N", "Q", "C", "G", "P", "A", "V", "I", "L", "M", "F", "Y", "W", "B", "J", "O"]): 

	list_pos = [] #creates list to contain numbered positions across the motif length, to use as column headers in weighted matrices
	for i in range(1, int(motif_length) + 1): 
		list_pos.append("#" + str(i))

	generic_matrix_df = pd.DataFrame(index = amino_acid_list, columns = list_pos)
	generic_matrix_df = generic_matrix_df.fillna(0)

	position_indices = np.arange(0, motif_length)
	if position_for_filtering != None: 
		position_indices_filtered = np.delete(position_indices, position_for_filtering - 1)

	#Default to no filtering if the number of members is below the minimum.

	num_qualifying_entries = 0
	for i in np.arange(len(source_dataframe)): 
		seq = source_dataframe.at[i, "BJO_Sequence"]
		seq = list(seq) #converts to aa list
		if seq[position_for_filtering - 1] in residues_included_at_filter_position: 
			num_qualifying_entries += 1
	if num_qualifying_entries < min_members: 
		residues_included_at_filter_position = amino_acid_list

	#Calculate the points and assign to the matrix.

	for i in np.arange(len(source_dataframe)): 
		seq = source_dataframe.at[i, "BJO_Sequence"]
		seq = list(seq) #converts to aa list

		if bait == "Significant": 
			passes = source_dataframe.at[i, "Significant"]
		elif source_dataframe.at[i, bait + "_Total_Pass"] == "Pass": 
			passes = "Yes"
		elif source_dataframe.at[i, bait + "_Total_Pass"] == "Borderline": 
			passes = "Yes"
		else: 
			passes = "No"
		
		if bait == "Significant": 
			value = source_dataframe.at[i, "Max_Bait_Mean"]
		else: 
			value = 0.5 * (source_dataframe.at[i, bait + "_1"] + source_dataframe.at[i, bait + "_2"])

		if seq[position_for_filtering - 1] in residues_included_at_filter_position: 
			for n in position_indices_filtered: 
				m = n + 1
				for aa in amino_acid_list: 
					if aa == seq[n]: 
						#Calculation of points: 
						if value > extreme_thres: 
							points = points_for_extreme
						elif value > high_thres: 
							points = points_for_high
						elif value > mid_thres: 
							points = points_for_mid
						else: 
							points = points_for_low
						#Pass-conditional point assignment: 
						if passes == "Yes": 
							generic_matrix_df.at[aa, "#" + str(m)] += points
						else: 
							continue
					else: 
						continue

	generic_matrix_df = generic_matrix_df.astype("float32")

	return generic_matrix_df

#--------------------------------------------------------------------------

#Begin position-aware analysis

slim_length = int(input("Enter the length of your SLiM of interest as an integer (e.g. 15):  ")) #This is the length of the motif being analyzed
print("----------------")

#Set the thresholds for the point assignment system

with open(os.path.join(os.getcwd(), "temp", "percentiles_dict.ob"), "rb") as f:
	percentiles_dict = pickle.load(f)

print("Setting thresholds for scoring.")
print("Guidance:")
print("\tUse more points for HIGH signal hits to produce a model that correlates strongly with signal intensity.")
print("\tUse more points for LOW signal hits to produce a model that is better at finding weak positives.")
print("We suggest the 90th, 80th, and 70th percentiles as thresholds, but this may vary depending on the number of hits expected.")

print("---")

use_percentiles = input("Would you like to use percentiles? If not, manually inputted numbers will be used. (Y/N)  ")

if use_percentiles == "Y": 
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

#Set number of points

points_extreme = float(input("How many points for values greater than " + str(thres_extreme) + "? Input:  "))
points_high = float(input("How many points for values greater than " + str(thres_high) + "? Input:  "))
points_mid = float(input("How many points for values greater than " + str(thres_mid) + "? Input:  "))
points_low = float(input("Some hits are marked significant but fall below the signal threshold. How many points for these lower hits? Input:  "))

print("----------------")

print("Inputted point System: ")
print("If max_bait >", thres_extreme, "and passes, points =", points_extreme)
print("If max_bait >", thres_high, "and passes, points =", points_high)
print("If max_bait >", thres_mid, "and passes, points =", points_mid)
print("If max_bait > 0 and passes, points =", points_low)
print("----------------")

#Construction of weighted matrices that are position-aware

print("Constructing position-aware weighted matrices.")
print("----------------")

dictionary_of_matrices = {}

for col_num in range(1, slim_length + 1): 
	for charac, mem_list in aa_charac_dict.items(): 

		weighted_matrix_containing_charac = WeightedMatrix("Significant", slim_length, dens_df, minimum_members, list_aa, thres_extreme, thres_high, thres_mid, points_extreme, points_high, points_mid, points_low, position_for_filtering = col_num, residues_included_at_filter_position = mem_list)
		
		for n in np.arange(1, slim_length + 1): 
			col_name = "#" + str(n)
			max_value = weighted_matrix_containing_charac[col_name].max()
			if max_value == 0: 
				max_value = 1
			for i, row in weighted_matrix_containing_charac.iterrows(): 
				weighted_matrix_containing_charac.at[i, col_name] = weighted_matrix_containing_charac.at[i, col_name] / max_value

		dict_key_name = "#" + str(col_num) + "=" + charac

		dictionary_of_matrices[dict_key_name] = weighted_matrix_containing_charac

#Substitution for always-permitted residues with optional user input

input_always_allowed = input("Would you like to input residues always allowed at certain positions, rather than auto-generating? (Y/N)  ")

always_allowed_dict = {}

for i in np.arange(1, slim_length + 1): 
	position = "#" + str(i)
	if input_always_allowed == "Y": 
		prompt = "Enter comma-delimited residues always allowed at position " + position + ": "
		allowed_str = input(prompt)
	else: 
		allowed_str = ""
	try: 
		allowed_list = allowed_str.split(",")
	except: 
		allowed_list = []
	always_allowed_dict[position] = allowed_list

print("---")

for key, df in dictionary_of_matrices.items(): 
	#Roll values for pS/pT/pY into S/T/Y because the phosphorylation status of a novel sequence cannot be known with certainty.
	for n in np.arange(1, slim_length + 1): 
		df.at["S", "#" + str(n)] = df.at["S", "#" + str(n)] + df.at["B", "#" + str(n)]
		df.at["T", "#" + str(n)] = df.at["T", "#" + str(n)] + df.at["J", "#" + str(n)]
		df.at["Y", "#" + str(n)] = df.at["Y", "#" + str(n)] + df.at["O", "#" + str(n)]
	df.drop(labels = ["B", "J", "O"], axis = 0, inplace = True)

	for i in np.arange(1, slim_length + 1): 
		position = "#" + str(i)
		always_allowed_residues = always_allowed_dict.get(position)
		for residue in always_allowed_residues: 
			df.at[residue, position] = 1

	dictionary_of_matrices[key] = df

#Create position weight dictionary
#Usage: Adjust these values to adjust the relative contribution of each position to the suboptimal element score

print("Creating a dictionary of position weights.")
print("Enter numerical weights for each position based on their expected structural importance. If unknown, use 1.")
pos_weights = {}
for position in np.arange(1, slim_length + 1): 
	pos_weights[position] = float(input("\tEnter weight for position " + str(position) + ":  "))
print("Inputted dict:")
print(pos_weights)

print("-------------------")

print("Constructing dictionary of weighted matrices and save the dataframes by key name.")

dictionary_of_weighted_matrices = {}

matrix_directory = os.path.join(os.getcwd(), "Pairwise_Matrices")
if not os.path.exists(matrix_directory): 
	os.makedirs(matrix_directory) #Makes the directory for outputted weighted matrices

for key, df in dictionary_of_matrices.items(): 
	for i in np.arange(1, slim_length + 1): 
		position = "#" + str(i)
		position_weight = pos_weights.get(i)
		for aa in list_aa_no_phos:
			df.at[aa, position] = df.at[aa, position] * position_weight
	dictionary_of_weighted_matrices[key] = df
	df.to_csv(os.path.join(matrix_directory, key + ".csv"))

print("Done! Saved", str(len(dictionary_of_weighted_matrices)), "matrices to", matrix_directory)

#--------------------------------------------------------------------------

#Begin scoring algorithm construction

dens_scored_df = dens_df.copy()

#Calculate individual residue scores based on neighbouring residues' matching matrices in dictionary_of_weighted_matrices

def CharacAA(amino_acid, dict_of_aa_characs = aa_charac_dict): 
	for charac, mem_list in dict_of_aa_characs.items(): 
		if amino_acid in mem_list: 
			charac_result = charac
	return charac_result

def DynamicAAScorer(index, sequence): 
	output_total_score = 0
	for j in np.arange(1, slim_length + 1): 
		res = sequence[j-1:j]
		res_previous = sequence[j-2:j-1]
		res_subsequent = sequence[j:j+1]
		
		res_previous_position = j - 1
		res_subsequent_position = j + 1

		if res_previous not in list_aa_no_phos: 
			res_previous = res_subsequent #If there is no previous residue (i.e. start of seq), use the subsequent residue twice
			res_previous_position = j + 1

		if res_subsequent not in list_aa_no_phos: 
			res_subsequent = res_previous #If there is no subsequent residue (i.e. end of seq), use the preevious residue twice
			res_subsequent_position = j - 1

		res_previous_charac = CharacAA(res_previous)
		res_subsequent_charac = CharacAA(res_subsequent)

		res_previous_weighted_matrix_key = "#" + str(res_previous_position) + "=" + res_previous_charac
		res_subsequent_weighted_matrix_key = "#" + str(res_subsequent_position) + "=" + res_subsequent_charac

		res_previous_weighted_matrix = dictionary_of_weighted_matrices.get(res_previous_weighted_matrix_key)
		res_subsequent_weighted_matrix = dictionary_of_weighted_matrices.get(res_subsequent_weighted_matrix_key)

		dens_scored_df.at[index, "No_Phos_Res_" + str(j)] = res

		score_previous = res_previous_weighted_matrix.at[res, "#" + str(j)]
		score_subsequent = res_subsequent_weighted_matrix.at[res, "#" + str(j)]

		score_current_position = score_previous + score_subsequent

		output_total_score = output_total_score + score_current_position
	return output_total_score

for i in np.arange(len(dens_scored_df)): 
	seq = dens_scored_df.at[i, "No_Phos_Sequence"]
	total_score = DynamicAAScorer(i, seq)
	dens_scored_df.at[i, "SLiM_Score"] = total_score

print("Produced edited densitometry data file containing scores for each peptide.")

#--------------------------------------------------------------------------

#User selection of threshold for calling hits as TP/FP/TN/FN

#Define function that divides numbers and returns a code value for divide_by_zero errors
def divide_inf(numerator, denominator, infinity_value = 999): 
	if denominator == 0: 
		value = inf_value
	else: 
		value = numerator / denominator
	return value

def diagnostic_value(score_threshold, dataframe, significance_col = "Significant", score_col = "SLiM_Score"): 
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

min_score = dens_scored_df["SLiM_Score"].min()
max_score = dens_scored_df["SLiM_Score"].max()
score_range_series = np.linspace(min_score, max_score, num = 100)

threshold_selection_dict = {}

for i in score_range_series: 
	i_rounded = round(i, 1)
	pv_dict = diagnostic_value(i, dens_scored_df)
	ppv_npv_list = [pv_dict.get("PPV"), pv_dict.get("NPV")]
	threshold_selection_dict[i_rounded] = ppv_npv_list

print("Threshold selection information:")

for key, value in threshold_selection_dict.items(): 
	print(key, ":", "PPV =", value[0], "\tNPV =", value[1], "\tFDR =", round(1 - value[0], 3), "\tFOR =", round(1 - value[1], 3))

print("-------------------")

selected_threshold = float(input("Input your selected threshold for calling hits:  "))

for i in np.arange(len(dens_scored_df)): 
	current_score = dens_scored_df.at[i, "SLiM_Score"]
	sig_y_n = dens_scored_df.at[i, "Significant"]
	if current_score >= selected_threshold: 
		dens_scored_df.at[i, "Call"] = "Positive"
		if sig_y_n == "Yes":
			call_type = "TP"
		else: 
			call_type = "FP"
	else: 
		dens_scored_df.at[i, "Call"] = "-"
		if sig_y_n == "Yes": 
			call_type = "FN"
		else: 
			call_type = "TN"
	dens_scored_df.at[i, "Call_Type"] = call_type

print("Applied hit calls based on threshold.")

#Save to output folder

output_filename = "pairwise_scored_data" + "_thres" + str(selected_threshold) + ".csv"
output_file_path = os.path.join(os.getcwd(), "Array_Output", output_filename)
dens_scored_df.to_csv(output_file_path)

with open(os.path.join(os.getcwd(), "temp", "pairwise_results_filename.ob"), "wb") as f:
	pickle.dump(output_file_path, f)

print("Saved! Filename:", output_file_path)
print("-------------------")



def make_pairwise_matrices(dens_df, list_of_baits, control_bait_name):
	'''
	Main function for making pairwise position-weighted matrices

	Args:
		dens_df (pd.DataFrame): the dataframe containing densitometry values for the peptides being analyzed
		list_of_baits (list): the list of bait proteins, not including the control bait
		control_bait_name (str): the name of the control bait (e.g. "Secondary-only")

	Returns:

	'''



