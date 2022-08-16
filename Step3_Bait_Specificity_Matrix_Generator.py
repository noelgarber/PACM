#This script constructs a weighted matrix to predict bait-bait specificity in SLiM sequences. 

#Import required functions and packages

import numpy as np
import pandas as pd
import os
import pickle
from PACM_General_Functions import FilenameSubdir, NumInput, ListInputter
from PACM_General_Vars import list_aa, list_aa_no_phos

print("----------------")
print("Script 3: ")
print("This script generates SLiM bait specificity scores based on inputted baits. ")
print("IMPORTANT: Specificity scores are, in our experience, only interpretable if the underlying SLiM score passes a defined threshold; they cannot be interpreted in isolation.")
print("----------------")

with open("Step2_Results_Filename.ob", "rb") as f:
	filename = pickle.load(f)
use_last_generated = input("The last generated file from Step 2 was " + filename + ". Would you like to use a different filename? (Y/N)  ")
if use_last_generated == "Y": 
	filename = input("Enter the filename containing pairwise scored densitometry data, located in /Output:  ")

dens_final_df = pd.read_csv(FilenameSubdir("Output", filename), index_col = 0)

with open("list_of_baits.ob", "rb") as lob:
	list_of_baits = pickle.load(lob)

#--------------------------------------------------------------------------

#Begin matrix building

slim_length = 15 #This is the length of the motif being analyzed
positions = np.arange(1, slim_length + 1)
print("This script allows pooling of similar baits; for example, specificity between Protein A and Proteins B & C.")

def CheckPooling(bait_number): 
	print("Setting bait(s) for use as Comparator #" + str(bait_number))
	bait_pool_list = ListInputter("Input the baits one at a time.")
	return bait_pool_list

bait1_inputted = CheckPooling(1)
bait2_inputted = CheckPooling(2)
print("There are two options for scoring method:")
print("    Option 1 --> Score to predict hits specific to Comparator 1 vs. hits specific to Comparator 2")
print("    Option 2 --> Score to predict hits specific to Comparator 1 vs. all hits that bind Comparator 2, either equally or specifically")

valid_input = False
while not valid_input: 
	points_handling = NumInput("Which option? Select 1 or 2:")
	if points_handling == 1 or points_handling == 2: 
		valid_input = True

#--------------------------------------------------------------------------

def LeastDiffFold(b1_list, b2_list, index, lookup_df): 
	least_diff_log2fc = 9999
	least_diff_bait_combo = []
	for b1 in b1_list: 
		for b2 in b2_list: 
			if b1 != b2: 
				b1_b2_log2fc = lookup_df.at[index, b1 + "_" + b2 + "_log2fc"]
				if abs(b1_b2_log2fc) < abs(least_diff_log2fc): 
					least_diff_log2fc = b1_b2_log2fc
					least_diff_bait_combo = [b1, b2]
	return least_diff_log2fc, least_diff_bait_combo

def SpecRatioBias(bait1_pool, bait2_pool, source_df, pos_thres, neg_thres, passes_col = "Significant"): 
	above_thres_count = 0
	below_neg_thres_count = 0

	for i in np.arange(len(source_df)): 
		passes = source_df.at[i, passes_col]

		bait1_bait2_log2fc, bait1_bait2_list = LeastDiffFold(bait1_pool, bait2_pool, i, source_df)

		if passes == "Yes" and bait1_bait2_log2fc >= pos_thres: 
			above_thres_count += 1
		elif passes == "Yes" and bait1_bait2_log2fc <= neg_thres: 
			below_neg_thres_count += 1
	if below_neg_thres_count != 0: 
		ratio = above_thres_count / below_neg_thres_count
	else: 
		ratio = 100 #Arbitrary value; would be infinity
	return ratio

def PointsFinder(output_df, hthres, mthres, bias_ratio, sequence, seq_length, seq_log2fc, pass_call, aalist): 
	#print(output_df)
	indices = np.arange(seq_length)
	if hthres == "Continuous" or mthres == "Continuous": 
		for n in indices: 
			m = n + 1
			for aa in aalist: 
				if aa == sequence[n]: 
					points = seq_log2fc
					if pass_call == "Yes": 
						output_df.at[aa, "#" + str(m)] += points
	elif points_handling == 1: 
		for n in indices: 
			m = n + 1
			for aa in aalist: 
				if aa == sequence[n]: 
					#Calculation of points: 
					if seq_log2fc >= hthres: 
						points = 2
					elif seq_log2fc >= mthres: 
						points = 1
					elif seq_log2fc <= (mthres * -1): 
						points = -1 * bias_ratio
					elif seq_log2fc <= (hthres * -1): 
						points = -1 * bias_ratio
					else: 
						points = 0
					#Pass-conditional point assignment: 
					if pass_call == "Yes": 
						output_df.at[aa, "#" + str(m)] += points
					else: 
						continue
	elif points_handling == 2: 
		for n in indices: 
			m = n + 1
			for aa in aalist: 
				if aa == sequence[n]: 
					#Calculation of points: 
					if seq_log2fc >= hthres: 
						points = 2
					elif seq_log2fc >= mthres: 
						points = 1
					elif seq_log2fc <= (mthres * -1): 
						points = -2 * bias_ratio
					elif seq_log2fc <= 0: 
						points = -1 * bias_ratio
					else: 
						points = 0
					#Pass-conditional point assignment: 
					if pass_call == "Yes": 
						output_df.at[aa, "#" + str(m)] += points
					else: 
						continue

def SpecWeightedMatrix(motif_length, bait1_list, bait2_list, source_dataframe, amino_acid_list, log2fc_high_thres, log2fc_mid_thres): 
	print("Executing SpecWeightedMatrix()")
	updated_output_df = source_dataframe.copy()

	#Construct empty matrix
	list_pos = []
	for i in range(1, motif_length + 1): 
		list_pos.append("#" + str(i)) #Makes a list of strings from "#1" to "#(motif_length)"
	generic_matrix_df = pd.DataFrame(index = amino_acid_list, columns = list_pos)
	generic_matrix_df = generic_matrix_df.fillna(0)

	#Adjust for inequality of hits specific to one bait vs. the other
	if points_handling == 1: 
		above_below_thres_ratio = SpecRatioBias(bait1_pool = bait1_list, bait2_pool = bait2_list, source_df = source_dataframe, 
			pos_thres = log2fc_mid_thres, neg_thres = (-1 * log2fc_mid_thres), passes_col = "Significant")
	elif points_handling ==2: 
		above_below_thres_ratio = SpecRatioBias(bait1_pool = bait1_list, bait2_pool = bait2_list, source_df = source_dataframe, 
			pos_thres = log2fc_mid_thres, neg_thres = 0, passes_col = "Significant")
	else: 
		raise Exception("Unacceptable value for points_handling! Value:" + str(points_handling))
	print("above_below_thres_ratio =", above_below_thres_ratio)

	#Score the sequences
	for i in np.arange(len(updated_output_df)): 
		seq = updated_output_df.at[i, "BJO_Sequence"]
		#seq = list(seq) #converts to aa list

		passes = updated_output_df.at[i, "Significant"]

		bait1_bait2_log2fc, bait1_bait2_list = LeastDiffFold(bait1_list, bait2_list, i, updated_output_df)
		updated_output_df.at[i, "Least_Different_Log2fc"] = bait1_bait2_log2fc

		PointsFinder(output_df = generic_matrix_df, hthres = log2fc_high_thres, mthres = log2fc_mid_thres, bias_ratio = above_below_thres_ratio, 
			sequence = seq, seq_length = slim_length, seq_log2fc = bait1_bait2_log2fc, pass_call = passes, aalist = amino_acid_list)

	generic_matrix_df = generic_matrix_df.astype("float32")

	return generic_matrix_df, updated_output_df

#--------------------------------------------------------------------------

#Set the thresholds for the point assignment system

print("Log2fc is used to decide which hits to use for matrix construction.")
print("---")
log2fc_upper_thres = NumInput("Please enter the UPPER log2fc threshold as an absolute value (e.g. 1):")
log2fc_lower_thres = NumInput("Please enter the LOWER log2fc threshold as an absolute value (e.g. 0.5):")
print("---")

print("Point System: ")
print("If |log2fc| >", log2fc_upper_thres, "and passes, points = 2")
print("If |log2fc| >", log2fc_lower_thres, "and passes, points = 1")
print("Points sign = log2fc sign. Positive sign = MOSPD2 preference.")
print("---")

#--------------------------------------------------------------------------

#Construction of weighted matrix

weighted_matrix_df, dens_final_scored_df = SpecWeightedMatrix(slim_length, bait1_inputted, bait2_inputted, dens_final_df, list_aa, log2fc_upper_thres, log2fc_lower_thres)

merge_phospho = input("Generated matrix. Merge phospho-residues with their unphosphorylated counterparts? (Y/N)  ")
if merge_phospho == "Y": 
	for n in np.arange(1, slim_length + 1): 
		weighted_matrix_df.at["S", "#" + str(n)] = weighted_matrix_df.at["S", "#" + str(n)] + weighted_matrix_df.at["B", "#" + str(n)]
		weighted_matrix_df.at["T", "#" + str(n)] = weighted_matrix_df.at["T", "#" + str(n)] + weighted_matrix_df.at["J", "#" + str(n)]
		weighted_matrix_df.at["Y", "#" + str(n)] = weighted_matrix_df.at["Y", "#" + str(n)] + weighted_matrix_df.at["O", "#" + str(n)]
	weighted_matrix_df.drop(labels = ["B", "J", "O"], axis = 0, inplace = True)
print("Done!")
print("---")

#Save the generated matrix

print("-------------------")
weighted_matrix_df.to_csv(FilenameSubdir("Output", "Linear_Specificity_Matrix.csv"))
print("Saved! Filename: Linear_Specificity_Matrix.csv")
print("-------------------")

#--------------------------------------------------------------------------

print("Computing edited densitometry data file containing scores for each peptide.")

#Calculate individual residue scores and sum them

spec_col_name = ""
for a in bait1_inputted: 
	spec_col_name = spec_col_name + a + "_"
spec_col_name = spec_col_name + "vs_"
for b in bait2_inputted: 
	spec_col_name = spec_col_name + b + "_"
spec_col_name = spec_col_name + "Specificity_Score"

for i in np.arange(len(dens_final_scored_df)): 
	seq = dens_final_scored_df.at[i, "No_Phos_Sequence"]
	total_score = 0
	for j in np.arange(1, slim_length + 1): 
		res = seq[j-1:j]
		score_current_position = weighted_matrix_df.at[res, "#" + str(j)]
		total_score = total_score + score_current_position

	dens_final_scored_df.at[i, spec_col_name] = total_score

#Final adjustment and output of score transformation method

#list_of_scores = dens_final_scored_df.loc[spec_col_name].tolist()
#max_of_scores = max(list_of_scores)
#min_of_scores = 

#for i in np.arange(len(dens_final_scored_df)): 
#	untransformed_score = dens_final_scored_df.at[i, spec_col_name]


#Save to output

dens_final_scored_df.to_csv(FilenameSubdir("Output", "Specificity_Scored_Dens_DF.csv"))
print("Saved! Filename: Specificity_Scored_Dens_DF.csv")
print("-------------------")
