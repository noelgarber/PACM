#This script constructs a weighted matrix to predict bait-bait specificity in SLiM sequences. 

#Import required functions and packages

import numpy as np
import pandas as pd
import os
from PACM_General_Functions import FilenameSubdir, NumInput
from PACM_General_Vars import list_aa, list_aa_no_phos

print("----------------")
print("Script 3: ")
print("This script generates SLiM bait specificity scores based on inputted baits. ")
print("IMPORTANT: Specificity scores are, in our experience, only interpretable if the underlying SLiM score passes a defined threshold; they cannot be interpreted in isolation.")
print("----------------")

filename = input("Enter the filename containing pairwise scored densitometry data, located in /Output:  ")
dens_final_df = pd.read_csv(FilenameSubdir("Output", filename), index_col = 0)

with open("list_of_baits.ob", "rb") as lob:
	list_of_baits = pickle.load(lob)

#--------------------------------------------------------------------------

#Begin matrix building

slim_length = 15 #This is the length of the motif being analyzed
positions = np.arange(1, slim_length + 1)
bait1_inputted = input("First bait to compare (1 of 2):  ")
bait2_inputted = input("Second bait to compare (2 of 2):  ")

#--------------------------------------------------------------------------

def SpecRatioBias(b1, b2, source_df, thres, passes_col = "Significant"): 
	above_thres_count = 0
	below_neg_thres_count = 0
	for i in np.arange(len(source_df)): 
		passes = source_df.at[i, passes_col]

		bait1_bait2_log2fc = source_df.at[i, bait1 + "_" + bait2 + "_log2fc"]

		if passes == "Yes" and bait1_bait2_log2fc >= thres: 
			above_thres_count += 1
		elif passes == "Yes" and bait1_bait2_log2fc <= (-1 * thres): 
			below_neg_thres_count += 1
	if below_neg_thres_count != 0: 
		ratio = above_thres_count / below_neg_thres_count
	else: 
		ratio = 1
	return ratio, bait1_bait2_log2fc

def PointsFinder(output_df, hthres, mthres, bias_ratio, seq_length, sequence_log2fc, pass_call, aalist): 
	indices = np.arange(seq_length)
	if hthres == "Continuous" or mthres == "Continuous": 
		for n in indices: 
			m = n + 1
			for aa in aalist: 
				if aa == seq[n]: 
					points = sequence_log2fc
					if pass_call == "Yes": 
						generic_matrix_df.at[aa, "#" + str(m)] += points
	else: 
		for n in indices: 
			m = n + 1
			for aa in aalist: 
				if aa == seq[n]: 
					#Calculation of points: 
					if sequence_log2fc >= hthres: 
						points = 2
					elif sequence_log2fc >= mthres: 
						points = 1
					elif sequence_log2fc <= (mthres * -1): 
						points = -1 * bias_ratio
					elif sequence_log2fc <= (hthres * -1): 
						points = -1 * bias_ratio
					else: 
						points = 0
					#Pass-conditional point assignment: 
					if pass_call == "Yes": 
						generic_matrix_df.at[aa, "#" + str(m)] += points
					else: 
						continue

def SpecWeightedMatrix(motif_length, bait1, bait2, source_dataframe, amino_acid_list, log2fc_high_thres, log2fc_mid_thres): 
	#Construct empty matrix
	list_pos = []
	for i in range(1, motif_length + 1): 
		list_pos.append("#" + str(i)) #Makes a list of strings from "#1" to "#(motif_length)"
	generic_matrix_df = pd.DataFrame(index = amino_acid_list, columns = list_pos)
	generic_matrix_df = generic_matrix_df.fillna(0)

	#Adjust for inequality of hits specific to one bait vs. the other
	above_below_thres_ratio = SpecRatioBias(bait1, bait2, source_dataframe, log2fc_mid_thres) 

	#Score the sequences
	for i in np.arange(len(source_dataframe)): 
		seq = source_dataframe.at[i, "BJO_Sequence"]
		seq = list(seq) #converts to aa list

		passes = source_dataframe.at[i, "Significant"]

		bait1_bait2_log2fc = source_dataframe.at[i, bait1 + "_" + bait2 + "_log2fc"]

		PointsFinder(output_df = generic_matrix_df, hthres = log2fc_high_thres, mthres = log2fc_mid_thres, bias_ratio = above_below_thres_ratio, 
			seq_length = slim_length, sequence_log2fc = bait1_bait2_log2fc, pass_call = passes, aalist = amino_acid_list)

	generic_matrix_df = generic_matrix_df.astype("float32")

	return generic_matrix_df

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

weighted_matrix_df = SpecWeightedMatrix(slim_length, bait1_inputted, bait2_inputted, dens_final_df, list_aa, log2fc_upper_thres, log2fc_lower_thres)

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

#Begin scoring algorithm construction

dens_final_scored_df = dens_final_df.copy()

#Calculate individual residue scores and sum them

for i in np.arange(len(dens_final_scored_df)): 
	seq = dens_final_scored_df.at[i, "No_Phos_Sequence"]
	total_score = 0
	for j in np.arange(1, slim_length + 1): 
		res = seq[j-1:j]
		score_current_position = weighted_matrix_df.at[res, "#" + str(j)]
		total_score = total_score + score_current_position

	dens_final_scored_df.at[i, bait1_inputted + "_" + bait2_inputted + "_Specificity_Score"] = total_score

#Save to output

dens_final_scored_df.to_csv(FilenameSubdir("Output", "Specificity_Scored_Dens_DF.csv"))
print("Saved! Filename: Specificity_Scored_Dens_DF.csv")
print("-------------------")
