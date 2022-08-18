#This script assigns MOSPD2-VAP FFAT specificity scores to protein lists.

#Import required functions and packages

import numpy as np
import pandas as pd
import pickle
from PACM_General_Functions import FilenameSubdir

print("----------------")
print("Script 5: ")
print("This script assigns bait-bait specificity scores based on an auto-generated matrix in Step 3.")
print("Specificity scores may only be interpreted if the SLiM also passes with an acceptable SLiM score.")
print("----------------")

with open("Step4_output_filename.ob", "rb") as f: 
	file_to_import = pickle.load(f)

use_loaded_file = input("The last output file from Step4_Pairwise_SLiM_Predictor.py was " + file_to_import + ". Would you like to use this file? (Y/N)  ")

if use_loaded_file != "Y": 
	file_to_import = input("Enter the alternative file path:  ")

imported_seqs_df = pd.read_csv(file_to_import, index_col = 0)

slim_length = 15

#--------------------------------------------------------------------------

#Import weighted matrix to use

use_non_default = input("The default matrix file is Linear_Specificity_Matrix.csv. Use this one? (Y/N)  ")
if use_non_default != "Y": 
	weighted_matrix_filename = input("Input the alternative file path:  ")
else: 
	weighted_matrix_filename = FilenameSubdir("Output", "Linear_Specificity_Matrix.csv")

weighted_matrix_df = pd.read_csv(weighted_matrix_filename, index_col = 0)

#--------------------------------------------------------------------------

imported_seqs_scored_df = imported_seqs_df.copy()

#Calculate individual residue scores and sum them

with open("input_private.ob", "rb") as f:
	input_private = pickle.load(f) 

if input_private == "Y": 
	columns_list = ["Best_SLiM", "Second_Best_SLiM", "External_Best_SLiM", "External_Runner-up_SLiM"]
else: 
	columns_list = ["Best_SLiM", "Second_Best_SLiM"]

imported_df_length = len(imported_seqs_scored_df)

for i in np.arange(imported_df_length): 
	print("Processing", i, "of", imported_df_length)
	for col in columns_list: 
		seq = imported_seqs_scored_df.at[i, col]
		total_score = 0
		try: 
			for j in np.arange(1, len(seq) + 1): 
				res = seq[j-1:j]
				score_current_position = weighted_matrix_df.at[res, "#" + str(j)]
				total_score = total_score + score_current_position
			imported_seqs_scored_df.at[i, col + "_Specificity_Score"] = total_score
		except: 
			print("Could not assign specificity score to", imported_seqs_scored_df.at[i, "Protein"], col, "carrying the sequence", str(seq) + ".")

print("Applied specificity scores for predicted FFATs in", file_to_import + ".")

output_filename_csv = file_to_import[:-4] + "_with_Specificity_Scores.csv"
imported_seqs_scored_df.to_csv(output_filename_csv)

output_filename_xlsx = file_to_import[:-4] + "_with_Specificity_Scores.xlsx"
imported_seqs_scored_df.to_excel(output_filename_xlsx)

print("Saved! Filenames:")
print(output_filename_csv, "and", output_filename_xlsx)
print("-------------------")















