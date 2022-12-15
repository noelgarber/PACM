#This script takes protein sequences and computes their SLiM scores based on the results of Step2_Pairwise_SLiM_Matrices_Generator.py

#Import required functions and packages

import numpy as np
import pandas as pd
import time
import glob
import os
import pickle
from PACM_General_Functions import CharacAA, FilenameSubdir, ListInputter
from PACM_General_Vars import list_aa_no_phos


print("----------------")
print("Script 4:")
print("This script performs pairwise SLiM scoring for lists of protein sequences.")
print("----------------")

print("If you want to compare outside methods in parallel, input the scoring function into a script called /Private/Method.py.")
print("Ensure that the function is named \"ImportedScorer\".")
input_private = input("Do you want to import a parallel method? (Y/N)  ")
if input_private == "Y": 
	try: 
		from Private.Method import ImportedScorer
	except: 
		raise Exception("Import failed!")

with open("input_private.ob", "wb") as f: 
	pickle.dump(input_private, f)

#Declare motif characteristics

motif_length = 15

#--------------------------------------------------------------------------

#Import dictionary of conditional weighted matrices generated by Step2_Pairwise_SLiM_Matrices_Generator.py

dictionary_of_weighted_matrices = {}

current_directory = os.getcwd()
destination_directory = os.path.join(current_directory, r"dictionary_of_weighted_matrices")

matrices_filenames_list = glob.glob(os.path.join(destination_directory, "#*=*.csv"))
for entry in matrices_filenames_list: 
	root, filename = entry.split("#")
	filename = filename[:-4]
	key = "#" + filename
	value_df = pd.read_csv(entry, index_col = 0)
	dictionary_of_weighted_matrices[key] = value_df

#Construct dictionary of matrices by position that returns lists

dictionary_of_weighted_lists = {}

for key, value_df in dictionary_of_weighted_matrices.items(): 
	for n in np.arange(1, motif_length + 1): 
		new_key = key + "_#" + str(n) + "_list"
		new_value_list = value_df.loc[:, "#" + str(n)].values.tolist()
		dictionary_of_weighted_lists[new_key] = new_value_list


#Define dictionary to get index of an amino acid for matrix lookup

index_aa_dict = {
	"D": 0,
	"E": 1,
	"R": 2,
	"H": 3,
	"K": 4,
	"S": 5,
	"T": 6,
	"N": 7,
	"Q": 8,
	"C": 9,
	"G": 10,
	"P": 11,
	"A": 12,
	"V": 13,
	"I": 14,
	"L": 15,
	"M": 16,
	"F": 17,
	"Y": 18,
	"W": 19
}

#--------------------------------------------------------------------------

#Read the list of protein sequences to analyze into a Pandas DataFrame

use_non_default = input("The default file to process is Uniprot_All_Human_Protein_Isoforms.csv. Would you like to process a different file? (Y/N)  ")
if use_non_default == "Y":
	proteins_to_process_filename = input("Input the filename in the current directory: ")
else: 
	proteins_to_process_filename = "Uniprot_All_Human_Protein_Isoforms.csv"

protein_seqs_df = pd.read_csv(proteins_to_process_filename)

#--------------------------------------------------------------------------

#Define function for inputting rules for use in ManualRuleChecker

def ManualRuleDefiner(frame_length): 
	manual_rules_dict = {}
	print("Defining manual rules that, when broken, cause a motif to be immediately rejected.")
	for i in np.arange(1, frame_length + 1): 
		manual_rule = input("At position " + str(i) + ", do you want a manual rule? (Y/n)  ")
		if manual_rule == "Y": 
			residues_allowed = ListInputter("Requiring the following residues at position #" + str(i) + ":")
			manual_rules_dict[i] = residues_allowed
	return manual_rules_dict

#Define function to reject motifs not following manual rules; increases speed massively

def ManualRuleChecker(seq_input, rules_dict): 
	seq_input_length = len(seq_input)
	rule_broken = False
	for position, allowed_residues in rules_dict.items(): 
		seq_at_position = seq_input[position - 1]
		if seq_at_position not in allowed_residues: 
			rule_broken = True
	return rule_broken

#Define motif finder based on SPOT data

def MotifFinder(sequence, length, target_motif_length, rules):
	target_motif_positions = np.arange(target_motif_length)

	possible_motifs = length - target_motif_length + 1

	first_score = 0
	second_score = 0
	first_motif = ""
	second_motif = ""

	for i in np.arange(0, possible_motifs): 
		score = 0 #Declares variable
		motif_start = i.item()
		motif_end = motif_start + motif_length
		motif_seq = sequence[motif_start:motif_end]

		#Ensures sequences with incomputable/incomplete sequences are not considered
		if "B" in motif_seq or "J" in motif_seq or "O" in motif_seq or "U" in motif_seq or "X" in motif_seq or "Z" in motif_seq: 
			continue

		#Check manual rules
		motif_rule_broken = ManualRuleChecker(motif_seq, rules)
		if motif_rule_broken:
			continue

		for j in target_motif_positions: 
			residue = motif_seq[j]
			residue_row_index = index_aa_dict.get(residue)

			if j == 0: 
				previous_residue = motif_seq[j+1]
				previous_residue_position = j + 1
			else: 
				previous_residue = motif_seq[j-1]
				previous_residue_position = j - 1

			if j == target_motif_length - 1: 
				subsequent_residue = motif_seq[j-1]
				subsequent_residue_position = j - 1
			else: 
				subsequent_residue = motif_seq[j+1]
				subsequent_residue_position = j + 1

			try: 
				previous_residue_characteristic = CharacAA(previous_residue) #returns the biochemical aa characteristic name
				subsequent_residue_characteristic = CharacAA(subsequent_residue)
			except: 
				print(motif_seq, "gave CharacAA error")
				print(stop)

			key1 = "#" + str(previous_residue_position + 1) + "=" + previous_residue_characteristic + "_#" + str(j + 1) + "_list"
			key2 = "#" + str(subsequent_residue_position + 1) + "=" + subsequent_residue_characteristic + "_#" + str(j + 1) + "_list"

			list_at_residue_for_previous = dictionary_of_weighted_lists.get(key1)
			list_at_residue_for_subsequent = dictionary_of_weighted_lists.get(key2)

			score_previous = list_at_residue_for_previous[residue_row_index]
			score_subsequent = list_at_residue_for_subsequent[residue_row_index]

			score_current_position = score_previous + score_subsequent
			score = score + score_current_position

		if score > first_score and motif_seq[7] in ["F", "Y"]: 
			second_score = first_score #Assigns the previously best score as the next-best
			second_motif = first_motif
			first_score = score #Assigns the new best score as best
			first_motif = motif_seq

	return first_motif, first_score, second_motif, second_score

#--------------------------------------------------------------------------

#Define motif score assigning function

def MotifScoreAssigner(index, destination_dataframe, screening_rules):
	protein_name = destination_dataframe.at[index, "Protein"]
	whole_sequence = destination_dataframe.at[index, "Sequence"]
	whole_sequence = "GGGGGG" + whole_sequence #Adjusts for sequences that have a truncated N-terminal tract
	protein_length = len(whole_sequence)

	best_SLiM, best_score, second_best_SLiM, second_best_score = MotifFinder(whole_sequence, protein_length, motif_length, screening_rules)

	destination_dataframe.at[index, "Best_SLiM"] = best_SLiM
	destination_dataframe.at[index, "Best_SLiM_Score"] = best_score
	destination_dataframe.at[index, "Second_Best_SLiM"] = second_best_SLiM
	destination_dataframe.at[index, "Second_Best_SLiM_Score"] = second_best_score

	if input_private == "Y": 
		external_best_motif, external_best_score, external_runnerup_motif, external_runnerup_score = ImportedScorer(whole_sequence, protein_length)

		destination_dataframe.at[index, "External_Best_SLiM"] = external_best_motif
		destination_dataframe.at[index, "External_Best_Score"] = external_best_score
		destination_dataframe.at[index, "External_Runner-up_SLiM"] = external_runnerup_motif
		destination_dataframe.at[index, "External_Runner-up_Score"] = external_runnerup_score

#--------------------------------------------------------------------------

#Iterate through all the protein sequences

defined_rules = ManualRuleDefiner(motif_length)

number_of_proteins = len(protein_seqs_df)

for i in np.arange(number_of_proteins): 
	MotifScoreAssigner(i, protein_seqs_df, defined_rules)
	print("Completed", i, "of", number_of_proteins)

print("----------------")

output_filename = FilenameSubdir("Output", proteins_to_process_filename[:-4] + "_Scored_SLiMs.csv")

with open("Step4_Output_Filename.ob", "wb") as f:
	pickle.dump(output_filename, f)

protein_seqs_df.to_csv(output_filename)
print("Output saved! Filename:", output_filename)
print("----------------")