#For each predicted SLiM, determine if its closest-aligned homologous counterpart in another species is also predicted to be this SLiM

import numpy as np
import pandas as pd
import math
import os
import glob
import time
import scipy.stats as stats
from Bio import pairwise2
from PACM_General_Functions import CharacAA, FindIdenticalResidues, NumInput, FilenameSubdir

#Import the data

slim_filename = input("Input the filename containing predicted SLiMs:  ")
data_df = pd.read_csv(slim_filename)

input_col_yn = input("Use default column name for SLiM sequence to analyze (Best_SLiM)? (Y/N)  ")
if input_col_yn == "Y": 
	col_with_seq = "Best_SLiM"
else: 
	col_with_seq = input("Input the column name to use:  ")

#Import BioMart sequences

homologs_filename = input("Input the filename containing BioMart homologs and their sequences in the target species:  ")
homologs_df = pd.read_csv(homologs_filename, low_memory = False)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Assemble homolog info as dict (faster than scanning through dataframe every time)
print("Assembling a dictionary of human-mouse homologs. This may take a minute.")

def homologs_df2dict(reference_df, source_id_col, target_id_col, target_seq_col): 
	output_dict = {}
	for i in np.arange(len(reference_df)): 
		key = reference_df.at[i, source_id_col]
		target_id = reference_df.at[i, target_id_col]
		target_seq = reference_df.at[i, target_seq_col]
		value = (target_id, target_seq)

		#For occurrences where a human protein ID is homologous to more than one mouse ID, take the one with the longest sequence
		current_keys = list(output_dict.keys())
		if key in current_keys: 
			existing_key = key
			existing_value = output_dict.get(existing_key)
			existing_seq = existing_value[1]
			if len(str(target_seq)) > len(str(existing_seq)): 
				output_dict[key] = value
		else:
			output_dict[key] = value
	return output_dict

homologs_dict = homologs_df2dict(homologs_df, source_id_col = "Human_protein_transcript_ID", target_id_col = "Mouse_protein_ID", target_seq_col = "Mouse_protein_sequence")
print("Success!")

#Define functions for pairwise SLiM homology analysis

def retrieve_homolog(source_species_id, homologs_dictionary): 
	homolog_tuple = homologs_dictionary.get(source_species_id)
	if homolog_tuple is not None: 
		homolog_id = homolog_tuple[0]
		homolog_seq = homolog_tuple[1]
		homolog_length = len(homolog_seq)
	else: 
		homolog_id = None
		homolog_seq = None
		homolog_length = 0

	if homolog_length > 3: 
		homolog_exists = True
	else: 
		homolog_exists = False

	return homolog_id, homolog_seq, homolog_length, homolog_exists

def slim_pairwise_homology(slim_sequence, target_sequence, open_gap_penalty = "Default", extend_gap_penalty = "Default"): 
	print("Starting slim_pairwise_homology()")
	#Find the contiguous homologous SLiM

	if open_gap_penalty == "Default":
		open_gap_penalty = -1 * len(slim_sequence)
	if extend_gap_penalty == "Default":
		extend_gap_penalty = -1 * len(slim_sequence)

	slim_alignments_xs = pairwise2.align.globalxs(slim_sequence, target_sequence, open_gap_penalty, extend_gap_penalty, penalize_end_gaps = False)

	seqA = slim_alignments_xs[0][0]
	seqB = slim_alignments_xs[0][1]

	print("seqB =", seqB)

	#Find index for start of homologous aligned SLiM
	for k, char in enumerate(seqA): 
		if char != "-": 
			alignment_start = k
			break

	homolog_slim_xs = seqB[alignment_start : alignment_start + len(slim_sequence)]
	if homolog_slim_xs == "": 
		homolog_slim_xs = "None found in homolog"

	print("homolog_slim_xs =", homolog_slim_xs)

	#For assessing the quality of the SLiM alignment
	align_score = slim_alignments_xs[0][2]
	print("align_score =", align_score)
	align_score_ratio = align_score / len(slim_sequence)
	print("align_score / len =", align_score_ratio)
	align_identical = FindIdenticalResidues(seqA, seqB)
	print("align_identical =", align_identical)
	align_identity_ratio = align_identical / len(slim_sequence)
	print("align_identity_ratio =", align_identity_ratio)

	return homolog_slim_xs, align_score, align_score_ratio, align_identical, align_identity_ratio

def parent_pairwise_homology(sequence_A, sequence_B):
	#For human protein to homologous protein
	protein_alignments_xx = pairwise2.align.globalxx(human_protein, mouse_homolog_seq)
	protein_align_score = protein_alignments_xx[0][2]
	protein_align_score_ratio = protein_align_score / len(human_protein)
	protein_align_identical = FindIdenticalResidues(protein_alignments_xx[0][0], protein_alignments_xx[0][1])
	protein_align_identity_ratio = protein_align_identical / len(human_protein)

	return protein_align_score, protein_align_score_ratio, protein_align_identical, protein_align_identity_ratio

#Find matching SLiM sequences in homologs

slim_align_ratios = []
slim_identity_ratios = []
protein_align_ratios = []
protein_identity_ratios = []

for i in np.arange(len(data_df)): 
	human_protein_name = data_df.at[i, "Protein"]
	slim_seq = data_df.at[i, col_with_seq]

	human_protein = data_df.at[i, "Sequence"]
	human_protein_id = data_df.at[i, "Ensembl_ID"]
	
	#Get mouse protein ID
	
	mouse_homolog_id, mouse_homolog_seq, mouse_homolog_length, has_homolog = retrieve_homolog(human_protein_id, homologs_dict)

	#Perform pairwise BLAST alignment
	if has_homolog: 
		print("Current SLiM/protein with a mouse homolog:", human_protein_name, "(" + str(i + 1), "of", str(len(data_df)) + ")")
		print("Source SLiM:", slim_seq)

		homolog_slim, slim_align_score, slim_align_score_ratio, slim_align_identical, slim_align_identity_ratio = slim_pairwise_homology(slim_sequence = slim_seq, target_sequence = mouse_homolog_seq)

		whole_align_score, whole_align_score_ratio, whole_align_identical, whole_align_identity_ratio = parent_pairwise_homology(human_protein, mouse_homolog_seq)

		slim_align_ratios.append(slim_align_score_ratio)
		slim_identity_ratios.append(slim_align_identity_ratio)
		protein_align_ratios.append(whole_align_score_ratio)
		protein_identity_ratios.append(whole_align_identity_ratio)

		print("Done! SLiM homolog:", homolog_slim)
	else: 
		homolog_slim = "No known homolog"
		slim_align_score, slim_align_score_ratio, slim_align_identical, slim_align_identity_ratio = None, None, None, None
		whole_align_score, whole_align_score_ratio, whole_align_identical, whole_align_identity_ratio = None, None, None, None

	data_df.at[i, "Mouse_Homolog_ID"] = mouse_homolog_id
	data_df.at[i, "Mouse_Best_SLiM"] = homolog_slim

	data_df.at[i, "SLiM_Align_Score_globalxs"] = slim_align_score
	data_df.at[i, "SLiM_Align_Score_Ratio_globalxs"] = slim_align_score_ratio
	data_df.at[i, "SLiM_Align_Identical_Residues"] = slim_align_identical
	data_df.at[i, "SLiM_Align_Identity"] = slim_align_identity_ratio

	data_df.at[i, "Protein_Align_Score_globalxx"] = whole_align_score
	data_df.at[i, "Protein_Align_Score_Ratio_globalxx"] = whole_align_score_ratio
	data_df.at[i, "Protein_Align_Identical_Residues"] = whole_align_identical
	data_df.at[i, "Protein_Align_Identity"] = whole_align_identity_ratio

#Test whether motif alignment scores exceed global scores on a relative basis that is statistically significant

results_output = []

results_output.append("Mean SliM alignment score / SLiM length = " + str(sum(slim_align_ratios) / len(slim_align_ratios)) + "\n")
results_output.append("Mean protein alignment score / human protein length = " + str(sum(protein_align_ratios) / len(protein_align_ratios)) + "\n")

tstatistic, pvalue = stats.ttest_rel(slim_align_ratios, protein_align_ratios)
results_output.append("t-statistic for paired t-test = " + str(tstatistic) + "\n")
results_output.append("p-value = " + str(pvalue) + "\n")
results_output.append("\n")

results_output.append("Mean SLiM identity percent = " + str(sum(slim_identity_ratios) / len(slim_identity_ratios)) + "\n")
results_output.append("Mean protein identity percent = " + str(sum(protein_identity_ratios) / len(protein_identity_ratios)) + "\n")

tstatistic, pvalue = stats.ttest_rel(slim_identity_ratios, protein_identity_ratios)
results_output.append("t-statistic for paired t-test = " + str(tstatistic) + "\n")
results_output.append("p-value = " + str(pvalue) + "\n")

with open("Output/Step7_relative_scores_results.txt", "w") as results: 
	results.writelines(results_output)
	print(results)
	print("Saved! Path = Output/Step7_relative_scores_results.txt")
print("-----")

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Apply Step2 position-weighted matrices to the identified homologous SLiMs

print("Applying SLiM scoring to homologous SLiMs")
motif_length = NumInput("Enter the motif length:", use_int = True)

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

#Define motif finder based on SPOT data

def MotifFinder(motif_seq, target_motif_length, gap_interpretation = "G"):
	target_motif_positions = np.arange(target_motif_length)

	skip = False

	#Ensure sequences with incomputable/incomplete sequences are not considered
	if motif_seq == "No known homolog": 
		print("Skipping", motif_seq, "- no known homolog")
		skip = True
		skip_reason = "No known homolog"
	elif "B" in motif_seq or "J" in motif_seq or "O" in motif_seq or "U" in motif_seq or "X" in motif_seq or "Z" in motif_seq: 
		print("Skipping", motif_seq, "- contains non-standard residue(s)")
		skip = True
		skip_reason = "Contains forbidden characters"
	elif len(motif_seq) != target_motif_length: 
		print("Skipping " + str(motif_seq) + ": length is " + str(len(motif_seq)) + ", not " + str(target_motif_length))
		skip = True
		skip_reason = "Incorrect length"

	score = 0
	if not skip: 
		print("Processing", motif_seq)

		for i in target_motif_positions: 
			position_number = i + 1

			residue = motif_seq[i]
			if residue == "-": 
				residue = gap_interpretation #Interpret start/end gaps as G, i.e. no side chain

			residue_row_index = index_aa_dict.get(residue)

			print("Processing", residue, "at", str(position_number))

			if i == 0: 
				previous_residue = motif_seq[i+1] #Takes subsequent residue instead when previous residue does not exist (beginning of sequence)
				previous_residue_position = position_number + 1 #Represents actual position, not index
			else: 
				previous_residue = motif_seq[i-1] #Previous residue
				previous_residue_position = position_number - 1 #Represents actual position, not index

			if i == target_motif_length - 1: 
				subsequent_residue = motif_seq[i-1] #Takes previous residue instead when subsequent residue does not exist (end of sequence)
				subsequent_residue_position = position_number - 1 #Represents actual position, not index
			else: 
				subsequent_residue = motif_seq[i+1] #Subsequent residue
				subsequent_residue_position = position_number + 1 #Represents actual position, not index

			if previous_residue == "-": 
				previous_residue = gap_interpretation
			if subsequent_residue == "-": 
				subsequent_residue = gap_interpretation

			#Retrieve biochemical aa characteristic name to be used for conditional matrix lookup
			try: 
				previous_residue_characteristic = CharacAA(previous_residue)
			except: 
				raise Exception(str(motif_seq) + " gave CharacAA error when assessing previous_residue (relative to position " + str(position_number) + "), set to " + previous_residue)

			try: 
				subsequent_residue_characteristic = CharacAA(subsequent_residue)
			except: 
				raise Exception(str(motif_seq) + " gave CharacAA error when assessing subsequent_residue (relative to position " + str(position_number) + "), set to " + subsequent_residue)

			key1 = "#" + str(previous_residue_position) + "=" + previous_residue_characteristic + "_#" + str(position_number) + "_list"
			print("Previous residue key:", key1)
			key2 = "#" + str(subsequent_residue_position) + "=" + subsequent_residue_characteristic + "_#" + str(position_number) + "_list"
			print("Subsequent residue key:", key2)

			list_at_residue_for_previous = dictionary_of_weighted_lists.get(key1)
			print("list_at_residue_for_previous =", list_at_residue_for_previous)
			list_at_residue_for_subsequent = dictionary_of_weighted_lists.get(key2)
			print("list_at_residue_for_subsequent =", list_at_residue_for_subsequent)

			score_previous = list_at_residue_for_previous[residue_row_index]
			print("score_previous =", score_previous)
			score_subsequent = list_at_residue_for_subsequent[residue_row_index]
			print("score_subsequent =", score_subsequent)

			score_current_position = score_previous + score_subsequent
			print("score_current_position =", score_current_position)
			score = score + score_current_position

		print("total score =", score)

	else: 
		score = skip_reason

	return score

#Define motif score assigning function

def MotifScoreAssigner(index, destination_dataframe):
	human_sequence = destination_dataframe.at[index, "Best_SLiM"]
	motif_length = len(human_sequence)
	mouse_sequence = destination_dataframe.at[index, "Mouse_Best_SLiM"]

	mouse_slim_score = MotifFinder(mouse_sequence, motif_length)
	destination_dataframe.at[index, "Mouse_Best_SLiM_Score"] = mouse_slim_score

#Iterate through all the protein sequences

for i in np.arange(len(data_df)): 
	MotifScoreAssigner(i, data_df)

data_df.to_csv(FilenameSubdir("Output", "SLiM_conservation_results.csv"))

print("Saved! Path: Output/SLiM_conservation_results.csv")
print("-----")


