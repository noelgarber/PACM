#For each predicted SLiM, determine if its closest-aligned homologous counterpart in another species is also predicted to be this SLiM

import numpy as np
import pandas as pd
import math
import os
import sys
import glob
import time
import requests
import scipy.stats as stats
import warnings
import multiprocessing
from functools import partial
from tqdm import tqdm
from biomart import BiomartServer
from Bio import pairwise2
from PACM_General_Vars import index_aa_dict
from PACM_General_Functions import CharacAA, NumInput, FilenameSubdir, use_default, dict_inputter, ListInputter

# Vectorization is not possible for this procedure; suppress Pandas performance warnings arising from fragmentatiom
warnings.simplefilter(action = "ignore", category = pd.errors.PerformanceWarning)

def motif_pairwise_homology(seq_pair, open_gap_penalty = None, extend_gap_penalty = None):
	'''
	Finds the best-matching homologous motif in a target sequence for a given linear motif

	Args:
		seq_pair (tuple|list): 				 tuple of (short linear motif sequence, longer sequence to be searched)
		open_gap_penalty (None|int|float):   open gap penalty; setting to None sets it to the negative of motif length
		extend_gap_penalty (None|int|float): extend gap penalty; setting to None sets it to the negative of motif length

	Returns:
		homolog_motif_xs (str):      homologous motif sequence found in target sequence
		align_identical (int):       number of identical residues
		align_identity_ratio (int):  percentage identity
	'''

	motif_sequence, target_sequence = seq_pair

	if not isinstance(motif_sequence, str) or not isinstance(target_sequence, str):
		return ("",0,0.0)

	if open_gap_penalty is None:
		open_gap_penalty = -1 * len(motif_sequence)
	if extend_gap_penalty is None:
		extend_gap_penalty = -1 * len(motif_sequence)

	motif_alignments_xs = pairwise2.align.globalxs(motif_sequence, target_sequence, open_gap_penalty,
												   extend_gap_penalty, penalize_end_gaps = False)

	if len(motif_alignments_xs) == 0:
		return ("",0,0.0)
	seqA = motif_alignments_xs[0][0]
	seqB = motif_alignments_xs[0][1]

	# Extract homologous motif sequence
	alignment_start = 0
	for k, char in enumerate(seqA):
		if char != "-":
			alignment_start = k
			break
	homolog_motif_xs = seqB[alignment_start : alignment_start + len(motif_sequence)]

	if homolog_motif_xs == "":
		homolog_motif_xs = "None found in homolog"

	# Extract the number of identical residues (equal to alignment score for this method)
	align_identical = motif_alignments_xs[0][2]
	align_identity_ratio = align_identical / len(motif_sequence)

	return (homolog_motif_xs, align_identical, align_identity_ratio)

def parent_pairwise_homology(seq_pair):
	'''
	Checks parental sequence homology

	Args:
		seq_pair (tuple|list): pair of source and target parental sequences to check homology for

	Returns:
		protein_align_identical (int):  number of identical residues
		protein_identity_ratio (float): ratio of identical residues within the motif frame
	'''

	sequence_A, sequence_B = seq_pair

	if not isinstance(sequence_A, str) or not isinstance(sequence_B, str):
		return (0,0.0)

	protein_alignments_xx = pairwise2.align.globalxx(sequence_A, sequence_B)
	if len(protein_alignments_xx) == 0:
		return (0,0.0)

	protein_align_identical = protein_alignments_xx[0][2]
	protein_identity_ratio = protein_align_identical / len(sequence_A)

	return (protein_align_identical, protein_identity_ratio)

def evaluate_homologs(data_df, motif_seq_cols, parent_seq_col, homolog_seq_cols):
	'''
	Main function to evaluate homologs in a dataframe for homology with the motif of interest

	Args:
		data_df (pd.DataFrame):  dataframe containing motif, parent, and homolog seqs
		motif_seq_cols (list):   col names holding predicted motifs from parental protein sequences
		parent_seq_col (str):    col name for parental seqs
		homolog_seq_cols (list): col names with homolog protein sequences to be searched

	Returns:
		data_df (pd.DataFrame):  dataframe with added motif homology columns
	'''

	cols = list(data_df.columns)
	parent_seqs = data_df[parent_seq_col].to_list()

	motif_col_count = len(motif_seq_cols)
	for i, motif_seq_col in enumerate(motif_seq_cols):
		print(f"Evaluating homologs for {motif_seq_col} ({i+1} of {motif_col_count})...")
		motif_seqs = data_df[motif_seq_col].to_list()

		homolog_col_count = len(homolog_seq_cols)
		for j, homolog_seq_col in enumerate(homolog_seq_cols):
			# Get the col prefix not including "_seq"
			col_prefix = homolog_seq_col.rsplit("_",1)[0]
			print(f"\tEvaluating homolog {col_prefix} ({j+1} of {homolog_col_count})")
			col_idx = data_df.columns.get_loc(homolog_seq_col)

			# Evaluate motif homologies
			homolog_seqs = data_df[homolog_seq_col].to_list()
			motif_homology_tuples = []
			motif_pairwise_partial = partial(motif_pairwise_homology, open_gap_penalty=None, extend_gap_penalty=None)
			zipped_seqs = zip(motif_seqs, homolog_seqs)
			seq_pairs = [(motif, target) for motif, target in zipped_seqs]
			with tqdm(total=len(motif_seqs), desc="Processing pairwise motif homologies") as pbar:
				pool = multiprocessing.pool.Pool()
				for result in pool.imap_unordered(motif_pairwise_partial, seq_pairs):
					motif_homology_tuples.append(result)
					pbar.update()
				pool.join()
				pool.close()
				pbar.close()

			# Assign motif homologies to dataframe
			homolog_motif_seqs = [motif_homology_tuple[0] for motif_homology_tuple in motif_homology_tuples]
			data_df[col_prefix+"_matching_motif"] = homolog_motif_seqs
			del homolog_motif_seqs

			homolog_motif_identities = [motif_homology_tuple[1] for motif_homology_tuple in motif_homology_tuples]
			data_df[col_prefix+"_motif_identical_residues"] = homolog_motif_identities
			del homolog_motif_identities

			homolog_motif_ratios = [motif_homology_tuple[2] for motif_homology_tuple in motif_homology_tuples]
			data_df[col_prefix+"_motif_identity_ratios"] = homolog_motif_ratios
			del homolog_motif_ratios

			# Evaluate parental sequence homologies
			zipped_parental_seqs = zip(parent_seqs, homolog_seqs)
			parent_seq_pairs = [(sequence_A, sequence_B) for sequence_A, sequence_B in zipped_parental_seqs]
			parent_tuples = []
			with tqdm(total=len(parent_seqs), desc="Processing pairwise parental sequence homologies") as pbar:
				pool = multiprocessing.pool.Pool()
				for result in pool.imap_unordered(parent_pairwise_homology, parent_seq_pairs):
					parent_tuples.append(result)
					pbar.update()
				pool.join()
				pool.close()
				pbar.close()

			homolog_parental_identities = [parent_tuple[0] for parent_tuple in parent_tuples]
			homolog_parental_ratios = [parent_tuple[1] for parent_tuple in parent_tuples]

			# Assign parental homologies to dataframe
			data_df[col_prefix+"_parent_identical_residues"] = homolog_parental_identities
			data_df[col_prefix+"_parent_identity_ratios"] = homolog_parental_ratios

			# Reorder cols
			preceding_cols = cols[0:col_idx+1]
			new_cols = [col_prefix+"_matching_motif",
						col_prefix+"_motif_identical_residues",
						col_prefix+"_motif_identity_ratios",
						col_prefix+"_parent_identical_residues",
						col_prefix+"_parent_identity_ratios"]
			following_cols = cols[col_idx+1:]
			reordered_cols = preceding_cols + new_cols + following_cols
			data_df = data_df[reordered_cols]

		del motif_seqs

	return data_df






# TODO Sort the rest of this legacy code below
'''
def homology_analyzer(slim_input_df, parameters_dict): 
	slim_identities = []
	protein_identities = []

	source_slim_seq_col = parameters_dict.get("source_slim_seq_col")
	homolog_species = parameters_dict.get("homolog_species")
	source_protein_name_col = parameters_dict.get("source_protein_name_col")
	source_protein_ensembl_col = parameters_dict.get("source_protein_ensembl_col")
	source_protein_seq_col = parameters_dict.get("source_protein_seq_col")
	target_protein_ensembl_col = parameters_dict.get("target_protein_ensembl_col")
	target_protein_seq_col = parameters_dict.get("target_protein_seq_col")
	target_protein_identity_col = parameters_dict.get("target_protein_identity_col")

	print("Applying homology_analyzer() to column:", target_protein_ensembl_col)

	dataframe_length = len(slim_input_df)

	for i in np.arange(dataframe_length): 
		print("\tProcessing", (i + 1), "of", (dataframe_length + 1))
		source_protein_name = slim_input_df.at[i, source_protein_name_col]
		slim_seq = slim_input_df.at[i, source_slim_seq_col]
		print("\tSource SLiM sequence:", slim_seq)

		source_protein_seq = slim_input_df.at[i, source_protein_seq_col]
		source_protein_id = slim_input_df.at[i, source_protein_ensembl_col]
		target_homolog_seq = slim_input_df.at[i, target_protein_seq_col]
		target_homolog_id = slim_input_df.at[i, target_protein_ensembl_col]
		
		#Check if this line has a homolog in the column
		if len(str(target_homolog_seq)) >= len(slim_seq): 
			has_homolog = True
			print("\tTarget homolog sequence length:", str(len(target_homolog_seq)))
		else: 
			has_homolog = False
			print("\tTarget homolog sequence:", target_homolog_seq)

		print("\tHas homolog in this column:", has_homolog)

		#Perform pairwise BLAST alignment
		if has_homolog: 
			#Perform source-target SLiM alignment with pairwise2 to obtain homolog SLiM sequence and sequence identity
			homolog_slim, slim_align_identical, slim_align_identity_ratio = motif_pairwise_homology(motif_sequence = slim_seq, target_sequence = target_homolog_seq)
			
			#Perform or retieve source-target whole protein sequence identity for comparison
			if homolog_request_method == "REST": 
				#REST method retrieves the total source-homolog sequence identity, so it can be referenced instead of calculated again
				whole_align_identity_ratio = slim_input_df.at[i, target_protein_identity_col] / 100 #Given as percentage, not decimal
			elif homolog_request_method == "ALL": 
				#biomart package method does not retrieve sequence identity, so we must calculate it with pairwise2 for the whole proteins
				whole_align_identical, whole_align_identity_ratio = parent_pairwise_homology(source_protein_seq, target_homolog_seq)
				slim_input_df.at[i, target_protein_ensembl_col + "_identity"] = whole_align_identity_ratio #Adds value to dataframe, since it had to be computed
			else: 
				raise Exception("Forbidden value for homolog_request_method; value was " + homolog_request_method + ", expected either REST or ALL.")

			#Append sequence identity values to list, which will be used later for statistical analysis
			slim_identities.append(slim_align_identity_ratio)
			protein_identities.append(whole_align_identity_ratio)

		else: 
			homolog_slim = "No known homolog"
			slim_align_identity_ratio = None
			whole_align_identity_ratio = None

		slim_input_df.at[i, target_protein_ensembl_col + "_best_slim"] = homolog_slim
		slim_input_df.at[i, target_protein_ensembl_col + "_slim_align_identity"] = slim_align_identity_ratio

		print("\tComplete: SLiM homolog =", homolog_slim)
		print("\t----------------------------------------")

	return slim_identities, protein_identities

#Find matching SLiM sequences in homologs

slim_protein_identities_dict = {}

for homolog_cols_tuple in homolog_col_list: 
	homolog_col_species = homolog_cols_tuple[0].split("_")[0] + "_" + homolog_cols_tuple[0].split("_")[1] #Removes trailing info and leaves only genus_species (e.g. mus_musculus_homolog_1 becomes mus_musculus)

	print("Currently processing column group:", homolog_col)

	homology_analyzer_parameters = {
		"source_slim_seq_col": col_with_seq,
		"homolog_species": homolog_col_species, 
		"source_protein_name_col": "Protein", 
		"source_protein_ensembl_col": ensembl_peptide_col, 
		"source_protein_seq_col": "Sequence", 
		"target_protein_ensembl_col": homolog_cols_tuple[0], 
		"target_protein_identity_col": homolog_cols_tuple[1],
		"target_protein_seq_col": homolog_cols_tuple[2]
	}

	#Perform in-place homology analysis on whole columns, which also returns lists for statistical analysis
	target_slim_identity_ratios, target_protein_identity_ratios = homology_analyzer(slim_input_df = data_df, parameters_dict = homology_analyzer_parameters)

	#Get existing/running identities lists in the form of a tuple
	slim_protein_identities_tuple = slim_protein_identities_dict.get(homolog_col_species)

	if slim_protein_identities_tuple != None: 
		slim_identities_for_species = slim_protein_identities_tuple[0]
		protein_identities_for_species = slim_protein_identities_tuple[1]
	else: 
		slim_identities_for_species = []
		protein_identities_for_species = []		

	#Extend identities lists
	slim_identities_for_species.extend(target_slim_identity_ratios)
	protein_identities_for_species.extend(target_protein_identity_ratios)
	slim_protein_identities_tuple = (slim_identities_for_species, protein_identities_for_species)

	#Reassign to dict as a tuple
	slim_protein_identities_dict[homolog_col_species] = slim_protein_identities_tuple

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Test whether motif alignment scores exceed global scores on a relative basis that is statistically significant

print("-------------------------------------------------------")
print("Testing whether motif sequence identity differs from global sequence identity...")

results_output = []

for target_species, slim_protein_identities in slim_protein_identities_dict.items(): 
	slim_identity_ratios = slim_protein_identities[0]
	protein_identity_ratios = slim_protein_identities[1]

	results_output.append("Analysis: " + target_species + "\n")
	results_output.append("Mean SLiM identity percent = " + str(sum(slim_identity_ratios) / len(slim_identity_ratios)) + "\n")
	results_output.append("Mean protein identity percent = " + str(sum(protein_identity_ratios) / len(protein_identity_ratios)) + "\n")

	tstatistic, pvalue = stats.ttest_rel(slim_identity_ratios, protein_identity_ratios)
	results_output.append("t-statistic for paired t-test = " + str(tstatistic) + "\n")
	results_output.append("p-value = " + str(pvalue) + "\n")
	results_output.append("-------------------------------------------------------\n")

for result_line in results_output: 
	print(result_line)

with open("Output/Step7_relative_scores_results.txt", "w") as results: 
	results.writelines(results_output)
	print("Saved! Path = Output/Step7_relative_scores_results.txt")





#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Apply Step2 position-weighted matrices to the identified homologous SLiMs

print("--------------------------------------------------------------------------------")
print("Applying SLiM scoring to homologous SLiMs in all selected species")

motif_length = NumInput("Enter the motif length:", use_int = True)
print("---")
print("Note: This method will use the most recently generated results of Step2_Pairwise_SLiM_Matrices_Generator.py")

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
			if residue == "-" or residue == "*": 
				residue = gap_interpretation #Interpret start/end gaps and stops as G, i.e. no side chain

			residue_row_index = index_aa_dict.get(residue) #Note that index_aa_dict was imported from PACM_General_Vars.py rather than generated in this script

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
			if subsequent_residue == "-" or subsequent_residue == "*": 
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

def retrieve_motif_score(index, input_df, slim_id_col, slim_seq_col, length, scoring_verbose = False):
	if scoring_verbose: 
		print("Launching retrieve_motif_score()...")
		print("\tindex = ", index)
	seq_to_score = input_df.at[index, slim_seq_col]
	if scoring_verbose: 
		print("\tseq_to_score = ", seq_to_score)
		print("\tLaunching MotifFinder()...")
	slim_score = MotifFinder(seq_to_score, length)
	if scoring_verbose: 
		print("\t\tslim_score =", slim_score)
	destination_col = slim_id_col + "_slim_score"
	input_df.at[index, destination_col] = slim_score
	if scoring_verbose: 
		print("slim_score assigned to input dataframe at [" + str(index) + ", \"" + destination_col + "\"]")

#Iterate through all the protein sequences

for homolog_cols_tuple in homolog_col_list: 
	homolog_col_seq = homolog_cols_tuple[2]
	for i in np.arange(len(data_df)): 
		retrieve_motif_score(index = i, input_df = data_df, slim_id_col = homolog_col, slim_seq_col = homolog_col_seq, length = motif_length, scoring_verbose = True)

data_df.to_csv(FilenameSubdir("Output", "SLiM_conservation_results.csv"))

print("Saved! Path: Output/SLiM_conservation_results.csv")
print("-----")

'''