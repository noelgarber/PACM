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
from Bio.Align import substitution_matrices
from PACM_General_Vars import index_aa_dict
from PACM_General_Functions import CharacAA, NumInput, FilenameSubdir, use_default, dict_inputter, ListInputter

# Vectorization is not possible for this procedure; suppress Pandas performance warnings arising from fragmentatiom
warnings.simplefilter(action = "ignore", category = pd.errors.PerformanceWarning)

blosum62 = substitution_matrices.load("BLOSUM62")

def motif_similarity(motif_seq, target_seq):
	'''
	Finds the gap-less segment of a target sequence that is most similar to a short motif sequence

	Args:
		motif_seq (str):  short linear motif to check against target sequence
		target_seq (str): target homolog sequence

	Returns:
		best_motif (str): best matching motif
		best_score (int): BLOSUM62 similarity score
	'''

	motif_length = len(motif_seq)
	if motif_length == 0:
		return ("",0,0.0)

	motif_seq = np.array(list(motif_seq))
	target_seq = np.array(list(target_seq))

	slice_indices = np.arange(len(target_seq) - motif_length + 1)[:, np.newaxis] + np.arange(motif_length)
	sliced_target_2d = target_seq[slice_indices]

	scores_2d = np.zeros(shape=sliced_target_2d.shape)
	for i, motif_aa in enumerate(motif_seq):
		col_residues = sliced_target_2d[:,i]
		unique_residues_col = np.unique(col_residues)
		col_scores = np.zeros(shape=col_residues.shape)
		for target_aa in unique_residues_col:
			matrix_value = blosum62.get((motif_aa, target_aa))
			mask = np.char.equal(col_residues, target_aa)
			col_scores[mask] = matrix_value
		scores_2d[:,i] = col_scores

	scores = scores_2d.sum(axis=1)
	best_score_idx = np.argmax(scores)
	best_score = scores[best_score_idx]

	best_motif_arr = sliced_target_2d[best_score_idx]
	best_motif = "".join(best_motif_arr)

	identity_mask = np.char.equal(motif_seq, best_motif_arr)
	best_identity_ratio = identity_mask.mean()

	return (best_motif, best_score, best_identity_ratio)

def motif_pairwise_chunk(seq_list):
	'''
	Simple parallelizable version of motif_pairwise_homology that accepts lists of (motif, target) sequence pairs

	Args:
		seq_list (list): list of tuples of (motif sequence, target homolog sequence)

	Returns:
		results (list): list of results
	'''

	results = []
	for seq_pair in seq_list:
		if isinstance(seq_pair[0], str) and isinstance(seq_pair[1], str):
			result = motif_similarity(seq_pair[0], seq_pair[1])
		else:
			result = ("",0,0.0)
		results.append(result)

	return results

def evaluate_homologs(data_df, motif_seq_cols, homolog_seq_cols):
	'''
	Main function to evaluate homologs in a dataframe for homology with the motif of interest

	Args:
		data_df (pd.DataFrame):  dataframe containing motif, parent, and homolog seqs
		motif_seq_cols (list):   col names holding predicted motifs from parental protein sequences
		homolog_seq_cols (list): col names with homolog protein sequences to be searched

	Returns:
		data_df (pd.DataFrame):  dataframe with added motif homology columns
	'''

	motif_col_count = len(motif_seq_cols)
	for i, motif_seq_col in enumerate(motif_seq_cols):
		print(f"Evaluating homologs for {motif_seq_col} ({i+1} of {motif_col_count})...")
		motif_seqs = data_df[motif_seq_col].copy()

		homolog_col_count = len(homolog_seq_cols)
		for j, homolog_seq_col in enumerate(homolog_seq_cols):
			# Get the col prefix not including "_seq"
			col_prefix = homolog_seq_col.rsplit("_",1)[0] + f"_vs_Motif_#{i}"
			print(f"\tEvaluating homolog {col_prefix} ({j+1} of {homolog_col_count}) for motif #{i}...")
			col_idx = data_df.columns.get_loc(homolog_seq_col)
			new_cols_df = pd.DataFrame()

			# Full homolog sequences are only necessary during evaluation, so we pop them out for better memory savings
			homolog_seqs = data_df[homolog_seq_col]

			# Get pairs of sequences to be evaluated
			seq_pairs = [(motif, target) for motif, target in zip(motif_seqs, homolog_seqs)]
			chunk_size = 1000
			seq_pairs_chunks = [seq_pairs[i:i+chunk_size] for i in range(0, len(seq_pairs), chunk_size)]
			del seq_pairs # no longer necessary

			# Parallel processing of motif homologies
			motif_homology_tuples = []
			pool = multiprocessing.pool.Pool()
			for chunk_results in pool.map(motif_pairwise_chunk, seq_pairs_chunks):
				motif_homology_tuples.extend(chunk_results)
			pool.close()
			pool.join()

			del seq_pairs_chunks # no longer necessary

			# Assign motif homologies to new cols dataframe
			new_cols_df[col_prefix+"_matching_motif"] = [motif_homolog[0] for motif_homolog in motif_homology_tuples]
			new_cols_df[col_prefix+"_motif_similarity"] = [motif_homolog[1] for motif_homolog in motif_homology_tuples]
			new_cols_df[col_prefix+"_motif_identity"] = [motif_homolog[2] for motif_homolog in motif_homology_tuples]
			del motif_homology_tuples # no longer necessary

			# Merge into dataframe
			original_cols = list(data_df.columns)
			new_cols = list(new_cols_df.columns)
			reordered_cols = original_cols[0:col_idx] + new_cols + original_cols[col_idx:]
			data_df = pd.concat([data_df, new_cols_df], axis=1)
			data_df = data_df[reordered_cols]
			del new_cols_df

	# Delete unnecessary whole homolog sequences
	data_df.drop(homolog_seq_cols, axis=1)

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