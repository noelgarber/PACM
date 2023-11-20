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
from tqdm import trange
from Bio.Align import substitution_matrices
from PACM_General_Vars import index_aa_dict
from PACM_General_Functions import CharacAA, NumInput, FilenameSubdir, use_default, dict_inputter, ListInputter

# Vectorization is not possible for this procedure; suppress Pandas performance warnings arising from fragmentatiom
warnings.simplefilter(action = "ignore", category = pd.errors.PerformanceWarning)

blosum62 = substitution_matrices.load("BLOSUM62")

def motif_similarity(seqs_tuple):
	'''
	Finds the gap-less segment of a target sequence that is most similar to a short motif sequence

	Args:
		seqs_tuple (list): tuple of (motif_seqs, target_seqs, col_names)

	Returns:

	'''

	motif_seqs, target_seqs, col_names, accession_col_idx = seqs_tuple

	homologous_motifs = []
	homologous_similarity_scores = []
	homologous_identity_ratios = []

	for motif_seq, target_seq in zip(motif_seqs, target_seqs):
		motif_length = len(motif_seq)
		if motif_length == 0 or len(target_seq) == 0:
			homologous_motifs.append("")
			homologous_similarity_scores.append(0.0)
			homologous_identity_ratios.append(0.0)
			continue

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

		homologous_motifs.append(best_motif)
		homologous_similarity_scores.append(best_score)
		homologous_identity_ratios.append(best_identity_ratio)

	return (homologous_motifs, homologous_similarity_scores, homologous_identity_ratios, col_names, accession_col_idx)

def evaluate_homologs(data_df, motif_seq_cols, homolog_seq_cols):
	'''
	Main function to evaluate homologs in a dataframe for homology with the motif of interest

	Args:
		data_df (pd.DataFrame):    dataframe containing motif, parent, and homolog seqs
		motif_seq_cols (list):     col names holding predicted motifs from parental protein sequences
		homolog_seq_cols (list):   col names with homolog protein sequences to be searched

	Returns:
		data_df (pd.DataFrame):    dataframe with added motif homology columns
		homolog_motif_cols (list): list of column names containing homologous motif sequences
	'''

	print("Preparing homolog and motif seqs for similarity analysis...")
	homolog_motif_cols = []
	seqs_tuples = []
	for homolog_seq_col in homolog_seq_cols:
		col_idx = data_df.columns.get_loc(homolog_seq_col)
		homolog_seqs = data_df.pop(homolog_seq_col).fillna("").to_list()
		current_insertion_point = col_idx
		for i, motif_seq_col in enumerate(motif_seq_cols):
			motif_seqs = data_df[motif_seq_col].fillna("").to_list()
			col_prefix = homolog_seq_col.rsplit("_", 1)[0] + "_vs_" + motif_seq_col
			cols = [col_prefix + "_matching_motif", col_prefix + "_motif_similarity", col_prefix + "_motif_identity"]
			seqs_tuples.append((motif_seqs, homolog_seqs, cols, current_insertion_point))
			homolog_motif_cols.append(col_prefix + "_matching_motif")
			current_insertion_point += 3

	new_cols_names = []
	new_cols_data = []
	with trange(len(seqs_tuples), desc="Evaluating motif homolog similarities by column pair...") as pbar:
		pool = multiprocessing.Pool()
		for result in pool.imap(motif_similarity, seqs_tuples):
			homologous_motifs, homologous_similarities, homologous_identities, col_names, insertion_idx = result

			new_cols_names.extend(col_names)
			new_cols_data.extend([homologous_motifs, homologous_similarities, homologous_identities])

			pbar.update()

	pool.close()
	pool.join()

	homologous_motifs_df = pd.DataFrame(dict(zip(new_cols_names, new_cols_data)))
	data_df = pd.concat([data_df, homologous_motifs_df], axis=1)

	cols = list(data_df.columns)
	homolog_accession_cols = [homolog_seq_col.rsplit("_",1)[0] for homolog_seq_col in homolog_seq_cols]
	reordered_cols = [col for col in cols if "homolog" not in col]
	for homolog_accession_col in homolog_accession_cols:
		subset_cols = []
		for col in cols:
			if homolog_accession_col in col:
				subset_cols.append(col)
		reordered_cols.extend(subset_cols)

	data_df = data_df[reordered_cols]

	return data_df, homolog_motif_cols