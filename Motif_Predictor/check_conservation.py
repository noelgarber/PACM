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
from tqdm import tqdm
from biomart import BiomartServer
from Bio import pairwise2
from PACM_General_Vars import index_aa_dict
from PACM_General_Functions import CharacAA, FindIdenticalResidues, NumInput, FilenameSubdir, use_default, dict_inputter, ListInputter

warnings.simplefilter(action = "ignore", category = pd.errors.PerformanceWarning) #Vectorization is impossible; suppress Pandas performance warnings from fragmentatiom

#Import the data

slim_filename = input("Input the filename containing predicted SLiMs:  ")
data_df = pd.read_csv(slim_filename)

input_col_yn = input("Use default column name for SLiM sequence to analyze (Best_SLiM)? (Y/N)  ")
if input_col_yn == "Y": 
	col_with_seq = "Best_SLiM"
else: 
	col_with_seq = input("Input the column name to use:  ")

#Import BioMart sequences

#homologs_filename = input("Input the filename containing BioMart homologs and their sequences in the target species:  ")
#homologs_df = pd.read_csv(homologs_filename, low_memory = False)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Select method for getting homologs

print("---")
print("Homologs may either be obtained by REST (one at a time) or requesting lists from BioMart.")
print("WARNING: Requesting whole lists only works for some species, but not others. E.g. Plant homologs of human proteins cannot be obtained this way.")
print("")
print("To use REST, input \"REST\". To use whole lists, input \"ALL\".")

homolog_request_method = ""
allowed_methods = ["REST", "ALL"]
while homolog_request_method not in allowed_methods: 
	homolog_request_method = input("Enter the method to use (REST or ALL):  ")
	if homolog_request_method not in allowed_methods: 
		print("Invalid response, please try again.")

#Define  function to request data from BioMart in chunks

def dataset_search_chunks(id_list, id_type, dataset, chunk_size, attributes = [], peptide_sequence_mode = False, verbose = False, dataset_verbose = False): 
	print("---")
	print("Dataset being queried:", dataset)
	if dataset_verbose: 
		dataset.verbose = True
	chunk_start_index = 0
	no_more_chunks = False
	if peptide_sequence_mode: 
		collected_response_df = pd.DataFrame(columns = ["peptide", "ensembl_peptide_id"])
	else: 
		collected_response_df = pd.DataFrame(columns = attributes)

	print("Querying", len(id_list), "IDs in chunks of size", chunk_size, "...")
	while not no_more_chunks: 
		if verbose: 
			print("Processing chunk beginning at index:", chunk_start_index)
		
		chunk_end_index = chunk_start_index + chunk_size
		if chunk_end_index > len(source_id_list): 
			id_chunk = id_list[chunk_start_index:]
			no_more_chunks = True
		else: 
			id_chunk = id_list[chunk_start_index:chunk_end_index]

		#Exit loop if the length of the query chunk is zero, which occurs if the length of the whole list is a multiple of the chuck size
		if len(id_chunk) == 0: 
			break

		filters_dict = {id_type:id_chunk}
		
		if peptide_sequence_mode: 
			chunk_response = dataset.search({"filters":filters_dict, "attributes":["peptide", "ensembl_peptide_id"]})
			if verbose: 
				print("filters_dict =", filters_dict)
		else: 
			chunk_response = dataset.search({"filters":filters_dict, "attributes":attributes})
	
		for i, line in enumerate(chunk_response.iter_lines()): 
			line = line.decode().split("\t")
			if peptide_sequence_mode and verbose: 
				print("chunk line", i)
				print(line)
			collected_response_df.loc[len(collected_response_df)] = line
			
		chunk_start_index += chunk_size

		time.sleep(0.2) #Prevents throttling

	return collected_response_df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

#METHOD 1 - Ensembl REST API

if homolog_request_method == "REST": 

	source_species = input("Enter the Ensembl name for the source species (e.g. homo_sapiens):  ")

	gene_ids_in_data = input("Does your input data have Ensembl gene IDs? If not, they will be applied. (Y/N)  ")
	if gene_ids_in_data == "Y": 
		gene_ids_in_data = True
		ensembl_gene_col = input("Enter the column label containing Ensembl gene IDs:  ")
	else: 
		gene_ids_in_data = False
		ensembl_gene_col = "Ensembl_Gene_ID"

	default_peptide_col = input("Default column label for Ensembl peptide IDs is \"Ensembl_ID\". Use default? (Y/N)  ")
	if default_peptide_col == "Y": 
		ensembl_peptide_col = "Ensembl_ID"
	else: 
		ensembl_peptide_col = input("Enter the column label containing Ensembl peptide IDs:  ")

	print("---")

	target_species_list = ListInputter("Input the target species to be analyzed (e.g. homo_sapiens).")
	print("---")

	#------------------------------------------------------------------------------------------

	#Populate gene IDs into data_df

	if not gene_ids_in_data: 
		print("Requesting gene IDs from BioMart...")
		host_names = {
			"Main": "http://www.ensembl.org/biomart", 
			"Metazoa": "http://metazoa.ensembl.org/biomart", 
			"Fungi": "http://fungi.ensembl.org/biomart", 
			"Plants": "http://plants.ensembl.org/biomart", 
			"Protists": "http://protists.ensembl.org/biomart", 
			"Bacteria": "http://bacteria.ensembl.org/biomart"
		}

		print("Biomart servers: ")
		for key, value in host_names.items(): 
			print("\t" + key, ":", value)

		print("Enter the key of the server you wish to use, or leave blank to be prompted for a custom server path.")
		server_key = input("\tInput:  ")

		if server_key in host_names.keys(): 
			source_biomart_host = host_names.get(server_key)
		else: 
			source_biomart_host = input("Enter a custom server: ")

		print("Server is loading...")
		source_biomart_server = BiomartServer(source_biomart_host)
		print("\t...loaded!")

		use_default_dataset = input("Use the default dataset (hsapiens_gene_ensembl)? (Y/N)  ")

		if use_default_dataset == "Y": 
			dataset_name = "hsapiens_gene_ensembl"
		else: 
			dataset_name = "Enter the matching dataset name for source data (e.g. hsapiens_gene_ensembl):  "

		print("Dataset is loading...")
		source_biomart_dataset = source_biomart_server.datasets[dataset_name]
		print("\t...loaded!")

		print("Querying dataset for ensembl_peptide_id and ensembl_gene_id...")
		source_biomart_response = source_biomart_dataset.search({"attributes" : ["ensembl_peptide_id", "ensembl_gene_id"]})
		print("\t...response received")

		source_biomart_rdict = {}
		for line in source_biomart_response.iter_lines(): 
			line = line.decode().split("\t")
			ensp = line[0]
			ensg = line[1]
			if ensp != "" and ensp != None: 
				source_biomart_rdict[ensp] = ensg
		print("\t...and loaded into a dictionary.")

		#Apply the dictionary to the data

		print("Applying to the data to add gene IDs...")

		for i in np.arange(len(data_df)): 
			ensembl_peptide_id = data_df.at[i, ensembl_peptide_col]
			ensembl_gene_id = source_biomart_rdict.get(ensembl_peptide_id)
			data_df.at[i, ensembl_gene_col] = ensembl_gene_id

		print("\t...done!")

	#------------------------------------------------------------------------------------------

	#Retrieve all the homologies for each gene ID, one by one

	gene_id_list = data_df[ensembl_gene_col].tolist()
	gene_id_list = list(dict.fromkeys(gene_id_list)) #removes duplicates

	#Define variables for generating REST links to homology information

	rest_server = "https://rest.ensembl.org/homology/id/"
	rest_type_suffix = "?content-type=application/json;type=orthologues"
	rest_species_syntax = ";target_species="

	rest_species_suffix = ""
	for species in target_species_list: 
		rest_species_suffix = rest_species_suffix + rest_species_syntax + species

	rest_suffix = rest_type_suffix + rest_species_suffix

	#Begin retrieving homologies

	print("Retrieving homologies by REST - this may take a while.")

	gene_homologies_dict = {}
	homologies_verbose = True
	for i, gene_id in enumerate(gene_id_list):
		if gene_id != "None" and gene_id != None: 
			print("Retrieving homology data for", gene_id, "(" + str(i) + " of " + str(len(gene_id_list)) + ")")
			query_url = rest_server + gene_id + rest_suffix
			time1 = time.time()
			response = requests.get(query_url)
			time2 = time.time()
			if not response.ok: 
				response.raise_for_status()
			elif homologies_verbose: 
				print("Response ok. Status code:", response.status_code)
				print("Headers:")
				print(response.headers)
				print("Time taken:", time2 - time1, "seconds")

			decoded = response.json()

			decoded_data = decoded.get("data")
			if len(decoded_data) == 0: 
				decoded_data = {}
				homologies = []
			elif len(decoded_data) == 1: 
				decoded_data = decoded_data[0]
				homologies = decoded_data.get("homologies")
			else: 
				raise Exception("For " + gene_id + " in gene_id_list, decoded_data length was " + str(len(decoded_data)) + " (expected: 1)")

			decoded_size = sys.getsizeof(decoded_data) / 1000
			nominal_speed = decoded_size / (time2 - time1)
			print("Decoded object size:", decoded_size)
			print("Nominal speed (kilobytes per second):", nominal_speed)

			print("\t... retrieved! Data length:", len(homologies))

			gene_homologies_dict[gene_id] = homologies
			time.sleep(0.2) #Prevents throttling

	#------------------------------------------------------------------------------------------

	#Apply the homologies as based on species to be analyzed

	homolog_col_list = [] #Makes a list of tuples consisting of (homolog_col, homolog_col_seq)

	for i in np.arange(len(data_df)): 
		ensembl_gene_id = data_df.at[i, ensembl_gene_col]
		ensembl_peptide_id = data_df.at[i, ensembl_peptide_col]

		homologies = gene_homologies_dict.get(ensembl_gene_id)
		if homologies == None: 
			continue

		#Make a blank dict of number of homologs by target species
		homologs_count_by_species = {}
		for target_species in target_species_list: 
			homologs_count_by_species[target_species] = 0

		#Iterate through homologies
		for homology in homologies: 
			source_protein_id = homology.get("source").get("protein_id")
			target_species = homology.get("target").get("species")

			if source_protein_id == ensembl_peptide_id and target_species in target_species_list: 
				homologs_count_by_species[target_species] += 1
				current_number = homologs_count_by_species.get(target_species)

				target_protein_id = homology.get("target").get("protein_id")
				target_identity = homology.get("target").get("perc_id")
				target_align_seq = homology.get("target").get("align_seq")

				target_sequence = target_align_seq.replace("-", "") #Removes the dashes in the alignment to get the plain sequence

				homolog_cols_tuple = (target_species + "_homolog_" + str(current_number), 
					target_species + "_homolog_" + str(current_number) + "_identity",
					target_species + "_homolog_" + str(current_number) + "_seq")

				data_df.at[i, homolog_cols_tuple[0]] = target_protein_id
				data_df.at[i, homolog_cols_tuple[1]] = target_identity
				data_df.at[i, homolog_cols_tuple[2]] = target_sequence

				homolog_col_list.append(homolog_cols_tuple)

		#Add homolog counts by species to dataframe 
		for target_species, homologs_count in homologs_count_by_species.items(): 
			data_df.at[i, target_species + "_homologs_count"] = homologs_count

	homolog_col_list = list(dict.fromkeys(homolog_col_list)) #removes duplicate tuples

	print(data_df)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

#METHOD 2 - Requesting from Biomart as whole lists

elif homolog_request_method == "ALL": 

	#Declare target databases to use

	databases_in_use = {
		"Main": False, 
		"Metazoa": False, 
		"Fungi": False, 
		"Plants": False, 
		"Protists": False, 
		"Bacteria": False
	}

	print("Please select the target Ensembl databases you intend to query for homolog sequences.")
	for database, value in databases_in_use.items(): 
		use_database = input("Use the Ensembl database \"" + database + "\"? (Y/N)  ")
		if use_database == "Y": 
			if database == "Plants" or database == "Protists" or database == "Bacteria": 
				use_database = input("Warning: As of Ensembl Genes 107, homology is not shown between animals and " + database + ". Do you really want to use it? (Y/N)  ")
				
			if use_database == "Y": 
				databases_in_use[database] = True

	#Select species to query for homology

	db_default_species = {
		"Main": {"Mouse": "mmusculus_gene_ensembl", "Rat": "rnorvegicus_gene_ensembl", "Chicken": "ggallus_gene_ensembl", "Zebrafish": "drerio_gene_ensembl"},
		"Metazoa": {"Drosophila": "dmelanogaster_eg_gene", "Worm": "celegans_eg_gene"}, 
		"Fungi": {"Yeast": "scerevisiae_eg_gene"},
		"Plants": {"Arabidopsis": "athaliana_eg_gene"},
		"Protists": {},
		"Bacteria": {}
	}

	db_selected_species = {}

	for database, use_db in databases_in_use.items(): 
		if use_db: 
			default_datasets_dict = db_default_species.get(database)
			print("The default datasets for", database, "are:")
			print(default_datasets_dict)
			use_default_datasets = input("Use default? (Y/N)  ")
			if use_default_datasets == "Y": 
				inputted_datasets = default_datasets_dict
			else: 
				inputted_datasets = dict_inputter(key_prompt = "Enter key (common name, e.g. Mouse):  ", value_prompt = "Enter dataset name (e.g. mmusculus_gene_ensembl):  ")
			db_selected_species[database] = inputted_datasets

			print("---")

	#Connect to BioMart host

	host_names = {
		"Main": "http://www.ensembl.org/biomart", 
		"Metazoa": "http://metazoa.ensembl.org/biomart", 
		"Fungi": "http://fungi.ensembl.org/biomart", 
		"Plants": "http://plants.ensembl.org/biomart", 
		"Protists": "http://protists.ensembl.org/biomart", 
		"Bacteria": "http://bacteria.ensembl.org/biomart"
	}

	selected_servers = {}

	for database, use_db in databases_in_use.items(): 
		if use_db: 
			host_name = host_names.get(database)
			server = BiomartServer(host_name)
			selected_servers[database] = server

	#Declare source dataset

	use_default_mart = input("The default source mart to use is hsapiens_gene_ensembl. Use this mart? (Y/N)  ")
	if use_default_mart == "Y": 
		mart_to_use = "hsapiens_gene_ensembl"
		matching_server = selected_servers.get("Main")
		source_dataset = matching_server.datasets[mart_to_use]
	else: 
		mart_to_use = input("Input the mart to use:  ")
		matching_server_key = input("Enter the key name for the BioMart database containing this mart (e.g. Main, Metazoa, Fungi, etc.):  ")
		matching_server = selected_servers.get(matching_server_key)
		source_dataset = matching_server.datasets[mart_to_use]

	#Generate attributes to query source dataset for homologs (IDs only)

	def mart_homolog_attributes(mart_datasets):
		mart_species_list = []
		attributes_list = ["ensembl_peptide_id"]
		for mart_code in mart_datasets: 
				mart_species = mart_code.split("_")[0]
				mart_species_list.append(mart_species)
				attributes_list.append(mart_species + "_homolog_ensembl_peptide")
		return mart_species_list, attributes_list

	species_attributes_lists = {}

	for database, use_db in databases_in_use.items(): 
		if use_db: 
			species_datasets_dict = db_selected_species.get(database)
			datasets_list = list(species_datasets_dict.values())
			species_list, attr_list = mart_homolog_attributes(datasets_list)
			species_attributes_lists[database] = (species_list, attr_list) #Generates tuple containing the two lists

	#Query source dataset for homologs (IDs only)

	print("--------------------------------------------")
	print("Querying source dataset for homolog IDs:", source_dataset)

	source_id_list = data_df["Ensembl_ID"].tolist()

	source_response_dfs = {}

	for database, use_db in databases_in_use.items(): 
		if use_db: 
			homolog_attributes_list = species_attributes_lists.get(database)[1]
			homolog_attributes_list.insert(0, "ensembl_peptide_id") #Ensures that the paired query ensembl_peptide_id is returned along with the results
			print("Searching", database, "for inputted IDs...")
			source_response_df = dataset_search_chunks(id_list = source_id_list, id_type = "ensembl_peptide_id", dataset = source_dataset, attributes = homolog_attributes_list, chunk_size = 200)
			source_response_dfs[database] = source_response_df

	print("Received responses with all requested homologs.")

	#Make dictionaries containing unique homolog IDs

	print("Assembling dictionaries of homolog IDs...")

	def homologs_dict_maker(source_identifiers, homolog_attributes):
		homologs_dict = {}

		#Make dictionary of blank lists
		for source_id in source_identifiers: 
			attributes_dict = {}
			for homolog_attribute in homolog_attributes: 
				attributes_dict[homolog_attribute] = []
			homologs_dict[source_id] = attributes_dict

		#Add homolog IDs to the dictionary, including only unique values, where key = source ensembl id, and value = a dict containing the homolog_attribute as the key and a list of homologs as the value
		source_response_df = source_response_dfs.get(database)
		print("source_response_df =")
		print(source_response_df)
		for i in np.arange(len(source_response_df)): 
			source_id = source_response_df.at[i, "ensembl_peptide_id"]
			if source_id is not str: 
				source_id = source_response_df.at[i, "ensembl_peptide_id"][0]
			print("source_id =")
			print(source_id)
			print("source_id type =", type(source_id))

			current_dict = homologs_dict.get(source_id)
			for homolog_attribute in homolog_attributes:
				print("homolog_attribute =", homolog_attribute)
				current_homolog = source_response_df.at[i, homolog_attribute]
				print(type(current_homolog))
				print("current_homolog =", current_homolog)
				existing_homologs_list = current_dict.get(homolog_attribute)
				print("existing_homologs_list =")
				print(existing_homologs_list)
				if current_homolog not in existing_homologs_list: 
					existing_homologs_list.append(current_homolog)
					current_dict[homolog_attribute] = existing_homologs_list #Updates the list
			homologs_dict[source_id] = current_dict #Updates the dict containing updated lists

		return homologs_dict

	db_homologs_dicts = {}

	for database, use_db in databases_in_use.items(): 
		if use_db: 
			db_attributes = species_attributes_lists.get(database)[1]
			db_homologs_dict = homologs_dict_maker(source_identifiers = source_id_list, homolog_attributes = db_attributes)
			db_homologs_dicts[database] = db_homologs_dict

	print("Done!")

	#Apply the homolog IDs to the main dataframe

	homologs_dicts_by_db = {}

	for database, use_db in databases_in_use.items(): 
		if use_db: 
			db_homologs_dict = db_homologs_dicts.get(database)
			db_attributes = species_attributes_lists.get(database)[1]

			all_db_homologs_dict = {} #Make a running list of homologs from each species (attribute in the form of species_)
			for homolog_attribute in db_attributes:
				all_db_homologs_dict[homolog_attribute] = []

			for i in np.arange(len(data_df)): 
				source_ensembl_id = data_df.at[i, "Ensembl_ID"]
				#print("Current ID:", source_ensembl_id)

				matching_homologs_dict = homologs_dict.get(source_ensembl_id)

				for homolog_attribute in db_attributes:
					#print("Current homolog_attribute:", homolog_attribute)
					running_list_for_homolog = all_db_homologs_dict.get(homolog_attribute)
					target_homolog_list = matching_homologs_dict.get(homolog_attribute)
					for j, homolog in enumerate(target_homolog_list): 
						#print("Current homolog:", homolog)
						data_df.at[i, homolog_attribute + "_" + str(j + 1)] = homolog
						if homolog not in running_list_for_homolog: 
							running_list_for_homolog.append(homolog)
					all_db_homologs_dict[homolog_attribute] = running_list_for_homolog
					data_df.at[i, homolog_attribute + "_count"] = len(target_homolog_list)

			homologs_dicts_by_db[database] = all_db_homologs_dict

			print("Homolog IDs were applied for those belonging to the database:", database)

	#Create a dictionary of dictionaries of homolog sequences

	print("Beginning retrieval of homolog sequences...")

	db_homolog_sequences_dict = {}

	for database, use_db in databases_in_use.items(): 
		if use_db: 
			db_server = selected_servers.get(database)
			homolog_sequences_dict = {} #Declare the empty dictionary

			db_attributes = species_attributes_lists.get(database)
			for homolog_attribute in db_attributes: 
				homologs_list = all_db_homologs_dict.get(homolog_attribute)
				homolog_species = homolog_attribute.split("_")[0]
				print("Querying dataset for species:", homolog_species)
				
				#Find the matching Ensembl BioMart dataset code
				selected_species_in_db = db_selected_species.get(database)
				for mart_name, mart_code in selected_species_in_db.items(): 
					mart_species = mart_code.split("_")[0]
					if mart_species == homolog_species: 
						homolog_mart_code = mart_code

				#Retrieve matching dataset and make the request
				homolog_mart_dataset = db_server.datasets[homolog_mart_code]
				mart_response_df = dataset_search_chunks(id_list = homologs_list, id_type = "ensembl_peptide_id", dataset = homolog_mart_dataset, chunk_size = 200, peptide_sequence_mode = True)

				mart_response_dict = {}
				for i in np.arange(len(mart_response_df)): 
					key = mart_response_df.at[i, "ensembl_peptide_id"]
					value = mart_response_df.at[i, "peptide"]
					mart_response_dict[key] = value

				homolog_sequences_dict[homolog_attribute] = mart_response_dict

			db_homolog_sequences_dict[database] = homolog_sequences_dict

	print("Retrieved all the homolog sequences into a dictionary-of-dictionaries-of-dictionaries.")

	#Assign sequences to main dataframe

	print("Assigning sequences to the main dataframe...")
	for database, use_db in databases_in_use.items(): 
		if use_db: 
			homolog_sequences_dict = db_homolog_sequences_dict.get(database)
			db_attributes = species_attributes_lists.get(database)

			for i in np.arange(len(data_df)): 
				for homolog_attribute in db_attributes:
					homologs_count = data_df.at[i, homolog_attribute + "_count"]
					sequences_dict = homolog_sequences_dict.get(homolog_attribute)
					for j in np.arange(1, homologs_count + 1): 
						homolog_id = data_df.at[i, homolog_attribute + "_" + str(j)]
						homolog_sequence = sequences_dict.get(homolog_id)
						data_df.at[i, homolog_attribute + "_" + str(j) + "_sequence"]

	print("Done!")

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Define functions for finding homologous SLiMs

def slim_pairwise_homology(slim_sequence, target_sequence, open_gap_penalty = "Default", extend_gap_penalty = "Default"): 
	print("\tStarting slim_pairwise_homology()")
	#Find the contiguous homologous SLiM

	if open_gap_penalty == "Default":
		open_gap_penalty = -1 * len(slim_sequence)
	if extend_gap_penalty == "Default":
		extend_gap_penalty = -1 * len(slim_sequence)

	slim_alignments_xs = pairwise2.align.globalxs(slim_sequence, target_sequence, open_gap_penalty, extend_gap_penalty, penalize_end_gaps = False)

	seqA = slim_alignments_xs[0][0]
	seqB = slim_alignments_xs[0][1]

	#Find index for start of homologous aligned SLiM
	for k, char in enumerate(seqA): 
		if char != "-": 
			alignment_start = k
			break

	homolog_slim_xs = seqB[alignment_start : alignment_start + len(slim_sequence)]
	if homolog_slim_xs == "": 
		homolog_slim_xs = "None found in homolog"

	print("\thomolog_slim_xs =", homolog_slim_xs)

	#For assessing the quality of the SLiM alignment
	align_identical = FindIdenticalResidues(seqA, seqB)
	print("\talign_identical =", align_identical)
	align_identity_ratio = align_identical / len(slim_sequence)
	print("\talign_identity_ratio =", align_identity_ratio)

	return homolog_slim_xs, align_identical, align_identity_ratio

def parent_pairwise_homology(sequence_A, sequence_B):
	#For source protein to homologous target protein
	protein_alignments_xx = pairwise2.align.globalxx(sequence_A, sequence_B)
	protein_align_score = protein_alignments_xx[0][2]
	protein_align_score_ratio = protein_align_score / len(sequence_A)
	protein_align_identical = FindIdenticalResidues(protein_alignments_xx[0][0], protein_alignments_xx[0][1])
	protein_align_identity_ratio = protein_align_identical / len(sequence_A)

	return protein_align_identical, protein_align_identity_ratio

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
			homolog_slim, slim_align_identical, slim_align_identity_ratio = slim_pairwise_homology(slim_sequence = slim_seq, target_sequence = target_homolog_seq)
			
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

