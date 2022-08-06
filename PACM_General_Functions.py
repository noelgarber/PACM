#This script contains general functions and methods used in other scripts in this tool. 

import os
import math
import requests
from PACM_General_Vars import aa_charac_dict #dictionary of chemical characteristics of 20 amino acids

def FilenameSubdir(subdirectory, filename):
	curr_dir = os.getcwd()
	dest_subdir = os.path.join(curr_dir, subdirectory)
	if not os.path.exists(dest_subdir):
		os.makedirs(dest_subdir)
	dest_name = os.path.join(dest_subdir, filename)
	return dest_name

def Log2fc(mean1, mean2): 
	value = math.log2(mean1 + 0.01) - math.log2(mean2 + 0.01)
	return value

def BaitPermuter(bait_list, text_to_append): 
	output_list = []
	for bait1 in bait_list: 
		for bait2 in bait_list: 
			if bait1 == bait2: 
				continue
			else: 
				output_list.append(bait1 + "_" + bait2 + "_" + text_to_append)
	return output_list

def MoveCol(df, cols_to_move=[], ref_col='', place="After"):
	cols = df.columns.tolist()
	if place == "After":
		seg1 = cols[:list(cols).index(ref_col) + 1]
		seg2 = cols_to_move
	if place == "Before":
		seg1 = cols[:list(cols).index(ref_col)]
		seg2 = cols_to_move + [ref_col]

	seg1 = [i for i in seg1 if i not in seg2]
	seg3 = [i for i in cols if i not in seg1 + seg2]

	return(df[seg1 + seg2 + seg3])

def CharacAA(amino_acid): 
	for charac, mem_list in aa_charac_dict.items(): 
		if amino_acid in mem_list: 
			charac_result = charac
	return charac_result

def MapRetrieve(ids2map, source_fmt = "ENSEMBL_PROTEIN", target_fmt = "UniProtKB", output_fmt = "list"):

		# Map database identifiers from/to UniProt accessions.
		# The mapping is achieved using the RESTful mapping service provided by
		# UniProt. While a great many identifiers can be mapped the documentation
		# has to be consulted to check which options there are and what the database
		# codes are. Mapping UniProt to UniProt effectlvely allows batch retrieval
		# of entries.
		# Args:
		# ids2map (list or string): identifiers to be mapped
		# source_fmt (str, optional): format of identifiers to be mapped.
		# Defaults to ACC+ID, which are UniProt accessions or IDs.
		# target_fmt (str, optional): desired identifier format. Defaults
		# to ACC, which is UniProt accessions.
		# output_fmt (str, optional): return format of data. Defaults to list.
		# Returns:
		# mapped identifiers (str)

	BASE = "https://www.uniprot.org"
	TOOL_ENDPOINT = "/uploadlists/"

	if hasattr(ids2map, "pop"):
		ids2map = " ".join(ids2map)
	payload = {"from": source_fmt,
	"to": target_fmt,
	"format": output_fmt,
	"query": ids2map,
	}
	response = requests.get(BASE + TOOL_ENDPOINT, params = payload)
	if response.ok:
		return response.text
	else:
		response.raise_for_status()

def ListInputter(prompt): 
	print(prompt)
	lst = []
	more_entries = True
	while more_entries == True:
		next_entry = input("Next entry:  ")
		if next_entry == "": 
			more_entries = False
		else: 
			lst.append(next_entry)
	return lst

def NumInput(message_string):
	float_inputted = False
	while not float_inputted: 
		try: 
			value = input(message_string + "  ")
			value = float(value)
			float_inputted = True
		except: 
			print("Error! Please try again.")
	return value