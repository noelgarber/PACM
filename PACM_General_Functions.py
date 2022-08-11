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

def CharacAA(amino_acid, dict_of_aa_characs = aa_charac_dict): 
	for charac, mem_list in dict_of_aa_characs.items(): 
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

def NumInput(message_string, use_int = False):
	if use_int: 
		int_inputted = False
		while not int_inputted: 
			try: 
				value = input(message_string + "  ")
				value = int(value)
				int_inputted = True
			except: 
				print("Not an integer! Please try again.")
	else: 
		float_inputted = False
		while not float_inputted: 
			try: 
				value = input(message_string + "  ")
				value = float(value)
				float_inputted = True
			except: 
				print("Not a number! Please try again.")
	return value

def NumberedList(length)
	numbered_list = []
	for i in range(1, length + 1): 
		numbered_list.append("#" + str(i))
	return numbered_list
	#Makes a list of strings from "#1" to "#(length)"

#Define functions to compute sensitivity, specificity, PPV, NPV

def XDivYZ(X, Y, Z, inf_value = 999): 
	try: 
		value = X / (Y + Z)
	except: 
		value = inf_value
	return value

def PredVal(score_threshold, dataframe): 
	pred_val_dict = {
		"TP": 0,
		"FP": 0,
		"TN": 0,
		"FN": 0,
	}
	for i in np.arange(len(dataframe)): 
		sig_truth = dataframe.at[i, "Significant"]
		score = dataframe.at[i, "FFAT_Score"]
		if score >= score_threshold: 
			score_over_n = "Yes"
		else: 
			score_over_n = "No"

		if score_over_n == "Yes" and sig_truth == "Yes": 
			pred_val_dict["TP"] = pred_val_dict["TP"] + 1
		elif score_over_n == "Yes" and sig_truth == "No": 
			pred_val_dict["FP"] = pred_val_dict["FP"] + 1
		elif score_over_n != "Yes" and sig_truth == "Yes": 
			pred_val_dict["FN"] = pred_val_dict["FN"] + 1
		elif score_over_n != "Yes" and sig_truth == "No": 
			pred_val_dict["TN"] = pred_val_dict["TN"] + 1

	pred_val_dict["Sensitivity"] = round(XDivYZ(pred_val_dict.get("TP"), pred_val_dict.get("TP"), pred_val_dict.get("FN")), 3)
	pred_val_dict["Specificity"] = round(XDivYZ(pred_val_dict.get("TN"), pred_val_dict.get("TN"), pred_val_dict.get("FP")), 3)
	pred_val_dict["PPV"] = round(XDivYZ(pred_val_dict.get("TP"), pred_val_dict.get("TP"), pred_val_dict.get("FP")), 3)
	pred_val_dict["NPV"] = round(XDivYZ(pred_val_dict.get("TN"), pred_val_dict.get("TN"), pred_val_dict.get("FN")), 3)

	return pred_val_dict