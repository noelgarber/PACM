#For SLiMs that occur in membrane proteins, this script tests whether the SLiMs are on the cytosol-facing toplogical domains

import numpy as np
import pandas as pd
import json
import requests
import time
from PACM_General_Functions import FilenameSubdir, MapRetrieve

data_file_path = input("Please enter the path of the file you wish to process:  ")
print("Path:", data_file_path)

data_df = pd.read_csv(data_file_path)

#----------------------------------------------------------------------

#Convert each protein's Ensembl ID to a UniProt accession number

ensembl_id_list = data_df["Ensembl_ID"].tolist()
ensembl_id_list = list(dict.fromkeys(ensembl_id_list)) #Removes duplicates

URL = 'https://rest.uniprot.org/idmapping'

params = {
   'from': 'Ensembl_Protein',
   'to': 'UniProtKB',
   'ids': ' '.join(ensembl_id_list)
}

response = requests.post(f'{URL}/run', params)
job_id = response.json()['jobId']
job_status = requests.get(f'{URL}/status/{job_id}')
d = job_status.json()

ensembl_to_uniprot_dict = {}

#Retry getting results every 1 sec until they are ready
loop_end = ""
while loop_end != "Done":
   if d.get("job_status") == 'FINISHED' or d.get("results"):
      loop_end = "Done"
      job_results = requests.get(f'{URL}/results/{job_id}/?size=500')
      print("Request for Ensembl-to-UniProt IDs was successful!")
      results = job_results.json()
      for obj in results["results"]:
         ensembl_to_uniprot_dict[obj["from"]] = obj["to"]
         #print(f'{obj["from"]}\t{obj["to"]}')
      break
   time.sleep(1)

#----------------------------------------------------------------------

#Assign UniProt IDs by Ensembl ID

for i in np.arange(len(data_df)): 
	ensembl_id = data_df.at[i, "Ensembl_ID"]
	uniprot_id = ensembl_to_uniprot_dict.get(ensembl_id, "None") #Returns None when no UniProt ID is available
	data_df.at[i, "UniProt_ID"] = uniprot_id

#----------------------------------------------------------------------

#Retrieve topological information from UniProt about whether a SLiM faces the cytosol

#Define a function for returning localizations as a piped string
def list_localizations(comments_dict): 
	locs = ""
	for comment_dict in comments_dict: 
		comment_type = comment_dict.get("commentType")
		if comment_type == "SUBCELLULAR LOCATION": 
			locations_list_dicts = comment_dict.get("subcellularLocations")
			for location_info_dict in locations_list_dicts: 
				location_dict = location_info_dict.get("location")
				location = location_dict.get("value")
				locs = locs + location + " | "
	return locs

#Define a function that returns combined protein type; must be passed in a loop when multiple TMDs could be present, see below

def protein_type_assigner(dict_of_features): 
	var = ""
	if dict_of_features is not None: 
		for feature_dict in dict_of_features: 
			feature_type = feature_dict.get("type")
			if feature_type in ["Transmembrane", "Intramembrane"]: 
				if var == "": 
					var = feature_type #Sets protein type to either Transmembrane or Intramembrane
				elif var == "Transmembrane" and feature_type == "Intramembrane": 
					var = "Transmembrane/Intramembrane"
				elif var == "Intramembrane" and feature_type == "Transmembrane": 
					var = "Transmembrane/Intramembrane"
	return var

#Define a function that assigns topology information for where a motif occurs in a membrane protein
def motif_topo_assigner(slim_start_num, slim_end_num, features_dict, output_index, output_column, output_df): 
	if features_dict is not None: 
		for feature_dict in features_dict: 
			feature_type = feature_dict.get("type")

			if feature_type in ["Transmembrane", "Intramembrane", "Topological domain"]: 
				#print("Feature being processed:", feature_type)
				feature_start = feature_dict.get("location").get("start").get("value")
				feature_end = feature_dict.get("location").get("end").get("value")

				if feature_start is not None and feature_end is not None: 
					#print("feature_start and feature_end both exist")
					feature_positions = np.arange((feature_start - 1), (feature_end + 1) + 1)
					#print("positions within feature:")
					#print(feature_positions)
					#print("Current SLiM start/end:", slim_start_num, "to", slim_end_num)
					feature_description = feature_dict.get("description")
					#print("feature description:", feature_description)

					if slim_start_num in feature_positions and slim_end_num in feature_positions: 
						print("SLiM is in feature of type", feature_type)
						if feature_type == "Topological domain": 
							output_df.at[output_index, output_column] = feature_description
							print(output_column, "set to", feature_description)
						elif feature_type == "Transmembrane": 
							output_df.at[output_index, output_column] = "TMD"
							print(output_column, "set to TMD")
						elif feature_type == "Intramembrane": 
							output_df.at[output_index, output_column] = "IMD"
							print(output_column, "set to IMD")			

#Define a function that requests UniProt REST data and uses it to assign topological information to the motif of interest
def response_to_df(index, resp, slim, dataframe): 
	if resp.ok: 
		response_json = response.json() #Creates a dict of dicts

		comments_list_dicts = response_json.get("comments")
		if comments_list_dicts is not None:
			localizations = list_localizations(comments_list_dicts)
			dataframe.at[index, "UniProt_Localization"] = localizations

		uniprot_sequence = response_json.get("sequence").get("value")
		slim_length = len(slim)

		#Assign the type of protein (Transmembrane, Intramembrane, both, or neither)
		features_list_dicts = response_json.get("features")
		protein_type = protein_type_assigner(features_list_dicts)
		dataframe.at[index, "Type"] = protein_type

		#Check if the SLiM exists in the uniprot sequence for testing topology
		slim_found = False
		for i in np.arange(len(uniprot_sequence)):
			if slim == uniprot_sequence[i : i + slim_length]: 
				slim_start = i + 1
				slim_end = slim_start + slim_length
				slim_found = True

		#Assign where the SLiM occurs in the protein
		if slim_found: 
			motif_topo_assigner(slim_start_num = slim_start, slim_end_num = slim_end, features_dict = features_list_dicts, output_index = index, output_column = "Best_SLiM_Topology", output_df = dataframe)
		else: 
			print("SLiM", slim, "not found in provided sequence! Can be caused by Ensembl/UniProt discrepancy.")
			dataframe.at[index, "Best_SLiM_Topology"] = "SLiM not found (likely due to UniProt sequence discrepancy)"

#Loop through the dataframe to apply the function
for i in np.arange(len(data_df)): 
	uniprot_id = data_df.at[i, "UniProt_ID"]
	best_SLiM = data_df.at[i, "Best_SLiM"]
	best_core_SLiM = best_SLiM[6:13]
	print("Requesting and checking", uniprot_id, "(", best_core_SLiM, ") -", i + 1, "of", len(data_df))

	if uniprot_id != "None": 
		query_url = "https://rest.uniprot.org/uniprotkb/" + uniprot_id
		response = requests.get(query_url)
		response_to_df(index = i, resp = response, slim = best_core_SLiM, dataframe = data_df)
	else: 
		data_df.at[i, "Type"] = "Unknown - No ID"
		data_df.at[i, "Best_SLiM_Topology"] = "Unknown - No ID"
		print("^ No UniProt ID")

#----------------------------------------------------------------------

#Output files to Output folder

output_path_csv = data_file_path[:-4] + "_with_Topology.csv"
output_path_xlsx = data_file_path[:-4] + "_with_Topology.xlsx"
data_df.to_csv(output_path_csv)
data_df.to_excel(output_path_xlsx)
print("Saved! Paths:", output_path_csv, "and", output_path_xlsx)