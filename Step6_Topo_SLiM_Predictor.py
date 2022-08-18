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

#Define a function that assigns topology information for where a motif occurs in a membrane protein
def motif_topo_assigner(motif_seq, uni_seq, feat_type, feat_dict, dest_dataframe, df_column): 
	location_dict = feat_dict.get("location")
	description = feat_dict.get("description")

	start_position = location_dict.get("start").get("value")
	end_position = location_dict.get("end").get("value")

	feature_sequence_plusone = uni_seq[int(start_position) - 2 : int(end_position) + 1]

	if motif_seq in feature_sequence_plusone:
		if feat_type == "Topological domain": 
			dest_dataframe.at[i, df_column] = description
		elif feat_type == "Transmembrane": 
			dest_dataframe.at[i, df_column] = "TM - " + description
		elif feat_type == "Intramembrane": 
			dest_dataframe.at[i, df_column] = "IM - " + description

#Define a function that returns combined protein type; must be passed in a loop when multiple TMDs could be present, see below
def protein_type_assigner(feat_type, prot_type_var):
	if prot_type_var == "": 
		val = feat_type
	elif prot_type_var == "Transmembrane" and feat_type == "Intramembrane": 
		val = "Transmembrane/Intramembrane"
	elif prot_type_var == "Intramembrane" and feat_type == "Transmembrane": 
		val = "Transmembrane/Intramembrane"
	else: 
		val = prot_type_var
	return val

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

#Define a function that requests UniProt REST data and uses it to assign topological information to the motif of interest
def response_to_df(resp, slim, dataframe): 
	if resp.ok: 
		response_json = response.json() #Creates a dict of dicts

		comments_list_dicts = response_json.get("comments")
		if comments_list_dicts is not None:
			localizations = list_localizations(comments_list_dicts)
			dataframe.at[i, "UniProt_Localization"] = localizations

		uniprot_sequence = response_json.get("sequence").get("value")

		features_list_dicts = response_json.get("features")
		if features_list_dicts is not None:
			protein_type = ""
			for feature_dict in features_list_dicts: 
				feature_type = feature_dict.get("type")

				if feature_type in ["Transmembrane", "Intramembrane", "Topological domain"]:
					motif_topo_assigner(slim, uniprot_sequence, feature_type, feature_dict, dataframe, "Best_SLiM_Topology")

				if feature_type in ["Transmembrane", "Intramembrane"]: 
					protein_type = protein_type_assigner(feature_type, protein_type)

			dataframe.at[i, "Type"] = protein_type

#Loop through the dataframe to apply the function
for i in np.arange(len(data_df)): 
	uniprot_id = data_df.at[i, "UniProt_ID"]
	best_SLiM = data_df.at[i, "Best_SLiM"]
	best_core_SLiM = best_SLiM[6:13]
	print("Inspecting", uniprot_id, "(", best_core_SLiM, ") -", i + 1, "of", len(data_df))

	if uniprot_id != "None": 
		query_url = "https://rest.uniprot.org/uniprotkb/" + uniprot_id
		response = requests.get(query_url)
		response_to_df(response, best_core_SLiM, data_df)
		print("Checked", i, "of", len(data_df))
	else: 
		data_df.at[i, "Type"] = "Unknown - No ID"
		data_df.at[i, "Best_SLiM_Topology"] = "Unknown - No ID"
		print("Checked", i, "of", len(data_df), "- No UniProt ID")

#----------------------------------------------------------------------

#Output files to Output folder

output_path_csv = data_file_path[:-4] + "_with_Topology.csv"
output_path_xlsx = data_file_path[:-4] + "_with_Topology.xlsx"
data_df.to_csv(output_path_csv)
data_df.to_excel(output_path_xlsx)
print("Saved! Paths:", output_path_csv, "and", output_path_xlsx)