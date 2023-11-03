# For SLiMs that occur in membrane proteins, this script tests whether the SLiMs are on the cytosol-facing toplogical domains

import numpy as np
import pandas as pd
import requests
import time
from Motif_Predictor.predictor_config import predictor_params

def convert_ensembl(data_df, predictor_params = predictor_params):
    '''
    Function that converts Ensembl Protein IDs to UniProt IDs which will be used for pulling topology information

    Args:
        data_df (pd.DataFrame):  dataframe containing protein sequences and scored motifs
        predictor_params (dict): dict of user-defined parameters for the motif prediction workflow

    Returns:
        ensembl_uniprot_dict (dict): dictionary of Ensembl Protein ID --> UniProt ID
    '''

    # Extract Ensembl IDs from dataframe and remove duplicates
    if "Ensembl_Protein_ID" in data_df.columns:
        ensembl_ids = data_df["Ensembl_Protein_ID"].to_numpy()
    elif "Ensembl_ID" in data_df.columns:
        ensembl_ids = data_df["Ensembl_ID"].to_numpy()
    elif "Gene" in data_df.columns:
        ensembl_ids = data_df["Gene"].to_numpy()
    else:
        raise IndexError("convert_ensembl error: could not find Ensembl_ID or Ensembl_Protein_ID col in input data_df")
    unique_ensembl_ids = list(np.unique(ensembl_ids))

    # Define UniProt accession params
    URL = "https://rest.uniprot.org/idmapping"

    params = {"from": "Ensembl_Protein",
              "to": "UniProtKB",
              "ids": " ".join(unique_ensembl_ids)}

    response = requests.post(f'{URL}/run', params)
    job_id = response.json()["jobId"]
    job_status = requests.get(f'{URL}/status/{job_id}')
    d = job_status.json()

    ensembl_uniprot_dict = {}

    # Retry getting results periodically until they are ready
    uniprot_refresh_time = predictor_params["uniprot_refresh_time"]
    response_received = False
    while not response_received:
       if d.get("job_status") == 'FINISHED' or d.get("results"):
          response_received = True
          job_results = requests.get(f'{URL}/results/{job_id}/?size=500')
          print("Request for Ensembl-to-UniProt IDs was successful!")
          results = job_results.json()
          for obj in results["results"]:
             ensembl_uniprot_dict[obj["from"]] = obj["to"]
       else:
           time.sleep(uniprot_refresh_time)

    return ensembl_uniprot_dict, ensembl_ids

def assign_uniprot(data_df, predictor_params = predictor_params):
    '''
    Function to assign UniProt IDs to proteins in the dataframe

    Args:
        data_df (pd.DataFrame):  dataframe containing protein sequences and scored motifs
        predictor_params (dict): dict of user-defined parameters for the motif prediction workflow

    Returns:
        data_df (pd.DataFrame):  dataframe with added UniProt ID column beside Ensembl ID column
    '''

    # Insert blank Uniprot ID column
    cols = list(data_df.columns)

    if "Protein_ID" in cols:
        ensembl_col_idx = cols.index("Protein_ID")
    elif "Ensembl_Protein_ID" in cols:
        ensembl_col_idx = cols.index("Ensembl_Protein_ID")
    elif "Ensembl_ID" in cols:
        ensembl_col_idx = cols.index("Ensembl_ID")
    else:
        raise Exception("Ensembl protein ID could not be found in dataframe cols")

    cols.insert(ensembl_col_idx+1, "Uniprot_ID")

    data_df["Uniprot_ID"] = ""
    data_df = data_df[cols]

    # Get dict of Ensembl Protein ID --> Uniprot ID and map it onto the dataframe
    ensembl_uniprot_dict, ensembl_ids = convert_ensembl(data_df, predictor_params)
    uniprot_ids = [ensembl_uniprot_dict.get(ensembl_id) for ensembl_id in ensembl_ids]
    data_df["Uniprot_ID"] = uniprot_ids

    return data_df

def list_localizations(comments_dict):
    # Helper function that returns localizations as a piped string from a dict of comments

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

def protein_type_assigner(dict_of_features):
    '''
    Function that returns combined protein type; must be passed in a loop when multiple TMDs could be present

    Args:
        dict_of_features (dict): dictionary of features

    Returns:
        combined_protein_type (str): one of "Transmembrane", "Intermembrane", "Transmembrane/Intermembrane", or ""
    '''

    combined_protein_type = ""
    if dict_of_features is not None:
        for feature_dict in dict_of_features:
            feature_type = feature_dict.get("type")
            if feature_type in ["Transmembrane", "Intramembrane"]:
                if combined_protein_type == "":
                    combined_protein_type = feature_type # Sets protein type to either Transmembrane or Intramembrane
                elif combined_protein_type == "Transmembrane" and feature_type == "Intramembrane":
                    combined_protein_type = "Transmembrane/Intramembrane"
                elif combined_protein_type == "Intramembrane" and feature_type == "Transmembrane":
                    combined_protein_type = "Transmembrane/Intramembrane"

    return combined_protein_type

def motif_topo_assigner(motif_start_position, motif_end_position, features_dict, output_index, output_col, output_df,
                        verbose = False):
    '''
    Function that assigns motif topology calls into the dataframe for a given motif based on its start and end positions

    Args:
        motif_start_position (int): motif start point in protein sequence
        motif_end_position (int):   motif end point in protein sequence
        features_dict (dict):       dictionary of features
        output_index (int):         destination row index
        output_col (str):           destination column name
        output_df (pd.DataFrame):   destination dataframe
        verbose (bool):             whether to display verbose messages

    Returns:
        None; operation is performed in-place
    '''

    if features_dict is not None:
        for feature_dict in features_dict:
            feature_type = feature_dict.get("type")

            if feature_type in ["Transmembrane", "Intramembrane", "Topological domain"]:
                feature_start = feature_dict.get("location").get("start").get("value")
                feature_end = feature_dict.get("location").get("end").get("value")

                if feature_start is not None and feature_end is not None:
                    feature_positions = np.arange((feature_start - 1), (feature_end + 1) + 1)
                    feature_description = feature_dict.get("description")

                    if motif_start_position in feature_positions and motif_end_position in feature_positions:
                        print("Motif is in feature of type", feature_type) if verbose else None
                        if feature_type == "Topological domain":
                            output_df.at[output_index, output_col] = feature_description
                            print(output_col, "set to", feature_description) if verbose else None
                        elif feature_type == "Transmembrane":
                            output_df.at[output_index, output_col] = "TMD"
                            print(output_col, "set to TMD") if verbose else None
                        elif feature_type == "Intramembrane":
                            output_df.at[output_index, output_col] = "IMD"
                            print(output_col, "set to IMD") if verbose else None

def response_to_df(index, resp, motif_cols, dataframe, predictor_params = predictor_params):
    '''
    Function that requests UniProt REST data and uses it to assign topological information to the motif of interest

    Args:
        index (int):              current row index in dataframe
        resp (requests.Response): server response
        motif_cols (list):        list of columns holding motif sequences
        dataframe (pd.DataFrame): main dataframe
        predictor_params (dict):  dictionary of user-defined parameters for the motif prediction workflow

    Returns:
        None; operation is performed in-place
    '''

    if resp.ok:
        response_json = resp.json() # creates a dict of dicts

        comments_list_dicts = response_json.get("comments")
        if comments_list_dicts is not None:
            localizations = list_localizations(comments_list_dicts)
            dataframe.at[index, "UniProt_Localization"] = localizations

        # Assign the type of protein (Transmembrane, Intramembrane, both, or neither)
        features_list_dicts = response_json.get("features")
        protein_type = protein_type_assigner(features_list_dicts)
        dataframe.at[index, "Type"] = protein_type

        uniprot_sequence = response_json.get("sequence").get("value")

        core_start = predictor_params["core_start"]
        core_end = predictor_params["core_end"]

        for motif_col in motif_cols:
            motif_seq = dataframe.at[index, motif_col]
            motif_seq_core = motif_seq[core_start:core_end+1]
            motif_length = len(motif_seq_core)

            # Check if the SLiM exists in the uniprot sequence for testing topology and assign motif location info
            motif_found = motif_seq_core in uniprot_sequence
            output_col = motif_col + "_Topology"
            if motif_found:
                motif_start = uniprot_sequence.index(motif_seq_core)
                motif_end = motif_start + motif_length
                motif_topo_assigner(motif_start, motif_end, features_list_dicts, index, output_col, dataframe)
            else:
                dataframe.at[index, output_col] = "Motif not found (usually due to UniProt/Ensembl sequence discrepancy"

    else:
        print(f"Error {resp.status_code} at row {index} while requesting topology information from Uniprot")

def predict_topology(data_df, motif_cols, predictor_params = predictor_params):
    '''
    Main function to predict motif topologies

    Args:
        data_df (pd.DataFrame):   main dataframe containing protein sequences and predicted motifs
        motif_cols (list):        list of column names referring to motif sequences for topological analysis
        predictor_params (dict):  dictionary of user-defined parameters for the motif prediction workflow

    Returns:
        data_df (pd.DataFrame):   data_df with topology columns added
    '''

    protein_count = len(data_df)

    # Assign Uniprot IDs to Ensembl proteins
    data_df = assign_uniprot(data_df, predictor_params)

    # Loop through the dataframe to apply the function
    for i in np.arange(protein_count):
        uniprot_id = data_df.at[i, "Uniprot_ID"]

        if uniprot_id != "None" and uniprot_id != "":
            query_url = "https://rest.uniprot.org/uniprotkb/" + uniprot_id
            response = requests.get(query_url)
            response_to_df(i, response, motif_cols, data_df)

    # Save results
    output_path = predictor_params["protein_seqs_path"][:-4] + "_with_Topology.csv"
    data_df.to_csv(output_path)
    print(f"Saved topology results to {output_path}")

    return data_df