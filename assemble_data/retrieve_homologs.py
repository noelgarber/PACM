# This script retrieves homolog sequences from NCBI HomoloGene for given taxonomic identifiers (TaxIDs)

import numpy as np
import pandas as pd
import os
import pickle
import time
from io import StringIO
from Bio import Entrez, SeqIO
Entrez.email = "ngarber93@gmail.com"

def read_homologene(homologene_data_path):
    '''
    Loads homologene.data (from the NCBI FTP site) as a pandas dataframe

    Args:
        homologene_data_path (str): local path to homologene.data

    Returns:
        homologene_data (pd.DataFrame): homologene data in a dataframe
    '''

    homologene_data = pd.read_csv(homologene_data_path, header=None, sep="\t")
    columns = ["hid", "taxid", "gene_id", "gene_name", "protein_gi", "protein_accession"]
    homologene_data.columns = columns
    return homologene_data

def get_homolog_ids(homologene_data, reference_taxid = 9606, target_taxids = (3702,)):
    '''
    Retrieves homolog IDs matching target TaxIDs and the reference TaxID

    Args:
        homologene_data_df (pd.DataFrame): df containing data from the homologene.data file available over NCBI FTP
        reference_taxid (int):             taxonomic identifier for the reference species (e.g. human = 9606)
        target_taxids (list|tuple):        taxonomic identifiers for the target species set

    Returns:
        homologs (dict):                   dictionary of reference_protein_id --> homolog_taxid --> matching_ids
    '''

    reference_df = homologene_data[homologene_data["taxid"].eq(reference_taxid)]

    # Retrieve target species dataframes
    target_dfs = {}
    for target_taxid in target_taxids:
        target_df = homologene_data[homologene_data["taxid"].eq(target_taxid)]
        target_dfs[target_taxid] = target_df

    # Construct a non-redundant homolog dictionary
    reference_protein_ids = reference_df["protein_accession"].to_list()
    homologs = {}
    for reference_protein_id in reference_protein_ids:
        reference_entries_df = reference_df[reference_df["protein_accession"].eq(reference_protein_id)]
        reference_protein_hids = reference_entries_df["hid"].to_list()

        target_matches = {}

        for reference_protein_hid in reference_protein_hids:
            for target_taxid, target_df in target_dfs.items():
                filtered_target_df = target_df[target_df["hid"].eq(reference_protein_hid)]
                matching_target_ids = filtered_target_df["protein_accession"].to_list()
                target_matches[target_taxid] = matching_target_ids

        homologs[reference_protein_id] = target_matches

    return homologs

def fetch_sequences(accession_ids, batch_size = 100):
    '''
    Fetches protein sequences for a list of accession ids in batches from Entrez

    Args:
        accession_ids (list|tuple): NCBI protein accession IDs
        batch_size (int):           number of proteins to request at a time; NCBI recommends a limit of 100

    Returns:
        sequence_data (dict):       dictionary of NCBI protein identifiers --> protein sequences
    '''

    sequence_data = {}

    for start in range(0, len(accession_ids), batch_size):
        end = min(len(accession_ids), start + batch_size)
        print(f"Fetching records {start + 1} to {end}...")

        done_chunk = False
        while not done_chunk:
            try:
                ids_subset = ','.join(accession_ids[start:end])
                handle = Entrez.efetch(db="protein", id=ids_subset, rettype="fasta", retmode="text")
                fasta_data = handle.read()
                handle.close()
                for record in SeqIO.parse(StringIO(fasta_data), "fasta"):
                    sequence_data[record.id] = record.seq
                done_chunk = True
                time.sleep(0.2)
            except Exception as e:
                print(f"Error fetching records {start + 1} to {end}: {e} | retrying...")
                done_chunk = False

    return sequence_data

def retrieve_homologs(homologene_data_path, reference_taxid = 9606, target_taxids = (3702,)):
    '''

    Args:
        homologene_data_path (str): path to homologene.data file available over NCBI FTP
        reference_taxid (int):      taxonomic identifier for the reference species (e.g. human = 9606)
        target_taxids (list|tuple): taxonomic identifiers for the target species set

    Returns:
        homologs (dict):            dictionary of hid --> hid_matches, which is a dict of taxid --> protein_ids
        sequence_data (dict):       dictionary of NCBI protein identifiers --> protein sequences
    '''

    homologene_data = read_homologene(homologene_data_path)
    homologs = get_homolog_ids(homologene_data, reference_taxid, target_taxids)

    total_identifiers = []
    for reference_protein_id, target_matches in homologs.items():
        total_identifiers.append(reference_protein_id)
        for target_protein_ids in target_matches.values():
            for target_protein_id in target_protein_ids:
                total_identifiers.append(target_protein_id)

    total_identifiers = list(set(total_identifiers))
    sequence_data = fetch_sequences(total_identifiers)

    return homologs, sequence_data

if __name__ == "__main__":
    homologene_data_path = input("Enter path to homologene.data:  ")
    reference_taxid = int(input("Enter reference TaxID:  "))
    target_taxids = []
    while True:
        target_taxid = input("Enter target TaxID (leave blank when done):  ")
        if target_taxid != "":
            target_taxid = int(target_taxid)
            target_taxids.append(target_taxid)
        else:
            break

    homologs, sequence_data = retrieve_homologs(homologene_data_path, reference_taxid, target_taxids)

    cache_folder = input("Enter folder to save cached dict into:  ")
    current_dir = cache_folder if cache_folder != "" else os.getcwd()
    cache_id_path = os.path.join(current_dir, "cached_homolog_data.pkl")
    with open(cache_id_path, "wb") as f:
        pickle.dump((homologs, sequence_data), f)