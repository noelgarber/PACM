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
        homologene_data_df (pd.DataFrame): dataframe containing data from the homologene.data file available over NCBI FTP
        reference_taxid (int):             taxonomic identifier for the reference species (e.g. human = 9606)
        target_taxids (list|tuple):        taxonomic identifiers for the target species set

    Returns:
        homologs (pd.DataFrame):         dataframe with retrieved homologs
    '''

    reference_df = homologene_data[homologene_data["taxid"].eq(reference_taxid)]
    homology_group_ids = list(set(reference_df["hid"].to_list()))

    # Retrieve target species dataframes
    target_dfs = {}
    for target_taxid in target_taxids:
        target_df = homologene_data[homologene_data["taxid"].eq(target_taxid)]
        target_dfs[target_taxid] = target_df

    # Set up a recursive dictionary with the homolog data
    homologs = {}
    for hid in homology_group_ids:
        hid_matches = {}

        reference_match_df = reference_df[reference_df["hid"].eq(hid)]
        reference_gene_names = reference_match_df["gene_name"]
        reference_protein_ids = reference_match_df["protein_accession"]

        for target_taxid, target_df in target_dfs.items():
            target_match_df = target_df[target_df["hid"].eq(hid)]
            target_gene_names = target_match_df["gene_name"]
            target_protein_ids = target_match_df["protein_accession"]

            for reference_gene_name, reference_protein_id in zip(reference_gene_names, reference_protein_ids):
                matches = {}
                for target_gene_name, target_protein_id in zip(target_gene_names, target_protein_ids):
                    gene_names = {"reference_gene_name": reference_gene_name, "target_gene_name": target_gene_name}
                    matches[target_protein_id] = gene_names

                hid_matches[reference_protein_id] = matches

        homologs[hid] = hid_matches

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
        try:
            ids_subset = ','.join(accession_ids[start:end])
            handle = Entrez.efetch(db="protein", id=ids_subset, rettype="fasta", retmode="text")
            fasta_data = handle.read()
            handle.close()
            for record in SeqIO.parse(StringIO(fasta_data), "fasta"):
                sequence_data[record.id] = record.seq
            time.sleep(0.2)
        except Exception as e:
            print(f"Error fetching records {start + 1} to {end}: {e}")
            break

    return sequence_data

def retrieve_homologs(homologene_data_path, reference_taxid = 9606, target_taxids = (3702,)):
    '''

    Args:
        homologene_data_path (str): path to homologene.data file available over NCBI FTP
        reference_taxid (int):      taxonomic identifier for the reference species (e.g. human = 9606)
        target_taxids (list|tuple): taxonomic identifiers for the target species set

    Returns:
        homologs (pd.DataFrame):    dataframe with retrieved homologs
        sequence_data (dict):       dictionary of NCBI protein identifiers --> protein sequences
    '''

    homologene_data = read_homologene(homologene_data_path)
    homologs = get_homolog_ids(homologene_data, reference_taxid, target_taxids)

    total_identifiers = []
    for hid_matches in homologs.values():
        for reference_protein, value in hid_matches.items():
            total_identifiers.append(reference_protein)
            for target_protein in value.keys():
                total_identifiers.append(target_protein)

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