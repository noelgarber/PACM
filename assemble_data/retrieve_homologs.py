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
    Loads homologene.data (from the NCBI FTP site) and retrieve matching sequences

    Args:
        homologene_data_path (str): local path to homologene.data

    Returns:
        homologene_df (pd.DataFrame): homologene data in a dataframe
    '''

    # Parse the HomoloGene entries
    print("Loading homologene.data...")
    homologene_df = pd.read_csv(homologene_data_path, header=None, sep="\t")
    columns = ["hid", "taxid", "gene_id", "gene_name", "protein_gi", "protein_accession"]
    homologene_df.columns = columns

    # Fetch matching protein sequences
    print("Requesting sequences from Entrez...")
    accessions = homologene_df["protein_accession"].to_list()
    handle = Entrez.efetch(db="protein", id=accessions, rettype="fasta", retmode="text")
    fasta_data = handle.read()
    handle.close()

    print("Parsing records...")
    records = [record for record in SeqIO.read(StringIO(fasta_data), "fasta")]
    print("\tSample record: ")
    print(f"\t\tID: {record[0].id}")
    print(f"\t\tSeq: {record[0].seq}")

    print("Applying seqs to homologene dataframe...")
    lookup_dict = {record.id: str(record.seq) for record in records}
    seqs = [lookup_dict.get(accession) for accession in accessions]
    homologene_df["sequence"] = seqs

    return homologene_df

def get_homologs(homologene_data_path, reference_taxid = 9606, target_taxids = (3702,)):
    '''
    Retrieves homolog IDs matching target TaxIDs and the reference TaxID and sorts into a dictionary of dictionaries

    Args:
        homologene_data_path (str):  path to homologene.data (available from NCBI FTP)
        reference_taxid (int):       taxonomic identifier for the reference species (e.g. human = 9606)
        target_taxids (list|tuple):  taxonomic identifiers for the target species set

    Returns:
        homologs (dict):                   dictionary of reference_id --> homolog_taxid --> matching_tuples of (id, seq)
    '''

    homologene_df = read_homologene(homologene_data_path)
    reference_df = homologene_df[homologene_df["taxid"].eq(reference_taxid)]

    # Retrieve target species dataframes
    target_dfs = {}
    for target_taxid in target_taxids:
        target_df = homologene_df[homologene_df["taxid"].eq(target_taxid)]
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
                matching_target_seqs = filtered_target_df["sequence"].to_list()
                matching_target_tuples = []
                for id, seq in zip(matching_target_ids, matching_target_seqs):
                    matching_target_tuples.append((id, seq))
                target_matches[target_taxid] = matching_target_tuples

        homologs[reference_protein_id] = target_matches

    return homologs

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

    homologs = get_homologs(homologene_data_path, reference_taxid, target_taxids)

    saved = False
    while not saved:
        try:
            cache_folder = input("Enter folder to save cached dict into:  ")
            current_dir = cache_folder if cache_folder != "" else os.getcwd()
            cache_id_path = os.path.join(current_dir, "cached_homolog_data.pkl")
            with open(cache_id_path, "wb") as f:
                pickle.dump((homologs, sequence_data), f)
            saved = True
        except Exception as e:
            print(f"Error while saving ({e}); retrying...")