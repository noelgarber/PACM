# This script retrieves homolog sequences from NCBI HomoloGene for given taxonomic identifiers (TaxIDs)

import numpy as np
import pandas as pd
import os
import pickle
import time
from io import StringIO
from Bio import Entrez, SeqIO
Entrez.email = "ngarber93@gmail.com"

def read_homologene(homologene_data_path, get_sequences = True, seq_chunk_size = 5000):
    '''
    Loads homologene.data (from the NCBI FTP site) and retrieve matching sequences

    Args:
        homologene_data_path (str): local path to homologene.data
        get_sequences (bool):       whether to request sequences from Entrez
        seq_chunk_size (bool):      maximum number of sequences to request at once;
                                    setting this above 10000 can cause data to be truncated

    Returns:
        homologene_df (pd.DataFrame): homologene data in a dataframe
    '''

    # Parse the HomoloGene entries
    print("Loading homologene.data...")
    homologene_df = pd.read_csv(homologene_data_path, header=None, sep="\t")
    columns = ["hid", "taxid", "gene_id", "gene_name", "protein_gi", "protein_accession"]
    homologene_df.columns = columns

    # Fetch matching protein sequences
    if get_sequences:
        accessions = homologene_df["protein_accession"].to_list()
        chunks = [accessions[i:i+seq_chunk_size] for i in range(0, len(accessions), seq_chunk_size)]

        records = []
        for i, chunk in enumerate(chunks):
            print(f"Requesting sequences from Entrez for chunk #{i+1}...")
            handle = Entrez.efetch(db="protein", id=chunk, rettype="fasta", retmode="text")
            fasta_data = handle.read()
            handle.close()

            print("\tReceived; parsing records...")
            chunk_records = [record for record in SeqIO.parse(StringIO(fasta_data), "fasta")]
            records.extend(chunk_records)

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

    # Check if this task has already been done
    pickled_homologs_path = os.path.join(os.getcwd(), "homologs.pkl")
    if os.path.isfile(pickled_homologs_path):
        use_pickled_homologs = input("Pickled homologs dict was found; use it? (Y/n)  ")
        use_pickled_homologs = use_pickled_homologs == "Y" or use_pickled_homologs == "y"
        if use_pickled_homologs:
            with open(pickled_homologs_path, "rb") as f:
                homologs = pickle.load(f)
            return homologs

    # Read homologene and get sequences
    homologene_df = read_homologene(homologene_data_path)
    reference_df = homologene_df[homologene_df["taxid"].eq(reference_taxid)]

    # Retrieve target species dataframes
    target_dfs = {}
    for target_taxid in target_taxids:
        target_df = homologene_df[homologene_df["taxid"].eq(target_taxid)]
        target_dfs[target_taxid] = target_df

    # Construct a non-redundant homolog dictionary
    reference_protein_ids = reference_df["protein_accession"].to_list()
    print("Generating homologs dictionary...")
    homologs = {}
    for i, reference_protein_id in enumerate(reference_protein_ids):
        # print(f"Assigning host protein #{i} of {reference_count}")
        reference_entries_df = reference_df[reference_df["protein_accession"].eq(reference_protein_id)]
        reference_protein_hids = reference_entries_df["hid"].to_list()

        matches_by_taxid = {}

        for target_taxid, target_df in target_dfs.items():
            filtered_target_df = target_df[target_df["hid"].isin(reference_protein_hids)]
            matching_target_ids = filtered_target_df["protein_accession"].to_list()
            matching_target_seqs = filtered_target_df["sequence"].to_list()

            matching_target_tuples = []
            for matching_target_id, sequence in zip(matching_target_ids, matching_target_seqs):
                matching_target_tuples.append((matching_target_id, sequence))
                if sequence is None:
                    print(f"Caution: matching target ID {matching_target_id} has no associated sequence!")
            matches_by_taxid[target_taxid] = matching_target_tuples

        reference_base_id = reference_protein_id.split(".")[0]
        homologs[reference_base_id] = matches_by_taxid

    with open(pickled_homologs_path, "wb") as f:
        pickle.dump(homologs, f)

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