# This is the main file for generating large datasets to score and evaluate

import numpy as np
import pandas as pd
import os
from biomart import BiomartServer
from io import StringIO
from Bio import SeqIO

def fetch_accessions(dataset_name = "hsapiens_gene_ensembl"):
    '''
    Accession retrieval function

    Args:
        dataset_name (str):   Ensembl BioMart species dataset name

    Returns:
        accessions_df (pd.DataFrame): dataframe of accession numbers retrieved from BioMart
    '''

    # Connect to BioMart server
    print("Connecting to Biomart accessions database...")
    server = BiomartServer("http://www.ensembl.org/biomart")
    dataset = server.datasets[dataset_name]

    # Get the Ensembl and NCBI accession numbers
    main_attributes = ["ensembl_gene_id", "ensembl_transcript_id", "ensembl_peptide_id",
                       "external_gene_name", "gene_biotype", "refseq_peptide", "refseq_peptide_predicted"]
    col_names = main_attributes.copy()
    print("Requesting Ensembl and NCBI accessions...")
    main_response = dataset.search({"attributes": main_attributes})

    print("\tReceived! Parsing data...")
    main_response_lines = [line.decode("utf-8") for line in main_response.iter_lines()]
    main_data_rows = [row.split("\t") for row in main_response_lines if row]
    accessions_df = pd.DataFrame(main_data_rows, columns=col_names)

    # Get the Uniprot accession numbers separately (requesting with NCBI accessions results in an error)
    uniprot_attributes = ["ensembl_peptide_id", "uniprotswissprot", "uniprotsptrembl"]
    print("Requesting UniProt accessions...")
    uniprot_response = dataset.search({"attributes": uniprot_attributes})
    uniprot_response_lines = [line.decode("utf-8") for line in uniprot_response.iter_lines()]
    uniprot_dict = {}
    trembl_dict = {}
    print("\tReceived! Parsing data...")
    for row in uniprot_response_lines:
        if row:
            ensembl_id, uniprot_id, trembl_id = row.split("\t")
            uniprot_dict[ensembl_id] = uniprot_id
            trembl_dict[ensembl_id] = trembl_id

    # Merge data
    print("Merging data...")
    ensembl_peptide_ids = accessions_df["ensembl_peptide_id"].to_list()
    matching_uniprot_ids = [uniprot_dict.get(ensembl_id) for ensembl_id in ensembl_peptide_ids]
    matching_trembl_ids = [trembl_dict.get(ensembl_id) for ensembl_id in ensembl_peptide_ids]
    accessions_df["uniprot"] = matching_uniprot_ids
    accessions_df["trembl"] = matching_trembl_ids
    print("Success! Accessions received.")

    return accessions_df

def fetch_seqs(dataset_name = "hsapiens_gene_ensembl"):
    '''
    Sequence retrieval function

    Args:
        dataset_name (str): name of the BioMart dataset containing the sequences to be requested

    Returns:
        sequence_data (dict): dictionary of Ensembl protein ID --> protein sequence
    '''

    # Connect to BioMart server
    print("Connecting to Biomart sequence database...")
    server = BiomartServer("http://www.ensembl.org/biomart")
    dataset = server.datasets[dataset_name]

    # Build the query
    attributes = ["ensembl_peptide_id", "peptide"]
    print("Requesting sequences...")
    response = dataset.search({"attributes": attributes})
    fasta_data = response.text

    # Parse the FASTA data
    sequence_data = {}
    print("Parsing sequence data...")
    for record in SeqIO.parse(StringIO(fasta_data), "fasta"):
        sequence_data[record.id] = str(record.seq)

    return sequence_data

def generate_dataset(accession_dataset_name = "hsapiens_gene_ensembl", sequence_dataset_name = "hsapiens_gene_ensembl"):

    data_df = fetch_accessions(accession_dataset_name)
    sequence_data = fetch_seqs(sequence_dataset_name)

    ensembl_ids = data_df["ensembl_peptide_id"].to_list()
    seqs = [sequence_data.get(ensembl_id) for ensembl_id in ensembl_ids]
    data_df["sequence"] = seqs

    return data_df

if __name__ == "__main__":
    data_df = generate_dataset()
    save_folder = input("Success! Enter the folder to save the data to:  ")
    save_path = os.path.join(save_folder, "dataset.csv")
    data_df.to_csv(save_path)