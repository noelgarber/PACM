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
    attributes = ["ensembl_gene_id", "ensembl_transcript_id", "ensembl_peptide_id",
                  "external_gene_name", "gene_biotype", "refseq_peptide", "refseq_peptide_predicted"]
    print("Requesting Ensembl and NCBI accessions...")
    response = dataset.search({"attributes": attributes})

    print("\tReceived! Parsing data...")
    lines = response.text.split("\n")
    rows = [row.split("\t") for row in lines if row]
    accessions_df = pd.DataFrame(rows, columns=attributes)

    # Get the Uniprot accession numbers separately (requesting with NCBI accessions results in an error)
    uniprot_attributes = ["ensembl_peptide_id", "uniprotswissprot", "uniprotsptrembl"]
    print("Requesting UniProt accessions...")
    uniprot_response = dataset.search({"attributes": uniprot_attributes})
    uniprot_lines = uniprot_response.text.split("\n")

    print("\tReceived! Parsing data...")
    uniprot_rows = [row.split("\t") for row in uniprot_lines if row]
    uniprot_dict = {row[0]: row[1] for row in uniprot_rows if row}
    trembl_dict = {row[0]: row[2] for row in uniprot_rows if row}

    # Merge data
    print("Merging data...")
    ensembl_peptide_ids = accessions_df["ensembl_peptide_id"].to_list()
    matching_uniprot_ids = [uniprot_dict.get(ensembl_id) for ensembl_id in ensembl_peptide_ids]
    matching_trembl_ids = [trembl_dict.get(ensembl_id) for ensembl_id in ensembl_peptide_ids]
    accessions_df["uniprot"] = matching_uniprot_ids
    accessions_df["trembl"] = matching_trembl_ids
    print("\tDone! Accessions were successfully retrieved.")

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

    print("\tParsing data...")
    lines = response.text.split("\n")
    rows = [line.split("\t") for line in lines if line]
    first_row = rows[0]
    if first_row[0][0:4] == "ENSP":
        sequence_data = {row[0]: row[1] for row in rows}
    else:
        sequence_data = {row[1]: row[0] for row in rows}
    print("Done! Sequences were successfully retrieved.")

    return sequence_data

def generate_dataset(accession_dataset_name = "hsapiens_gene_ensembl", sequence_dataset_name = "hsapiens_gene_ensembl"):
    '''
    Main function that generates the dataset

    Args:
        accession_dataset_name (str): name of the Biomart dataset containing your species' accession numbers
        sequence_dataset_name (str):  name of the Biomart dataset containing your species' protein sequences

    Returns:
        data_df (pd.DataFrame): dataframe with the accessions and protein sequence data
    '''

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