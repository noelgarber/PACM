# This is the main file for generating large datasets to score and evaluate

import numpy as np
import pandas as pd
import os
from biomart import BiomartServer
from io import StringIO
from assemble_data.retrieve_homologs import retrieve_homologs

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
    response_tsv = StringIO(response.text)
    accessions_df = pd.read_csv(response_tsv, header=None, sep="\t")
    accessions_df.columns = attributes

    # Get the Uniprot accession numbers separately (requesting with NCBI accessions results in an error)
    uniprot_attributes = ["ensembl_peptide_id", "uniprotswissprot", "uniprotsptrembl"]
    print("Requesting UniProt accessions...")
    uniprot_response = dataset.search({"attributes": uniprot_attributes})

    print("\tReceived! Parsing data...")
    uniprot_tsv = StringIO(uniprot_response.text)
    uniprot_df = pd.read_csv(uniprot_tsv, header=None, sep="\t")
    uniprot_df.columns = uniprot_attributes

    ensembl_ids = uniprot_df["ensembl_peptide_id"].to_list()
    uniprot_ids = uniprot_df["uniprotswissprot"].to_list()
    trembl_ids = uniprot_df["uniprotsptrembl"].to_list()

    uniprot_dict = {ensembl_id: uniprot_id for ensembl_id, uniprot_id in zip(ensembl_ids, uniprot_ids)}
    trembl_dict = {ensembl_id: trembl_id for ensembl_id, trembl_id in zip(ensembl_ids, trembl_ids)}

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
    response_tsv = StringIO(response.text)
    print("\tMaking dataframe...")
    response_df = pd.read_csv(response_tsv, header=None, sep="\t")
    response_df.columns = ["sequence", "ensembl_peptide_id"]
    seqs = response_df["sequence"].to_list()
    ids = response_df["ensembl_peptide_id"].to_list()
    print("\tConverting to dictionary...")
    sequence_data = {id: seq for id, seq in zip(ids, seqs)}
    random_key = list(sequence_data.keys())[0]
    print("sequence_data sample entry:", random_key, "-->", sequence_data.get(random_key))

    print("Done! Sequences were successfully retrieved.")

    return sequence_data

def generate_dataset(accession_dataset_name = "hsapiens_gene_ensembl", sequence_dataset_name = "hsapiens_gene_ensembl",
                     retrieve_matching_homologs = True, homologene_path = None, reference_taxid = 9606,
                     target_taxids = (10090, 10116, 7955, 6239, 7227, 4932, 4896, 3702)):
    '''
    Main function that generates the dataset

    Args:
        accession_dataset_name (str):      name of the Biomart dataset containing your species' accession numbers
        sequence_dataset_name (str):       name of the Biomart dataset containing your species' protein sequences
        retrieve_matching_homologs (bool): whether to retrieve matching homolog sequences from NCBI
        homologene_data_path (str):        path to homologene.data file available over NCBI FTP
        reference_taxid (int):             taxonomic identifier for the reference species (e.g. human = 9606)
        target_taxids (list|tuple):        taxonomic identifiers for the target species set

    Returns:
        data_df (pd.DataFrame): dataframe with the accessions and protein sequence data
    '''

    # Get accession numbers for all host proteins
    data_df = fetch_accessions(accession_dataset_name)

    # Assign sequences to these proteins
    sequence_data = fetch_seqs(sequence_dataset_name)
    ensembl_ids = data_df["ensembl_peptide_id"].to_list()
    seqs = [sequence_data.get(ensembl_id) for ensembl_id in ensembl_ids]
    data_df["sequence"] = seqs

    # Assign homolog accessions and sequences
    if retrieve_matching_homologs:
        homologs, sequence_data = retrieve_homologs(homologene_path, reference_taxid, target_taxids)
        for i, reference_ensembl_id in enumerate(ensembl_ids):
            matches_by_taxid = homologs.get(reference_ensembl_id)
            if matches_by_taxid is not None:
                for taxid, matches in matches_by_taxid.items():
                    for j, match in enumerate(matches):
                        col_name = f"{taxid}_homolog_{j}"
                        data_df[i, col_name] = match

    return data_df

if __name__ == "__main__":
    data_df = generate_dataset()
    save_folder = input("Success! Enter the folder to save the data to:  ")
    save_path = os.path.join(save_folder, "dataset.csv")
    data_df.to_csv(save_path)