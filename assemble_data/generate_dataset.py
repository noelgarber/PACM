# This is the main file for generating large datasets to score and evaluate

import numpy as np
import pandas as pd
import os
import pickle
from biomart import BiomartServer
from io import StringIO
from Bio import SeqIO
from assemble_data.retrieve_homologs import get_homologs

def fetch_accessions(dataset_name = "hsapiens_gene_ensembl", biomart_url = "http://useast.ensembl.org/biomart"):
    '''
    Accession retrieval function

    Args:
        dataset_name (str):   Ensembl BioMart species dataset name

    Returns:
        accessions_df (pd.DataFrame): dataframe of accession numbers retrieved from BioMart
    '''

    # Connect to BioMart server
    print("Connecting to Ensembl Biomart accessions database...")
    server = BiomartServer(biomart_url)
    dataset = server.datasets[dataset_name]

    # Get the Ensembl and NCBI accession numbers
    attributes = ["ensembl_gene_id", "ensembl_transcript_id", "ensembl_peptide_id",
                  "external_gene_name", "gene_biotype", "refseq_peptide", "refseq_peptide_predicted"]
    print("Requesting Ensembl and NCBI accessions...")
    response = dataset.search({"attributes": attributes})

    print("\tStreaming data...")
    response_tsv = StringIO(response.text)
    accessions_df = pd.read_csv(response_tsv, header=None, sep="\t")
    accessions_df.columns = attributes

    # Get the Uniprot accession numbers separately (requesting with NCBI accessions results in an error)
    uniprot_attributes = ["ensembl_peptide_id", "uniprotswissprot", "uniprotsptrembl"]
    print("Requesting UniProt accessions...")
    uniprot_response = dataset.search({"attributes": uniprot_attributes})

    print("\tStreaming data...")
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

def generate_dataset(protein_fasta_path = None, retrieve_matching_homologs = True, homologene_path = None,
                     reference_taxid = 9606, target_taxids = (3702,), accession_dataset_name = "hsapiens_gene_ensembl"):
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

    # Fetch or load accession numbers for all host proteins
    accessions_path = os.path.join(os.getcwd(), "accessions_df.pkl")
    if os.path.isfile(accessions_path):
        discard = input("Discard existing accessions_df and request again? (y/N)  ")
        discard = discard == "N" or discard == "n"
        if discard:
            data_df = fetch_accessions(accession_dataset_name)
            with open(accessions_path, "wb") as f:
                pickle.dump(data_df, f)
        else:
            with open(accessions_path, "rb") as f:
                data_df = pickle.load(f)
    else:
        data_df = fetch_accessions(accession_dataset_name)
        with open(accessions_path, "wb") as f:
            pickle.dump(data_df, f)

    # Parse Ensembl protein sequence FASTA into a lookup dictionary of ID to protein sequence
    if protein_fasta_path is None:
        protein_fasta_path = input("Enter the path to Ensembl protein sequences (FASTA):  ")
    records = [record for record in SeqIO.parse(protein_fasta_path, "fasta")]
    ensembl_base_keys = [record.id.split(".")[0] for record in records]
    ensembl_base_values = [str(record.seq) for record in records]
    ensembl_sequence_dict = {key:value for key, value in zip(ensembl_base_keys, ensembl_base_values)}

    # Apply sequences to corresponding IDs in the dataframe
    ensembl_ids = data_df["ensembl_peptide_id"].to_list()
    seqs = [ensembl_sequence_dict.get(ensembl_id) for ensembl_id in ensembl_ids]
    data_df["sequence"] = seqs

    # Assign homolog accessions and sequences
    if retrieve_matching_homologs:
        # Load homologs dict
        if not homologene_path:
            homologene_path = input("Enter the path to homologene.data (available from NCBI FTP):  ")
        homologs = get_homologs(homologene_path, reference_taxid, target_taxids)

        # Extract IDs to find sequences for
        refseq_peptide_ids = data_df["refseq_peptide"].to_list()
        refseq_predicted_ids = data_df["refseq_peptide_predicted"].to_list()
        entries_count = len(refseq_peptide_ids)

        for i, (refseq_peptide_id, refseq_predicted_id) in enumerate(zip(refseq_peptide_ids, refseq_predicted_ids)):
            print(f"Looking up homologs for row {i} of {entries_count}")
            refseq_peptide_matches = homologs.get(refseq_peptide_id)
            refseq_predicted_matches = homologs.get(refseq_predicted_id)

            if refseq_peptide_matches is not None:
                print(f"\tHomologs found for {refseq_peptide_id} in species: {list(refseq_peptide_matches.keys())}")
                for taxid, matching_target_tuples in refseq_peptide_matches.items():
                    for j, match_tuple in enumerate(matching_target_tuples):
                        col_name = f"{taxid}_homolog_{j}"
                        data_df.at[i, col_name] = match_tuple[0]
                        data_df.at[i, col_name + "_sequence"] = match_tuple[1]

            if refseq_predicted_matches is not None:
                print(f"\tHomologs found for {refseq_predicted_id} in species: {list(refseq_predicted_matches.keys())}")
                for taxid, matching_target_tuples in refseq_predicted_matches.items():
                    for j, match_tuple in matching_target_tuples:
                        col_name = f"{taxid}_homolog_{j}"
                        data_df.at[i, col_name] = match_tuple[0]
                        data_df.at[i, col_name + "_sequence"] = match_tuple[1]

    return data_df

if __name__ == "__main__":
    default_fasta_path = os.path.join(os.getcwd(), "default_source_data/Homo_sapiens.GRCh38.pep.all.fa")
    fasta_path = default_fasta_path if os.path.isfile(default_fasta_path) else None

    retrieve_matching_homologs = True
    homologene_default_path = os.path.join(os.getcwd(), "default_source_data/homologene.data")
    homologene_path = homologene_default_path if os.path.isfile(homologene_default_path) else None

    reference_taxid = 9606 # human
    target_taxids = (10090, 10116, 7955, 6239, 7227, 4932, 4896, 3702)
    data_df = generate_dataset(fasta_path, retrieve_matching_homologs, homologene_path, reference_taxid, target_taxids)

    saved = False
    while not saved:
        try:
            save_folder = input("Success! Enter the folder for saving data (or leave blank to use current directory:  ")
            if save_folder == "": 
                save_folder = os.getcwd()
                save_folder = save_folder.rsplit("/",1)[0]
            save_path = os.path.join(save_folder, "dataset.csv")
            data_df.to_csv(save_path)
            saved = True
        except Exception as e:
            print(f"Error while saving: {e}")
            print("Retrying...")