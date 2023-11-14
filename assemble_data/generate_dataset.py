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

def generate_base_dataset(protein_fasta_path = None, accession_dataset_name = "hsapiens_gene_ensembl"):
    '''
    Main function that generates the dataset

    Args:
        protein_fasta_path (str):          path to fasta file with main protein sequences
        accession_dataset_name (str):      name of the Biomart dataset containing your species' accession numbers

    Returns:
        data_df (pd.DataFrame):            dataframe with the accessions and protein sequence data
    '''

    # Fetch or load accession numbers for all host proteins
    accessions_path = os.path.join(os.getcwd(), "accessions_df.pkl")
    if os.path.isfile(accessions_path):
        use_pickled = input("Pickled accessions_df was found. Would you like to use it? (Y/n)  ")
        use_pickled = use_pickled == "Y" or use_pickled == "y"
        if use_pickled:
            with open(accessions_path, "rb") as f:
                data_df = pickle.load(f)
        else:
            data_df = fetch_accessions(accession_dataset_name)
            with open(accessions_path, "wb") as f:
                pickle.dump(data_df, f)
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

    return data_df

def retrieve_matches(input_df, reference_taxid, target_taxids, homologene_path = None, homologs = None, verbose = True):
    '''
    Retrieves matching homologs and assigns to dataframe

    Args:
        input_df (pd.DataFrame):     main dataframe to add homologs into
        reference_taxid (int):       taxonomic identifier for the reference species (e.g. human = 9606)
        target_taxids (list|tuple):  taxonomic identifiers for the target species set
        homologene_path (str):       path to homologene.data file available over NCBI FTP
        homologs (dict):             optional, but may be given if pre-obtained for performance improvement
        verbose (bool):              whether to print progress information

    Returns:
        data_df (pd.DataFrame): dataframe with the accessions and protein sequence data
    '''

    data_df = input_df.copy()

    # Load homologs dict if required
    if homologs is None:
        if not homologene_path:
            homologene_path = input("Enter the path to homologene.data (available from NCBI FTP):  ")
        homologs = get_homologs(homologene_path, reference_taxid, target_taxids)

    # Extract IDs to find sequences for
    refseq_peptide_ids = data_df["refseq_peptide"].to_list()
    refseq_predicted_ids = data_df["refseq_peptide_predicted"].to_list()
    entries_count = len(refseq_peptide_ids)

    blank_list = list(np.full(shape=entries_count, fill_value="", dtype="U"))
    homolog_id_cols = {}
    homolog_seq_cols = {}

    for i, (refseq_peptide_id, refseq_predicted_id) in enumerate(zip(refseq_peptide_ids, refseq_predicted_ids)):
        if verbose:
            print(f"Looking up homologs for row {i} of {entries_count}")
        refseq_peptide_matches = homologs.get(refseq_peptide_id)
        refseq_predicted_matches = homologs.get(refseq_predicted_id)

        if refseq_peptide_matches is not None:
            if verbose:
                print(f"\tHomologs found for {refseq_peptide_id} in species: {list(refseq_peptide_matches.keys())}")
            for taxid, matching_target_tuples in refseq_peptide_matches.items():
                tuple_indices = np.arange(len(matching_target_tuples))
                for j, match_tuple in zip(tuple_indices, matching_target_tuples):
                    col_name = f"{taxid}_homolog_{j}"

                    # Define list if it doesn't already exist
                    if homolog_id_cols.get(col_name) is None:
                        homolog_id_cols[col_name] = blank_list.copy()
                        homolog_seq_cols[col_name + "_sequence"] = blank_list.copy()

                    # Assign values
                    homolog_id_cols[col_name][i] = match_tuple[0]
                    homolog_seq_cols[col_name + "_sequence"][i] = match_tuple[1]

        if refseq_predicted_matches is not None:
            if verbose:
                print(f"\tHomologs found for {refseq_predicted_id} in species: {list(refseq_predicted_matches.keys())}")
            for taxid, matching_target_tuples in refseq_predicted_matches.items():
                base_index = len(refseq_peptide_matches[taxid]) if refseq_peptide_matches is not None else 0
                tuple_indices = np.arange(base_index, len(matching_target_tuples) + base_index)
                for j, match_tuple in zip(tuple_indices, matching_target_tuples):
                    col_name = f"{taxid}_homolog_{j}"

                    # Define list if it doesn't already exist
                    if homolog_id_cols.get(col_name) is None:
                        homolog_id_cols[col_name] = blank_list.copy()
                        homolog_seq_cols[col_name + "_sequence"] = blank_list.copy()

                    # Assign values
                    homolog_id_cols[col_name][i] = match_tuple[0]
                    homolog_seq_cols[col_name + "_sequence"][i] = match_tuple[1]

    # Assign homolog cols to dataframe
    if verbose:
        print("Assigning homologs to dataframe...")
    zipped_dicts = zip(homolog_id_cols.items(), homolog_seq_cols.items())
    for (homolog_id_col, col_ids), (homolog_seq_col, col_seqs) in zipped_dicts:
        data_df[homolog_id_col] = col_ids
        data_df[homolog_seq_col] = col_seqs

    return data_df

def generate_dataset(protein_fasta_path = None, retrieve_matching_homologs = True, homologene_path = None,
                     reference_taxid = 9606, target_taxids = (3702,), separate_target_taxids = True,
                     accession_dataset_name = "hsapiens_gene_ensembl", save_folder = None, verbose = True):
    '''
    Main function that generates the dataset

    Args:
        protein_fasta_path (str):          path to fasta file with main protein sequences
        accession_dataset_name (str):      name of the Biomart dataset containing your species' accession numbers
        sequence_dataset_name (str):       name of the Biomart dataset containing your species' protein sequences
        retrieve_matching_homologs (bool): whether to retrieve matching homolog sequences from NCBI
        homologene_path (str):             path to homologene.data file available over NCBI FTP
        reference_taxid (int):             taxonomic identifier for the reference species (e.g. human = 9606)
        target_taxids (list|tuple):        taxonomic identifiers for the target species set
        separate_target_taxids (bool):     whether to generate separate dataframes for each taxid or leave together
        save_folder (str):                 folder to save dataframes into, as CSVs
        verbose (bool):                    whether to print progress information
    '''

    save_folder = os.getcwd().rsplit("/",1)[0]

    # Generate main dataframe with host accessions and sequences
    data_df = generate_base_dataset(protein_fasta_path, accession_dataset_name)

    # Assign homolog accessions and sequences
    if retrieve_matching_homologs:
        homologs = get_homologs(homologene_path, reference_taxid, target_taxids)
        if separate_target_taxids:
            for target_taxid in target_taxids:
                subset_data_df = retrieve_matches(data_df, reference_taxid, (target_taxid,),
                                                  homologs = homologs, verbose = verbose)
                save_path = os.path.join(save_folder, f"proteome_dataset_{target_taxid}_homologs.csv")
                subset_data_df.to_csv(save_path)
        else:
            data_df = retrieve_matches(data_df, reference_taxid, target_taxids, homologene_path, homologs, verbose)
            save_path = os.path.join(save_folder, f"proteome_dataset_all_homologs.csv")
            data_df.to_csv(save_path)
    else:
        save_path = os.path.join(save_folder, f"proteome_dataset.csv")
        data_df.to_csv(save_path)

if __name__ == "__main__":
    default_fasta_path = os.path.join(os.getcwd(), "default_source_data/Homo_sapiens.GRCh38.pep.all.fa")
    fasta_path = default_fasta_path if os.path.isfile(default_fasta_path) else None

    retrieve_matching_homologs = True
    separate_target_str = input("Separate homologs by target taxid? (Y/n):  ")
    separate_target_taxids = separate_target_str == "Y" or separate_target_str == "y"
    homologene_default_path = os.path.join(os.getcwd(), "default_source_data/homologene.data")
    homologene_path = homologene_default_path if os.path.isfile(homologene_default_path) else None

    reference_taxid = 9606 # human
    target_taxids = (10090, 10116, 7955, 6239, 7227, 4932, 4896, 3702)

    accession_dataset_name = "hsapiens_gene_ensembl"
    generate_dataset(fasta_path, retrieve_matching_homologs, homologene_path, reference_taxid, target_taxids,
                     separate_target_taxids, accession_dataset_name, verbose=True)