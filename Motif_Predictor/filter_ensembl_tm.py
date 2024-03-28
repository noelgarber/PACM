import numpy as np
import pandas as pd

def parse_ensembl_tm(path):
    '''
    Function to parse Ensembl BioMart entries in a CSV/TSV that has Ensembl protein IDs and starts/ends for TMDs

    Args:
        path (str): path to the CSV/TSV containing the downloaded BioMart data

    Returns:
        ensembl_tm_dict (dict): dictionary of Ensembl protein ID --> list of (start,end) tuples
    '''

    if path.rsplit(".",1)[1] == "csv":
        ensembl_tm_df = pd.read_csv(path)
    else:
        ensembl_tm_df = pd.read_csv(path, sep="\t")

    ensembl_id_col = "ensembl_peptide_id" if "ensembl_peptide_id" in ensembl_tm_df.columns else "Protein stable ID"
    start_col = "start" if "start" in ensembl_tm_df.columns else "Transmembrane helices start"
    end_col = "end" if "end" in ensembl_tm_df.columns else "Transmembrane helices end"

    ensembl_tm_df.dropna(subset=[start_col, end_col], inplace=True)
    ensembl_tm_df.sort_values(start_col, axis=0, ascending=True, inplace=True)
    zipped_cols = zip(ensembl_tm_df[ensembl_id_col].to_list(),
                      ensembl_tm_df[start_col].to_list(),
                      ensembl_tm_df[end_col].to_list())

    ensembl_tm_dict = {}
    for ensembl_id, start, end in zipped_cols:
        start, end = int(start), int(end)
        if ensembl_tm_dict.get(ensembl_id) is None:
            ensembl_tm_dict[ensembl_id] = [(start, end)]
        else:
            ensembl_tm_dict[ensembl_id].append((start, end))

    return ensembl_tm_dict

def apply_ensembl_tm(df, ensembl_id_col = "ensembl_peptide_id", seq_col = "sequence", start_tolerance = 0,
                     end_tolerance = 0, ensembl_tm_dict = None, ensembl_tm_path = None):
    '''
    Removes Ensembl transmembrane domains from scannable protein sequences

    Args:
        df (pd.DataFrame):      input dataframe containing sequences to sanitize
        ensembl_id_col (str):   column name in df containing ensembl protein ids
        seq_col (str):          column name in df containing protein sequences
        start_tolerance (int):  amount of N-terminal side of transmembrane domain to retain
        end_tolerance (int):    amount of C-terminal side of transmembrane domain to retain
        ensembl_tm_dict (dict): dictionary of ensembl protein id --> list of (start,end) for each transmembrane domain
        ensembl_tm_path (str):  if ensembl_tm_dict is not given, give a path to a CSV/TSV so it can be constructed

    Returns:
        df (pd.DataFrame):      modified input dataframe
    '''

    if ensembl_tm_dict is None and isinstance(ensembl_tm_path, str):
        ensembl_tm_dict = parse_ensembl_tm(ensembl_tm_path)
    elif ensembl_tm_dict is None:
        raise ValueError(f"ensembl_tm_path must be a valid path, but was set to {ensembl_tm_path}")

    for idx in df.index:
        ensembl_peptide_id = df.at[idx, ensembl_id_col]
        tm_ranges = ensembl_tm_dict.get(ensembl_peptide_id)
        if tm_ranges is not None:
            protein_seq = df.at[idx, seq_col]
            for start, end in reversed(tm_ranges):
                start = start + start_tolerance - 1 # convert to 0-indexing
                end = end + end_tolerance - 1
                protein_seq = protein_seq[:start] + "X" + protein_seq[end+1:]
                df.at[idx, seq_col] = protein_seq

    return df