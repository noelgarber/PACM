# For SLiMs that occur in membrane proteins, this script tests whether the SLiMs are on the cytosol-facing toplogical domains

import numpy as np
import pandas as pd
import multiprocessing
from tqdm import trange
from functools import partial
from assemble_data.parse_uniprot_topology import get_topological_domains
from Motif_Predictor.predictor_config import predictor_params

def predict_chunk(chunk_df, motif_col, topology_trim_begin, topology_trim_end, topological_domains, sequences):
    '''
    Lower-level function that predicts and assigns topologies for motifs in a chunk of a dataframe

    Args:
        chunk_df (pd.DataFrame):    chunk of main dataframe
        motif_col (str):            name of column holding motif sequences
        topology_trim_begin (int):  number of residues at the beginning of the motif to ignore when assigning topology
        topology_trim_end (int):    number of residues at the end of the motif to ignore when assigning topology
        topological_domains (dict): dictionary of accession --> topological domains list
        sequences (dict):           dictionary of accession --> sequence

    Returns:
        chunk_topology_df (pd.DataFrame):  dataframe with topology columns corresponding to chunk_df
    '''

    chunk_topology_df = pd.DataFrame()

    for i in chunk_df.index:
        motif_seq = chunk_df.at[i, motif_col]
        uniprot_swissprot_id = chunk_df.at[i, "uniprot"]
        if uniprot_swissprot_id:
            # Retrieve topological domains
            current_topological_domains = topological_domains.get(uniprot_swissprot_id)
            if current_topological_domains is None:
                print(f"\tCould not find topological domains for {uniprot_swissprot_id}")
                continue
            else:
                print(f"\tFound topological domains for {uniprot_swissprot_id}")

            # Retrieve sequence matching topological domain begin/end indices
            sequence = sequences.get(uniprot_swissprot_id)
            if sequence is None:
                print(f"\tCould not find matching sequence for {uniprot_swissprot_id}; skipping...")
                continue

            # Find the corresponding index of the motif in the protein sequence
            motif_len = len(motif_seq)
            trimmed_motif_seq = motif_seq[topology_trim_begin: motif_len - topology_trim_end]
            trimmed_begin = sequence.find(trimmed_motif_seq)
            if trimmed_begin == -1:
                print(f"\t\tFailed to find trimmed motif ({trimmed_motif_seq}) in sequence}")
                continue
            trimmed_end = trimmed_begin + len(trimmed_motif_seq)

            # Determine which topological domain the motif exists within
            cell_empty = True
            for domain_tuple in current_topological_domains:
                topology_domain_type_col = f"{motif_col}_topology_domain_type"
                topology_domain_description_col = f"{motif_col}_topology_domain_description"

                topological_domain_type, description, begin_position, end_position = domain_tuple
                begin_idx = begin_position - 1
                end_idx = end_position - 1

                motif_within = trimmed_begin >= begin_idx and trimmed_end <= end_idx
                head_within = trimmed_begin >= begin_idx and trimmed_begin <= end_idx and trimmed_end > end_idx
                tail_within = trimmed_begin < begin_idx and trimmed_end > begin_idx and trimmed_end <= end_idx
                middle_within = trimmed_begin < begin_idx and trimmed_end > end_idx

                if motif_within:
                    # Motif is fully within current topological domain
                    chunk_topology_df.at[i, topology_domain_type_col] = topological_domain_type
                    chunk_topology_df.at[i, topology_domain_description_col] = description
                    break

                elif head_within or tail_within or middle_within:
                    # First part of motif is within current topological domain
                    if cell_empty:
                        chunk_topology_df.at[i, topology_domain_type_col] = topological_domain_type
                        chunk_topology_df.at[i, topology_domain_description_col] = description
                        cell_empty = False

                    else:
                        # Append to existing info when motif spans more than one topological domain
                        domain_type_existing = chunk_topology_df.at[i, topology_domain_type_col]
                        description_existing = chunk_topology_df.at[i, topology_domain_description_col]
                        concatenated_types = f"{domain_type_existing};{topological_domain_type}"
                        concatenated_descriptions = f"{description_existing};{description}"
                        chunk_topology_df.at[i, topology_domain_type_col] = concatenated_types
                        chunk_topology_df.at[i, topology_domain_description_col] = concatenated_descriptions
                        cell_empty = False

    return chunk_topology_df

def predict_topology(data_df, motif_cols, predictor_params = predictor_params):
    '''
    Main function to predict motif topologies

    Args:
        data_df (pd.DataFrame):   main dataframe containing protein sequences and predicted motifs
        motif_cols (list):        list of column names referring to motif sequences for topological analysis
        predictor_params (dict):  dictionary of user-defined parameters for the motif prediction workflow

    Returns:
        data_df (pd.DataFrame):   data_df with topology columns added
    '''

    uniprot_path = predictor_params["uniprot_path"]
    topology_trim_begin = predictor_params["topology_trim_begin"]
    topology_trim_end = predictor_params["topology_trim_end"]
    chunk_size = predictor_params["topology_chunk_size"]

    topological_domains, sequences = get_topological_domains(path = uniprot_path)

    chunks_dfs = [data_df[i:i + chunk_size] for i in range(0, len(data_df), chunk_size)]

    for motif_col in motif_cols:
        partial_predictor = partial(predict_chunk, motif_col = motif_col, topology_trim_begin = topology_trim_begin,
                                    topology_trim_end = topology_trim_end, topological_domains = topological_domains,
                                    sequences = sequences)
        chunk_topology_dfs = []

        pool = multiprocessing.Pool()
        with trange(len(chunks_dfs), desc=f"Assigning topologies for {motif_col}") as pbar:
            for chunk_topology_df in pool.map(partial_predictor, chunks_dfs):
                chunk_topology_dfs.append(chunk_topology_df)
                pbar.update()

        pool.close()
        pool.join()

        motif_topology_df = pd.concat(chunk_topology_dfs, axis=0)
        data_df = pd.concat([data_df, motif_topology_df], axis=1)

    return data_df