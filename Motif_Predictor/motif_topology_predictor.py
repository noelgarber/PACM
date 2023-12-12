# For SLiMs that occur in membrane proteins, this script tests whether the SLiMs are on the cytosol-facing toplogical domains

import numpy as np
import pandas as pd
import multiprocessing
from tqdm import trange
from functools import partial
from assemble_data.parse_uniprot_topology import get_topological_domains
from Motif_Predictor.predictor_config import predictor_params

def predict_chunk(motifs_ids_tuple, topology_trim_begin, topology_trim_end, topological_domains, sequences,
                  verbose = False):
    '''
    Lower-level function that predicts and assigns topologies for motifs in a chunk of a dataframe

    Args:
        motifs_ids_tuple (tuple):   tuple of (chunk_motif_seqs, chunk_uniprot_ids)
        topology_trim_begin (int):  number of residues at the beginning of the motif to ignore when assigning topology
        topology_trim_end (int):    number of residues at the end of the motif to ignore when assigning topology
        topological_domains (dict): dictionary of accession --> topological domains list
        sequences (dict):           dictionary of accession --> sequence
        verbose (bool):             whether to print progress messages

    Returns:
        domain_types_vals (list):        list of topological domain types for each element in chunk_df
        domain_descriptions_vals (list): list of topological domain descriptions for each element in chunk_df
    '''

    motif_seqs, uniprot_ids = motifs_ids_tuple

    domain_types_vals = []
    domain_descriptions_vals = []

    for i, (motif_seq, uniprot_swissprot_id) in enumerate(zip(motif_seqs, uniprot_ids)):
        domain_types = []
        domain_descriptions = []

        if uniprot_swissprot_id:
            # Retrieve topological domains
            current_topological_domains = topological_domains.get(uniprot_swissprot_id)

            if current_topological_domains is not None:
                # Retrieve sequence matching topological domain begin/end indices
                sequence = sequences.get(uniprot_swissprot_id)

                if sequence is not None:
                    # Find the corresponding index of the motif in the protein sequence
                    motif_len = len(motif_seq)
                    trimmed_motif_seq = motif_seq[topology_trim_begin: motif_len - topology_trim_end]
                    trimmed_begin = sequence.find(trimmed_motif_seq)
                    del sequence

                    if trimmed_begin != -1:
                        trimmed_end = trimmed_begin + len(trimmed_motif_seq)

                        # Determine which topological domain the motif exists within
                        for domain_tuple in current_topological_domains:
                            topological_domain_type, description, begin_position, end_position = domain_tuple
                            begin_idx = begin_position - 1
                            end_idx = end_position - 1

                            if topological_domain_type is None:
                                topological_domain_type = ""
                            if description is None:
                                description = ""

                            motif_within = trimmed_begin >= begin_idx and trimmed_end <= end_idx
                            head_within = trimmed_begin >= begin_idx and trimmed_begin <= end_idx and trimmed_end > end_idx
                            tail_within = trimmed_begin < begin_idx and trimmed_end > begin_idx and trimmed_end <= end_idx
                            middle_within = trimmed_begin < begin_idx and trimmed_end > end_idx

                            if motif_within:
                                # Motif is fully within current topological domain
                                domain_types.append(topological_domain_type)
                                domain_descriptions.append(description)
                                break

                            elif head_within or tail_within or middle_within:
                                # First part of motif is within current topological domain
                                domain_types.append(topological_domain_type)
                                domain_descriptions.append(description)

            del current_topological_domains


        domain_types_vals.append(";".join(domain_types))
        domain_descriptions_vals.append(";".join(domain_descriptions))

    if len(domain_types_vals) != len(motif_seqs):
        raise Exception(f"Topology list (len={len(domain_type_vals)}) does not match chunk_df (len={len(motif_seqs)})")

    return domain_types_vals, domain_descriptions_vals

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
    verbose = predictor_params["topology_verbose"]

    topological_domains, sequences = get_topological_domains(path = uniprot_path)
    uniprot_ids = data_df["uniprot"]

    predict_partial = partial(predict_chunk, topology_trim_begin = topology_trim_begin,
                              topology_trim_end = topology_trim_end, topological_domains = topological_domains,
                              sequences = sequences, verbose = verbose)

    for motif_col in motif_cols:
        print(f"\tPredicting motif topologies for col: {motif_col}")
        motif_seqs = data_df[motif_col]

        domain_types_vals = []
        domain_descriptions_vals = []

        with trange(round(len(data_df) / chunk_size), desc="\tProcessing dataframe chunks...") as pbar:
            for n, i in enumerate(range(0, len(data_df), chunk_size)):
                chunk_tuple = (motif_seqs[i:i + chunk_size], uniprot_ids[i:i + chunk_size])
                chunk_domain_types_vals, chunk_domain_descriptions_vals = predict_partial(chunk_tuple)

                domain_types_vals.extend(chunk_domain_types_vals)
                domain_descriptions_vals.extend(chunk_domain_descriptions_vals)

                pbar.update()

        data_df[motif_col + "_topology_type"] = domain_types_vals
        data_df[motif_col + "_topology_description"] = domain_descriptions_vals

    return data_df