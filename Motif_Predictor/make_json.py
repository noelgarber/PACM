import os
import warnings
import yaml
from tqdm import trange
import multiprocessing
from functools import partial
import numpy as np
import pandas as pd
import json
import ujson
import pickle
from Motif_Predictor.load_predictor_config import load_config
from Motif_Predictor.map_homologies import map_homologies

predictor_params = load_config(verbose=True)

# ------------------------------------------ Parse DataFrame into Database ---------------------------------------------
def get_taxids(predictor_params = predictor_params):
    # Extracts reference and target taxids from params

    homolog_selection_mode = predictor_params["homology_params"]["homolog_selection_mode"]
    if homolog_selection_mode == "best":
        taxids = list(predictor_params["protein_seqs_paths"].keys())
        reference_taxid = taxids[0]
        target_taxids = taxids[1:]
    else:
        keys = list(predictor_params["protein_seqs_paths"].keys())
        reference_taxids = [key.split("_vs_")[0] for key in keys]
        if all([taxid == reference_taxids[0] for taxid in reference_taxids]):
            reference_taxid = int(reference_taxids[0])
        else:
            raise Exception(f"More than one reference taxid was found in the path keys: {reference_taxids}")
        target_taxids = [int(key.split("_vs_")[1]) for key in keys]

    return reference_taxid, target_taxids

def parse_dfs(csv_path, predictor_params = predictor_params):
    # Parse dataframe into taxid-specific dataframes

    df = pd.read_csv(csv_path)
    reference_taxid, target_taxids = get_taxids(predictor_params)

    # Get base cols for the reference taxid when homology is not being considered
    base_cols = []
    for col in df.columns:
        if all([not str(taxid) in col for taxid in target_taxids]):
            base_cols.append(col)

    # Get cols for each compared homologous taxid
    cols_by_taxid = {}
    for taxid in target_taxids:
        taxid_cols = [col for col in df.columns if str(taxid) in col]
        host_cols, homolog_cols = [], []
        if any(["Host" in col for col in taxid_cols]):
            if "Host" in col:
                host_cols.append(col)
            else:
                homolog_cols.append(col)
        cols_by_taxid[taxid] = (host_cols, homolog_cols)

    # Split dataframe into taxid-specific dataframes
    reference_df = df[base_cols].copy()
    target_taxid_dfs = {}
    for target_taxid in target_taxids:
        host_cols, homolog_cols = cols_by_taxid[target_taxid]
        host_taxid_df = df[host_cols].copy()
        homolog_taxid_df = df[homolog_cols].copy()
        target_taxid_dfs[target_taxid] = (host_taxid_df, homolog_taxid_df)

    return reference_taxid, reference_df, target_taxid_dfs

def apply_num_suffix(num):
    # Applies suffixes to numbers, i.e. 1st, 2nd, 3rd, 4th, etc.

    num = str(int(num))
    if num[-1] == "1":
        num_with_suffix = num + "st"
    elif num[-1] == "2":
        num_with_suffix = num + "nd"
    elif num[-1] == "3":
        num_with_suffix = num + "rd"
    else:
        num_with_suffix = num + "th"

    return num_with_suffix

def convert_chunk(chunk_tuple, gene_id_col, gene_name_col, peptide_id_col, novel_numbered, classical_numbered = None,
                  compare_classical_method = False):
    # Generates unpaired dictionary chunk

    df_chunk, taxid = chunk_tuple

    taxid_chunk_dict = {}
    chunk_specificity_unassigned = 0
    chunk_specificity_unassigned_toplevel = 0
    for i, row in df_chunk.iterrows():
        ensembl_gene_id = row[gene_id_col]
        gene_name = row[gene_name_col]
        if taxid_chunk_dict.get(ensembl_gene_id) is None:
            taxid_chunk_dict[ensembl_gene_id] = {"gene_name": gene_name}

        ensembl_peptide_id = row[peptide_id_col]
        if taxid_chunk_dict[ensembl_gene_id].get(ensembl_peptide_id) is None:
            taxid_chunk_dict[ensembl_gene_id][ensembl_peptide_id] = {}

        # Insert novel motifs into a dict called novel
        novel_vals_dict = {}
        for novel_num in novel_numbered:
            novel_num_vals_dict = {}

            # TODO Add mention of N-terminal phospho influence on binding strength; check congruence with phospho-tract SPR paper
            novel_num_vals_dict["start"] = row[f"{novel_num}_motif_start"]
            novel_num_vals_dict["sequence"] = row[f"{novel_num}_motif"]

            novel_num_vals_dict["topology"] = {}
            novel_num_vals_dict["topology"]["type"] = row[f"{novel_num}_motif_topology_type"]
            novel_num_vals_dict["topology"]["description"] = row[f"{novel_num}_motif_topology_description"]
            novel_num_vals_dict["topology"]["cytoplasmic_accessible"] = row[f"{novel_num}_motif_topology_accessible"]

            novel_num_vals_dict["classification_score"] = row[f"{novel_num}_total_motif_score"]
            novel_num_vals_dict["binding_score"] = row[f"{novel_num}_binding_motif_score"]
            novel_num_vals_dict["final_call"] = row[f"{novel_num}_final_call"]
            try:
                novel_num_vals_dict["specificity_score"] = row[f"{novel_num}_motif_specificity_score"]
            except Exception as e:
                novel_num_vals_dict["specificity_score"] = np.nan
                chunk_specificity_unassigned += 1
                if novel_num[-3:] == "1st":
                    chunk_specificity_unassigned_toplevel += 1

            if novel_num_vals_dict["final_call"]:
                novel_num_vals_dict["masked_binding_score"] = novel_num_vals_dict["binding_score"]
            else:
                novel_num_vals_dict["masked_binding_score"] = 0.0

            if compare_classical_method:
                novel_num_vals_dict["classical_score"] = row[f"{novel_num}_classical_score"]

            novel_vals_dict[f"{novel_num}_motif"] = novel_num_vals_dict

        taxid_chunk_dict[ensembl_gene_id][ensembl_peptide_id]["novel"] = novel_vals_dict

        # Insert classical motifs into a dict called classical
        if compare_classical_method:
            classical_vals_dict = {}
            for classical_num in classical_numbered:
                classical_num_vals_dict = {}

                classical_num_vals_dict["sequence"] = row[f"{classical_num}_motif"]

                classical_num_vals_dict["topology"] = {}
                classical_num_vals_dict["topology"]["type"] = row[f"{classical_num}_motif_topology_type"]
                classical_num_vals_dict["topology"]["description"] = row[f"{classical_num}_motif_topology_description"]
                classical_num_vals_dict["topology"]["cytoplasmic_accessible"] = row[f"{classical_num}_motif_topology_accessible"]

                classical_num_vals_dict["classical_score"] = row[f"{classical_num}_total_motif_score"]

                classical_vals_dict[f"{classical_num}_motif"] = classical_num_vals_dict

            taxid_chunk_dict[ensembl_gene_id][ensembl_peptide_id]["classical"] = classical_vals_dict

    return (taxid_chunk_dict, taxid, chunk_specificity_unassigned, chunk_specificity_unassigned_toplevel)

def parse_data(taxid_dfs, ref_gene_col, ref_gene_name_col, ref_protein_col, return_count,
               compare_classical_method = False, chunk_size = 1000):
    # Generates unpaired dictionary

    # Get substrings marking numbered novel and classical motifs
    nums_with_suffixes = [apply_num_suffix(num) for num in np.arange(1, return_count+1)]
    if compare_classical_method:
        novel_numbered = [f"Novel_{num_with_suffix}" for num_with_suffix in nums_with_suffixes]
        classical_numbered = [f"Classical_{num_with_suffix}" for num_with_suffix in nums_with_suffixes]
    else:
        novel_numbered = nums_with_suffixes
        classical_numbered = []

    # Get chunks of dataframes with associated taxids
    chunks = []
    for taxid, taxid_df in taxid_dfs.items():
        for i in range(0, len(taxid_df), chunk_size):
            df_chunk = taxid_df[i:i+chunk_size]
            if isinstance(df_chunk, pd.DataFrame):
                chunks.append((df_chunk, taxid))

    partial_func = partial(convert_chunk, gene_id_col = ref_gene_col, gene_name_col = ref_gene_name_col,
                           peptide_id_col = ref_protein_col, novel_numbered = novel_numbered,
                           classical_numbered = classical_numbered, compare_classical_method = compare_classical_method)
    pool = multiprocessing.Pool()
    unpaired_dict = {taxid: {} for taxid in taxid_dfs.keys()}
    specificity_unassigned_toplevel = {taxid: 0 for taxid in taxid_dfs.keys()}

    with trange(len(chunks), desc=f"Parsing dataframes into dictionary...") as pbar:
        for result in pool.imap_unordered(partial_func, chunks):
            taxid_chunk_dict, taxid, chunk_specificity_unassigned, chunk_specificity_unassigned_toplevel = result
            unpaired_dict[taxid] = unpaired_dict[taxid] | taxid_chunk_dict
            specificity_unassigned_toplevel[taxid] += chunk_specificity_unassigned_toplevel
            pbar.update()

    pool.close()
    pool.join()

    return unpaired_dict

def find_homologous_motifs(ref_motif_seq, homolog_gene_ids, target_taxid_dict, folder = "novel",
                           favor_cytoplasmic = True):
    # Find homologous motifs

    best_homolog_ids = []
    best_identity = 0.0
    for target_gene_id in homolog_gene_ids:
        target_gene_dict = target_taxid_dict[target_gene_id]
        target_gene_name = target_gene_dict.get("gene_name")
        for target_protein_id, target_protein_dict in target_gene_dict.items():
            target_novel_dict = target_protein_dict[folder]
            for target_novel_num_key, target_vals_dict in target_novel_dict.items():
                target_motif_seq = target_vals_dict["sequence"]
                residue_matches = np.equal(list(ref_motif_seq), list(target_motif_seq))
                identity_percent = residue_matches.mean()
                if identity_percent > best_identity:
                    best_homolog_ids = [(target_gene_id, target_gene_name, target_protein_id,
                                         target_novel_num_key, identity_percent)]
                    best_identity = identity_percent
                elif identity_percent == best_identity:
                    best_homolog_ids.append((target_gene_id, target_gene_name, target_protein_id,
                                             target_novel_num_key, identity_percent))

    # Get data for homologous motifs
    best_homolog_motifs = {}
    for i, id_tuple in enumerate(best_homolog_ids):
        target_gene_id, target_gene_name, target_protein_id, target_novel_num_key, identity_percent = id_tuple
        target_protein_dict = target_taxid_dict[target_gene_id][target_protein_id]
        target_vals_dict = target_protein_dict[folder][target_novel_num_key].copy()
        target_vals_dict["homolog_gene_id"] = target_gene_id
        target_vals_dict["homolog_gene_name"] = target_gene_name
        target_vals_dict["homolog_protein_id"] = target_protein_id
        target_vals_dict["homology_identity"] = identity_percent
        best_homolog_motifs[i] = target_vals_dict

    # Pick preferred homologous motif; preference is given to entries with topology information
    best_homolog_motif = next(best_homolog_motifs.values())
    for i, homolog_vals_dict in best_homolog_motifs.items():
        current_topology_type = homolog_vals_dict["topology"].get("type")
        if current_topology_type is not None and current_topology_type != "":
            best_topology_type = best_homolog_motif["topology"].get("type")
            if best_topology_type is not None and best_topology_type != "" and favor_cytoplasmic:
                current_cytoplasmic_accessible = homolog_vals_dict["topology"].get("cytoplasmic_accessible")
                best_cytoplasmic_accessible = best_homolog_motif["topology"].get("cytoplasmic_accessible")
                if current_cytoplasmic_accessible and not best_cytoplasmic_accessible:
                    best_homolog_motif = homolog_vals_dict
            else:
                best_homolog_motif = homolog_vals_dict

    best_homolog_motifs["best"] = best_homolog_motif

    return best_homolog_motifs

def assign_homolog_motifs(best_homolog_motifs, target_taxid, ref_gene_id, ref_protein_id, ref_num_key, correlated_dict):
    # Helper function that assigns best_homolog_motifs to correlated_dict at the appropriate sub-dict

    if correlated_dict[ref_gene_id][ref_protein_id][ref_num_key].get("homologs") is None:
        correlated_dict[ref_gene_id][ref_protein_id][ref_num_key]["homologs"] = {target_taxid: {}}
    elif correlated_dict[ref_gene_id][ref_protein_id][ref_num_key]["homologs"].get(target_taxid) is None:
        correlated_dict[ref_gene_id][ref_protein_id][ref_num_key]["homologs"][target_taxid] = {}

    correlated_dict[ref_gene_id][ref_protein_id][ref_num_key]["homologs"][target_taxid] = best_homolog_motifs

def correlate_homologs_gene(ref_gene_id, reference_gene_dict, homology_target_dict, target_taxid_dict,
                            compare_classical_method = False):
    '''
    Helper function that finds homologs for motifs belonging to a single reference gene

    Args:
        ref_gene_id (str):               Gene of interest
        reference_gene_dict (dict):      Main dictionary of motif results by taxid
        homology_target_dict (dict):     Dictionary of homologous genes for target taxid
        target_taxid_dict (dict):        Reference taxonomic identifier to look for homologs for
        compare_classical_method (bool): Whether to compare results from a classical model assessed in parallel

    Returns:
        novel_results (list):            List of novel motif homologs with keys for assigning back to the main dict
        classical_results (list):        List of classical motif homologs with keys for assigning back to the main dict
    '''

    novel_results = []
    classical_results = []

    # Get the list of Ensembl Gene IDs in the homologous target species
    homolog_gene_ids = homology_target_dict.get(ref_gene_id)

    # If there are known homologs, loop over reference protein isoforms to scan for homologous motifs within
    if homolog_gene_ids:
        ref_protein_ids = list(reference_gene_dict.keys())
        ref_protein_ids.remove("gene_name")
        for ref_protein in ref_protein_ids:
            reference_protein_dict = reference_gene_dict[ref_protein]
            reference_novel_dict = reference_protein_dict["novel"]

            for ref_novel_num, reference_vals_dict in reference_novel_dict.items():
                ref_motif_seq = reference_vals_dict["sequence"]
                best_novel_homolog_motifs = find_homologous_motifs(ref_motif_seq, homolog_gene_ids, target_taxid_dict)
                novel_results.append((best_novel_homolog_motifs, target_taxid, ref_gene, ref_protein, ref_novel_num))

            if compare_classical_method:
                reference_classical_dict = reference_protein_dict["classical"]
                for ref_classical_num, reference_classical_vals_dict in reference_classical_dict.items():
                    ref_motif_seq = reference_classical_vals_dict["sequence"]
                    best_classical_homolog_motifs = find_homologous_motifs(ref_motif_seq, homolog_gene_ids, target_taxid_dict)
                    classical_results.append((best_classical_homolog_motifs, target_taxid, ref_gene, ref_protein, ref_classical_num))

    return (novel_results, classical_results)

def correlate_homologs_chunk(chunk, homology_target_dict, target_taxid_dict, compare_classical_method=False):
    '''
    Function for correlating results from other taxids with the reference taxid to find homologous motifs

    Args:
        chunk (list):                    List of tuples of (ref_gene_id, reference_gene_dict)
        homology_target_dict (dict):     Dictionary of homologous genes for current taxid
        target_taxid_dict (int):         Results dictionary for target taxid
        compare_classical_method (bool): Whether to compare results from a classical model assessed in parallel

    Returns:
        novel_results_chunk (list):      List of novel motif homologs with keys for assigning back to the main dict
        classical_results_chunk (list):  List of classical motif homologs with keys for assigning back to the main dict
    '''

    novel_results_chunk = []
    classical_results_chunk = []

    # Loop over Ensembl Gene IDs in the reference taxid data and their associated dictionaries
    for ref_gene, reference_gene_dict in chunk:
        gene_results = correlate_homologs_gene(ref_gene, reference_gene_dict, homology_target_dict, target_taxid_dict,
                                               compare_classical_method)
        novel_gene_results, classical_gene_results = gene_results
        novel_results_chunk.extend(novel_gene_results)
        classical_results_chunk.extend(classical_gene_results)

    return (novel_results_chunk, classical_results_chunk)

def correlate_homologs(data_dict, homology_target_dicts, reference_taxid = 9606, compare_classical_method = False,
                       chunk_size = 500):
    '''
    Function for correlating results from other taxids with the reference taxid to find homologous motifs

    Args:
        data_dict (dict):                Main dictionary of motif results by taxid
        homology_target_dicts (dict):    Dictionary of homologous genes across taxids
        reference_taxid (int):           Reference taxonomic identifier to look for homologs for
        compare_classical_method (bool): Whether to compare results from a classical model assessed in parallel

    Returns:
        correlated_dict (dict):          Dictionary for the reference taxid with homologous motifs added for each motif
    '''

    taxids = list(data_dict.keys())
    target_taxids = taxids.copy()
    target_taxids.remove(reference_taxid)

    reference_taxid_dict = data_dict[reference_taxid]
    correlated_dict = reference_taxid_dict.copy()

    novel_results = []
    classical_results = []

    # Loop over taxids to search for homologs within
    for target_taxid in target_taxids:
        target_taxid_dict = data_dict[target_taxid]

        # Get dictionary of ref_gene_id --> target_gene_ids
        homology_target_dict = homology_target_dicts[target_taxid]

        pairs = [(ref_gene, reference_gene_dict) for ref_gene, reference_gene_dict in reference_taxid_dict.items()]
        chunks = []
        for i in np.arange(0, len(pairs), chunk_size):
            chunks.append(pairs[i:i+chunk_size])

        correlate_partial_func = partial(correlate_homologs_chunk, homology_target_dict = homology_target_dict,
                                         target_taxid_dict = target_taxid_dict,
                                         compare_classical_method = compare_classical_method)
        pool = multiprocessing.Pool()

        with trange(len(chunks), desc=f"Correlating homologs for target TaxID {target_taxid}...") as pbar:
            for results_chunk in pool.imap_unordered(correlate_partial_func, chunks):
                novel_results_chunk, classical_results_chunk = results_chunk
                novel_results.extend(novel_results_chunk)
                classical_results.extend(classical_results_chunk)
                pbar.update()

        pool.close()
        pool.join()

    print(f"Assigning results to correlated dictionary...")
    for best_novel_homolog_motifs, target_taxid, ref_gene, ref_protein, ref_novel_num in novel_results:
        assign_homolog_motifs(best_novel_homolog_motifs, target_taxid, ref_gene, ref_protein, ref_novel_num,
                              correlated_dict)
    for best_classical_homolog_motifs, target_taxid, ref_gene, ref_protein, ref_classical_num in classical_results:
        assign_homolog_motifs(best_classical_homolog_motifs, target_taxid, ref_gene, ref_protein, ref_classical_num,
                              correlated_dict)

    return correlated_dict

def blank_correlated_df(return_count, target_taxids, compare_classical_method):
    # Helper function to generate empty correlated_df

    base_cols = ["ensembl_gene_id", "external_gene_name"]
    novel_cols = []
    classical_cols = []
    for i in np.arange(1, return_count + 1):
        num_with_suffix = apply_num_suffix(i)
        novel_cols.extend([f"Novel_{num_with_suffix}_motif",
                           f"Novel_{num_with_suffix}_motif_topology_type",
                           f"Novel_{num_with_suffix}_motif_topology_description",
                           f"Novel_{num_with_suffix}_motif_cytoplasmic_accessible",
                           f"Novel_{num_with_suffix}_motif_classification_score",
                           f"Novel_{num_with_suffix}_motif_binding_score",
                           f"Novel_{num_with_suffix}_motif_masked_binding_score",
                           f"Novel_{num_with_suffix}_motif_final_call",
                           f"Novel_{num_with_suffix}_motif_specificity_score"])
        for target_taxid in target_taxids:
            novel_cols.extend([f"{target_taxid}_Novel_{num_with_suffix}_homolog_motif",
                               f"{target_taxid}_Novel_{num_with_suffix}_homolog_motif_topology_type",
                               f"{target_taxid}_Novel_{num_with_suffix}_homolog_motif_topology_description",
                               f"{target_taxid}_Novel_{num_with_suffix}_homolog_motif_cytoplasmic_accessible",
                               f"{target_taxid}_Novel_{num_with_suffix}_homolog_motif_classification_score",
                               f"{target_taxid}_Novel_{num_with_suffix}_homolog_motif_binding_score",
                               f"{target_taxid}_Novel_{num_with_suffix}_homolog_motif_masked_binding_score",
                               f"{target_taxid}_Novel_{num_with_suffix}_homolog_motif_final_call",
                               f"{target_taxid}_Novel_{num_with_suffix}_homolog_motif_specificity_score"])

        if compare_classical_method:
            novel_cols.append(f"Novel_{num_with_suffix}_motif_classical_score")
            classical_cols.extend([f"Classical_{num_with_suffix}_motif",
                                   f"Classical_{num_with_suffix}_motif_topology_type",
                                   f"Classical_{num_with_suffix}_motif_topology_description",
                                   f"Classical_{num_with_suffix}_motif_cytoplasmic_accessible",
                                   f"Classical_{num_with_suffix}_motif_classical_score"])
            for target_taxid in target_taxids:
                classical_cols.extend([f"{target_taxid}_Classical_{num_with_suffix}_homolog_motif",
                                       f"{target_taxid}_Classical_{num_with_suffix}_homolog_motif_topology_type",
                                       f"{target_taxid}_Classical_{num_with_suffix}_homolog_motif_topology_description",
                                       f"{target_taxid}_Classical_{num_with_suffix}_homolog_motif_cytoplasmic_accessible",
                                       f"{target_taxid}_Classical_{num_with_suffix}_homolog_motif_classical_score"])

    base_cols.extend(novel_cols)
    base_cols.extend(classical_cols)

    correlated_df = pd.DataFrame(columns=base_cols)

    return correlated_df

def generate_correlated_df(correlated_dict, target_taxids, return_count, compare_classical_method = False,
                           classical_lower_better = True):

    correlated_df = blank_correlated_df(return_count, target_taxids, compare_classical_method)

    with trange(len(correlated_dict), desc=f"Populating correlated homolog dataframe...") as pbar:
        for i, (ref_gene_id, reference_gene_dict) in enumerate(correlated_dict.items()):
            gene_name = reference_gene_dict.get("gene_name")
            correlated_df.at[i, "external_gene_name"] = gene_name

            protein_isoform_ids = list(reference_gene_dict.keys())
            protein_isoform_ids.remove("gene_name") # gene_name is the only other non-protein-id key in this sub-dict

            novel_masked_tuples = []
            novel_masked_scores = []
            classical_tuples = []
            classical_scores = []
            for ref_protein_id in protein_isoform_ids:
                novel_vals_dict = reference_gene_dict[ref_protein_id]["novel"]
                for novel_num, novel_num_vals_dict in novel_vals_dict.items():
                    masked_binding_score = novel_num_vals_dict["masked_binding_score"]
                    novel_masked_tuples.append((ref_protein_id, novel_num, masked_binding_score))
                    novel_masked_scores.append(masked_binding_score)
                if compare_classical_method:
                    classical_vals_dict = reference_gene_dict[ref_protein_id]["classical"]
                    for classical_num, classical_num_vals_dict in classical_vals_dict.items():
                        classical_score = classical_num_vals_dict["classical_score"]
                        classical_tuples.append((ref_protein_id, classical_num, classical_score))
                        classical_scores.append(classical_score)

            ranked_novel_indices = np.argsort(novel_masked_scores)
            if classical_lower_better:
                ranked_classical_indices = np.argsort(classical_scores)
            else:
                ranked_classical_indices = np.argsort(classical_scores * -1)

            for j in np.arange(return_count):
                # Process novel motifs
                ref_protein_id, novel_num, masked_binding_score = novel_masked_tuples[ranked_novel_indices[j]]
                novel_num_vals_dict = reference_gene_dict[ref_protein_id]["novel"][novel_num]

                correlated_df.at[i, f"{novel_num}_motif"] = novel_num_vals_dict["sequence"]
                correlated_df.at[i, f"{novel_num}_motif_start"] = novel_num_vals_dict["start"]

                topo_info = novel_num_vals_dict["topology"]
                correlated_df.at[i, f"{novel_num}_motif_topology_type"] = topo_info["type"]
                correlated_df.at[i, f"{novel_num}_motif_topology_description"] = topo_info["description"]
                correlated_df.at[i, f"{novel_num}_motif_cytoplasmic_accessible"] = topo_info["cytoplasmic_accessible"]

                correlated_df.at[i, f"{novel_num}_motif_classification_score"] = novel_num_vals_dict["classification_score"]
                correlated_df.at[i, f"{novel_num}_motif_binding_score"] = novel_num_vals_dict["binding_score"]
                correlated_df.at[i, f"{novel_num}_motif_final_call"] = novel_num_vals_dict["final_call"]
                correlated_df.at[i, f"{novel_num}_motif_specificity_score"] = novel_num_vals_dict["specificity_score"]
                if compare_classical_method:
                    correlated_df.at[i, f"{novel_num}_motif"] = novel_num_vals_dict["classical_score"]

                # Process novel motif homologs
                homologs = reference_gene_dict[ref_protein_id][novel_num]["homologs"]
                for target_taxid, best_homolog_motifs in homologs.items():
                    if best_homolog_motifs is not None:
                        best_homolog_motif = best_homolog_motifs["best"]
                        if best_homolog_motif is not None:
                            novel_homolog_num = f"{target_taxid}_{novel_num}_homolog"

                            correlated_df.at[i, f"{novel_homolog_num}_gene_id"] = best_homolog_motif["homolog_gene_id"]
                            correlated_df.at[i, f"{novel_homolog_num}_gene_name"] = best_homolog_motif["homolog_gene_name"]
                            correlated_df.at[i, f"{novel_homolog_num}_protein_id"] = best_homolog_motif["homolog_protein_id"]
                            correlated_df.at[i, f"{novel_homolog_num}_motif"] = best_homolog_motif["sequence"]
                            correlated_df.at[i, f"{novel_homolog_num}_motif_start"] = best_homolog_motif["start"]
                            correlated_df.at[i, f"{novel_homolog_num}_motif_identity"] = best_homolog_motif["homology_identity"]

                            homolog_topo_info = best_homolog_motif["topology"]
                            correlated_df.at[i, f"{novel_homolog_num}_motif_topology_type"] = homolog_topo_info["type"]
                            correlated_df.at[i, f"{novel_homolog_num}_motif_topology_description"] = topo_info["description"]
                            correlated_df.at[i, f"{novel_homolog_num}_motif_cytoplasmic_accessible"] = topo_info["cytoplasmic_accessible"]

                            correlated_df.at[i, f"{novel_homolog_num}_motif_classification_score"] = best_homolog_motif["classification_score"]
                            correlated_df.at[i, f"{novel_homolog_num}_motif_binding_score"] = best_homolog_motif["binding_score"]
                            correlated_df.at[i, f"{novel_homolog_num}_motif_final_call"] = best_homolog_motif["final_call"]
                            correlated_df.at[i, f"{novel_homolog_num}_motif_specificity_score"] = best_homolog_motif["specificity_score"]
                            if compare_classical_method:
                                correlated_df.at[i, f"{novel_homolog_num}_motif"] = best_homolog_motif["classical_score"]

                # Process classical motifs
                if compare_classical_method:
                    ref_protein_id, classical_num, classical_score = classical_tuples[ranked_classical_indices[j]]
                    classical_num_vals_dict = reference_gene_dict[ref_protein_id]["classical"][classical_num]

                    correlated_df.at[i, f"{classical_num}_motif"] = classical_num_vals_dict["sequence"]

                    topo_info = classical_num_vals_dict["topology"]
                    correlated_df.at[i, f"{classical_num}_motif_topology_type"] = topo_info["type"]
                    correlated_df.at[i, f"{classical_num}_motif_topology_description"] = topo_info["description"]
                    correlated_df.at[i, f"{classical_num}_motif_cytoplasmic_accessible"] = topo_info["cytoplasmic_accessible"]

                    correlated_df.at[i, f"{classical_num}_motif"] = classical_num_vals_dict["classical_score"]

                    # Process classical motif homologs
                    classical_homologs = reference_gene_dict[ref_protein_id][classical_num]["homologs"]
                    for target_taxid, best_homolog_motifs in classical_homologs.items():
                        if best_homolog_motifs is not None:
                            best_homolog_motif = best_homolog_motifs["best"]
                            if best_homolog_motif is not None:
                                classical_homolog_num = f"{target_taxid}_{classical_num}_homolog"

                                correlated_df.at[i, f"{classical_homolog_num}_gene_id"] = best_homolog_motif["homolog_gene_id"]
                                correlated_df.at[i, f"{classical_homolog_num}_gene_name"] = best_homolog_motif["homolog_gene_name"]
                                correlated_df.at[i, f"{classical_homolog_num}_protein_id"] = best_homolog_motif["homolog_protein_id"]
                                correlated_df.at[i, f"{classical_homolog_num}_motif"] = best_homolog_motif["sequence"]
                                correlated_df.at[i, f"{classical_homolog_num}_motif_identity"] = best_homolog_motif["homology_identity"]

                                homolog_topo_info = best_homolog_motif["topology"]
                                correlated_df.at[i, f"{classical_homolog_num}_motif_topology_type"] = homolog_topo_info["type"]
                                correlated_df.at[i, f"{classical_homolog_num}_motif_topology_description"] = topo_info["description"]
                                correlated_df.at[i, f"{classical_homolog_num}_motif_cytoplasmic_accessible"] = topo_info["cytoplasmic_accessible"]

                                correlated_df.at[i, f"{classical_homolog_num}_motif"] = best_homolog_motif["classical_score"]

            pbar.update()

    return correlated_df

def convert_to_json(taxid_dfs, predictor_params = predictor_params, db_path = None, correlate_homology = True):
    # Main function for converting dataset to JSON database

    taxids = list(taxid_dfs.keys())
    reference_taxid = predictor_params["reference_taxid"]
    compare_classical_method = predictor_params["compare_classical_method"]
    classical_lower_better = predictor_params["classical_lower_better"]
    return_count = predictor_params["return_count"]
    if not db_path:
        db_path = predictor_params["db_params"]["db_path"]

    # Generate dictionary and dump to json file
    ref_gene_col = predictor_params["homology_params"]["ref_gene_col"]
    ref_gene_name_col = predictor_params["homology_params"]["ref_gene_name_col"]
    ref_protein_col = predictor_params["homology_params"]["ref_protein_col"]
    data_dict = parse_data(taxid_dfs, ref_gene_col, ref_gene_name_col, ref_protein_col, return_count,
                           compare_classical_method)
    del taxid_dfs

    print(f"Saving data_dict to {db_path}")
    with open(db_path, "w") as json_file:
        ujson.dump(data_dict, json_file, indent=4)

    pkl_path = db_path.split(".json")[0] + ".pkl"
    print(f"Saving data_dict to {pkl_path} for added speed when reloading")
    with open(pkl_path, "wb") as file:
        pickle.dump(data_dict, file)
    print(f"Done!")

    if correlate_homology:
        target_taxids = list(taxids)
        target_taxids.remove(reference_taxid)
        homology_target_dicts = map_homologies(reference_taxid, target_taxids)

        correlated_dict = correlate_homologs(data_dict, homology_target_dicts, reference_taxid, compare_classical_method)
        correlated_db_path = db_path.split(".json")[0] + "_with_homologs.json"
        print(f"Saving correlated_dict to {correlated_db_path}")
        with open(correlated_db_path, "w") as correlated_json_file:
            ujson.dump(correlated_dict, correlated_json_file, indent=4)

        correlated_pkl_path = correlated_db_path.split(".json")[0] + ".pkl"
        print(f"Saving data_dict to {correlated_pkl_path} for added speed when reloading")
        with open(correlated_pkl_path, "wb") as file:
            pickle.dump(data_dict, file)
        print(f"Done!")

        target_taxids = list(taxids)[1:]
        correlated_df = generate_correlated_df(correlated_dict, target_taxids, return_count, compare_classical_method,
                                               classical_lower_better)
        correlated_csv_path = correlated_db_path.split(".json")[0] + ".csv"
        correlated_df.to_csv(correlated_csv_path)

    else:
        correlated_dict, correlated_df = None, None

    return data_dict, correlated_dict, correlated_df