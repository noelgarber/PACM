import os
import yaml
from tqdm import trange
import multiprocessing
from functools import partial
import numpy as np
import pandas as pd
import json
from Motif_Predictor.load_predictor_config import load_config

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

def generate_unpaired_chunk(chunk_tuple, gene_cols, reference_taxid, novel_numbered,
                            classical_numbered = None, compare_classical_method = False):
    # Generates unpaired dictionary chunk

    df_chunk, taxid = chunk_tuple

    taxid_chunk_dict = {}
    for i, row in df_chunk.iterrows():
        gene_col = gene_cols[taxid]
        ensembl_gene_id = row[gene_col]
        taxid_chunk_dict[ensembl_gene_id] = {}

        # Insert novel motifs into a dict called novel
        novel_vals_dict = {}
        for novel_num in novel_numbered:
            novel_num_vals_dict = {}

            val_col_names = [f"{novel_num}_motif", f"{novel_num}_motif_topology_type",
                             f"{novel_num}_motif_topology_description", f"{novel_num}_motif_topology_accessible",
                             f"{novel_num}_total_motif_score", f"{novel_num}_binding_motif_score",
                             f"{novel_num}_final_call", f"{novel_num}_motif_specificity_score"]
            if taxid != reference_taxid:
                val_col_names = [f"{taxid}_{col_name}" for col_name in val_col_names]
                classical_score_col = f"{taxid}_{novel_num}_classical_score"
            else:
                classical_score_col = f"{novel_num}_classical_score"

            novel_num_vals_dict["sequence"] = row[val_col_names[0]]

            novel_num_vals_dict["topology"] = {}
            novel_num_vals_dict["topology"]["type"] = row[val_col_names[1]]
            novel_num_vals_dict["topology"]["description"] = row[val_col_names[2]]
            novel_num_vals_dict["topology"]["cytoplasmic_accessible"] = row[val_col_names[3]]

            novel_num_vals_dict["classification_score"] = row[val_col_names[4]]
            novel_num_vals_dict["binding_score"] = row[val_col_names[5]]
            novel_num_vals_dict["final_call"] = row[val_col_names[6]]
            novel_num_vals_dict["specificity_score"] = row[val_col_names[7]]
            if novel_num_vals_dict["final_call"]:
                novel_num_vals_dict["masked_binding_score"] = novel_num_vals_dict["binding_score"]
            else:
                novel_num_vals_dict["masked_binding_score"] = 0.0

            if compare_classical_method:
                novel_num_vals_dict["classical_score"] = row[classical_score_col]

            novel_vals_dict[f"{novel_num}_motif"] = novel_num_vals_dict

        taxid_chunk_dict[ensembl_gene_id]["novel"] = novel_vals_dict

        # Insert classical motifs into a dict called classical
        if compare_classical_method:
            classical_vals_dict = {}
            for classical_num in classical_numbered:
                classical_num_vals_dict = {}

                val_col_names = [f"{classical_num}_motif", f"{classical_num}_motif_topology_type",
                                 f"{classical_num}_motif_topology_description",
                                 f"{classical_num}_motif_topology_accessible", f"{classical_num}_total_motif_score"]
                if taxid != reference_taxid:
                    val_col_names = [f"{taxid}_{col_name}" for col_name in val_col_names]

                classical_num_vals_dict["sequence"] = row[val_col_names[0]]

                classical_num_vals_dict["topology"] = {}
                classical_num_vals_dict["topology"]["type"] = row[val_col_names[1]]
                classical_num_vals_dict["topology"]["description"] = row[val_col_names[2]]
                classical_num_vals_dict["topology"]["cytoplasmic_accessible"] = row[val_col_names[3]]

                classical_num_vals_dict["classical_score"] = row[val_col_names[4]]

                classical_vals_dict[f"{classical_num}_motif"] = classical_num_vals_dict

            taxid_chunk_dict[ensembl_gene_id]["classical"] = classical_vals_dict

    return (taxid_chunk_dict, taxid)

def generate_unpaired(ref_gene_col, reference_taxid, reference_df, target_taxids, target_taxid_dfs, novel_numbered,
                      classical_numbered = None, compare_classical_method = False, json_path = None, chunk_size = 1000):
    # Generates unpaired dictionary

    taxids = list(target_taxids)
    taxids.insert(0, reference_taxid)
    gene_cols = {reference_taxid: ref_gene_col}
    for target_taxid in target_taxids:
        gene_cols[target_taxid] = f"{target_taxid}_best_homolog_id"

    dfs_by_taxid = target_taxid_dfs.copy()
    dfs_by_taxid[reference_taxid] = reference_df

    chunks = []
    for taxid in taxids:
        df = dfs_by_taxid[taxid]
        for i in range(0, len(df), chunk_size):
            df_chunk = df[i:i+chunk_size]
            if isinstance(df_chunk, pd.DataFrame):
                chunks.append((df_chunk, taxid))

    partial_func = partial(generate_unpaired_chunk, gene_cols = gene_cols, reference_taxid = reference_taxid,
                           novel_numbered = novel_numbered, classical_numbered = classical_numbered,
                           compare_classical_method = compare_classical_method)
    pool = multiprocessing.Pool()
    unpaired_dict = {taxid: {} for taxid in taxids}

    with trange(len(chunks), desc=f"\tParsing dataframes into dictionary...") as pbar:
        for result in pool.imap_unordered(partial_func, chunks):
            taxid_chunk_dict, taxid = result
            unpaired_dict[taxid] = unpaired_dict[taxid] | taxid_chunk_dict
            pbar.update()

    pool.close()
    pool.join()

    if isinstance(json_path, str):
        with open(json_path, "w") as json_file:
            json.dump(unpaired_dict, json_file, indent=4)
    elif json_path is not None:
        raise ValueError(f"json_path must be str, but was given as {type(json_path)}")

    return unpaired_dict

def generate_paired(ref_gene_col, reference_taxid, reference_df, target_taxids, target_taxid_dfs,
                    novel_numbered, classical_numbered = None, compare_classical_method = False):
    # Generates paired dictionary

    taxids = target_taxids.copy()
    taxids.insert(0, reference_taxid)
    gene_cols = {reference_taxid: ref_gene_col}
    for target_taxid in target_taxids:
        gene_cols[target_taxid] = f"{target_taxid}_best_homolog_id"

    dfs_by_taxid = target_taxid_dfs.copy()
    dfs_by_taxid[reference_taxid] = reference_df

    ref_taxid_id = insert_ref_taxid(db_path, reference_taxid)
    for target_taxid in target_taxids:
        target_taxid_id = insert_target_taxid(db_path, ref_taxid_id, target_taxid)
        for i, row in reference_df.iterrows():
            ensembl_gene_id = row[ref_gene_col]
            gene_key_id = insert_gene_entry(db_path, target_taxid_id, ensembl_gene_id)

            # Insert novel motifs into a table called novel
            novel_folder_id = insert_folder(db_path, gene_key_id, "novel")
            novel_best_homolog_id = row[f"{target_taxid}_best_homolog_id"]
            if compare_classical_method:
                best_classical_homolog_id = row[f"{target_taxid}_classical_best_homolog_id"]
            else:
                best_classical_homolog_id = None

            for novel_num in novel_numbered:
                val_col_names = [f"{target_taxid}_Host_Match_{novel_num}_motif",
                                 f"{target_taxid}_Host_Match_{novel_num}_motif_topology_type",
                                 f"{target_taxid}_Host_Match_{novel_num}_motif_topology_description",
                                 f"{target_taxid}_Host_Match_{novel_num}_total_motif_score",
                                 f"{target_taxid}_Host_Match_{novel_num}_binding_motif_score",
                                 f"{target_taxid}_Host_Match_{novel_num}_final_call",
                                 f"{target_taxid}_Host_Match_{novel_num}_motif_specificity_score"]

                vals = row[val_col_names].to_list()
                masked_binding_score = vals[4] if vals[5] else 0.0
                vals.insert(5, masked_binding_score)

                classical_score_col = f"{target_taxid}_Host_Match_{novel_num}_classical_score"
                classical_score = row[classical_score_col] if classical_score_col in df.columns else None
                vals.append(classical_score)

                vals.append(novel_best_homolog_id)
                homolog_val_col_names = [f"{target_taxid}_{novel_num}_motif_homolog",
                                         f"{target_taxid}_{novel_num}_motif_homolog_identity",
                                         f"{target_taxid}_{novel_num}_motif_homolog_classification_score",
                                         f"{target_taxid}_{novel_num}_motif_homolog_binding_score",
                                         f"{target_taxid}_{novel_num}_motif_homolog_masked_binding_score",
                                         f"{target_taxid}_{novel_num}_motif_homolog_call"]
                homolog_vals = row[homolog_val_col_names].to_list()
                vals.extend(homolog_vals)

                insert_novel_motif(db_path, novel_folder_id, *vals)

            # Insert classical motifs into a table called classical
            if compare_classical_method:
                classical_folder_id = insert_folder(db_path, gene_key_id, "classical")
                for classical_num in classical_numbered:
                    val_col_names = [f"{classical_num}_motif", f"{classical_num}_motif_topology_type",
                                     f"{classical_num}_motif_topology_description",
                                     f"{classical_num}_total_motif_score"]
                    if taxid != reference_taxid:
                        val_col_names = [f"{taxid}_{col_name}" for col_name in val_col_names]
                    vals = row[val_col_names].to_list()

                    vals.append(best_classical_homolog_id)
                    homolog_val_col_names = [f"{target_taxid}_{classical_num}_motif_homolog",
                                             f"{target_taxid}_{classical_num}_motif_homolog_identity",
                                             f"{target_taxid}_{classical_num}_motif_homolog_score"]
                    homolog_vals = row[homolog_val_col_names].to_list()
                    vals.extend(homolog_vals)

                    insert_classical_motif(db_path, classical_folder_id, *vals)

def convert_to_json(csv_path, predictor_params = predictor_params):
    # Main function for converting dataset to SQL database

    reference_taxid, reference_df, target_taxid_dfs = parse_dfs(csv_path, predictor_params = predictor_params)

    compare_classical_method = predictor_params["compare_classical_method"]
    return_count = predictor_params["return_count"]
    homolog_selection_mode = predictor_params["homology_params"]["homolog_selection_mode"]
    db_path = predictor_params["db_params"]["db_path"]

    # Get substrings marking numbered novel and classical motifs
    nums_with_suffixes = [apply_num_suffix(num) for num in np.arange(1, return_count+1)]
    if compare_classical_method:
        novel_numbered = [f"Novel_{num_with_suffix}" for num_with_suffix in nums_with_suffixes]
        classical_numbered = [f"Classical_{num_with_suffix}" for num_with_suffix in nums_with_suffixes]
    else:
        novel_numbered = nums_with_suffixes
        classical_numbered = []

    # Generate dictionary and dump to json file
    ref_gene_col = predictor_params["homology_params"]["ref_gene_col"]
    target_taxids = set(target_taxid_dfs.keys())
    if homolog_selection_mode == "best":
        '''When best motifs are picked for each comparator species irrespective of alignment with best host motifs, 
        the data is organized without trying to match host motifs to homolog motifs.'''
        unpaired_dict = generate_unpaired(ref_gene_col, reference_taxid, reference_df, target_taxids, target_taxid_dfs,
                                          novel_numbered, classical_numbered, compare_classical_method, db_path)
        return unpaired_dict
    else:
        paired_dict = generate_paired(ref_gene_col, reference_taxid, reference_df, target_taxids, target_taxid_dfs,
                                      novel_numbered, classical_numbered, compare_classical_method, db_path)
        return paired_dict

if __name__ == "__main__":
    csv_path = input("Input the path to the CSV file containing the data:  ")
    convert_to_json(csv_path)