# This is the workflow for predicting and analyzing motifs in proteins based on previously defined matrices.

import numpy as np
import pandas as pd
import os
import pickle
import warnings
from Motif_Predictor.score_protein_motifs import score_proteins, parse_ensembl_tm
from Motif_Predictor.specificity_score_assigner import apply_specificity_scores
from Motif_Predictor.check_conservation import evaluate_homologs
from Motif_Predictor.score_homolog_motifs import score_homolog_motifs
from Motif_Predictor.motif_topology_predictor import predict_topology
from Motif_Predictor.combine_dfs import fuse_dfs, infer_cytosolic_accessibility, make_gene_df
from Motif_Predictor.load_predictor_config import load_config
from Motif_Predictor.make_json import convert_to_json

predictor_params = load_config(verbose=True)

def process_chunks(protein_seqs_paths, keys, df_chunk_counts, topological_domains, sequences,
                   predictor_params = predictor_params):
    '''
    Helper function that processes dataframes in chunks; only used when homology_selection_mode is not "best"

    Args:
        protein_seqs_paths (list|tuple): list of paths for retrieving dataframes
        keys (list):                     corresponding keys for paths
        df_chunk_counts (list|tuple):    corresponding list of number of chunks to split each dataframe into
        topological_domains (dict):      dict of accession --> topological features list
        sequences (dict):                dict of accession --> sequence
        predictor_params (dict):         dict of user_defined parameters

    Returns:
        output_paths (list):             list of paths to processed data
    '''

    seq_col = predictor_params["seq_col"]
    output_paths = []
    for key, path, chunk_count in zip(keys, protein_seqs_paths, df_chunk_counts):
        current_taxid = int(key.split("_vs_")[0]) if "_vs_" in key else int(key)

        # Get row count for the whole spreadsheet
        with open(path, "r", encoding="utf-8") as file:
            row_count = sum(1 for row in file) - 1

        # Load dataframe in a memory-efficient manner
        chunk_size = np.ceil(row_count / chunk_count)
        cache_paths = []
        for i, chunk_df in enumerate(pd.read_csv(path, chunksize=chunk_size)):
            print(f"Processing chunk #{i+1} of {path}...")

            # Apply conditional matrices motif scoring
            results = score_proteins(chunk_df, predictor_params, current_taxid = current_taxid)

            chunk_df = results[0]
            valid_seqs_exist = results[1]
            novel_motif_cols = results[2]
            classical_motif_cols = results[10]

            all_motif_cols = novel_motif_cols.copy()
            all_motif_cols.extend(classical_motif_cols)

            # Apply bait specificity scoring of discovered motifs
            assign_specificities = predictor_params["assign_specificity_scores"]
            if assign_specificities and valid_seqs_exist:
                chunk_df = apply_specificity_scores(chunk_df, all_motif_cols, predictor_params)
            elif assign_specificities:
                # Insert NaN for specificity score columns when no valid sequences exist
                for motif_col in all_motif_cols:
                    motif_col_idx = chunk_df.columns.get_loc(motif_col)
                    specificity_col = f"{motif_col}_specificity_score"
                    chunk_df.insert(motif_col_idx+1, specificity_col, np.nan)

            if not any(["_specificity_score" in col for col in chunk_df.columns]):
                raise Exception(f"Specificity scores were not applied to chunk {i+1} of {path}")

            # Get topology for predicted motifs
            chunk_df = predict_topology(chunk_df, all_motif_cols, predictor_params, topological_domains, sequences)

            # Get homolog seq col names
            homolog_id_cols = [] # not currently used, but leaving it here for future use
            homolog_seq_cols = []
            for col in chunk_df.columns:
                if "homolog" in col and "seq" in col:
                    homolog_seq_cols.append(col)
                elif "homolog" in col and "seq" not in col:
                    homolog_id_cols.append(col)

            # Evaluate motif homology
            homology_params = predictor_params.get("homology_params")
            if isinstance(homology_params, dict):
                similarity_weights = homology_params.get("similarity_position_weights")
            else:
                similarity_weights = None

            replace_selenocysteine = predictor_params["replace_selenocysteine"]
            selenocysteine_substitute = predictor_params.get("selenocysteine_substitute")
            homology_results = evaluate_homologs(chunk_df, all_motif_cols, homolog_seq_cols, similarity_weights,
                                                 replace_selenocysteine, selenocysteine_substitute)
            chunk_df, homolog_motif_cols, homolog_motif_col_groups = homology_results

            # Score homologous motifs
            chunk_df, homolog_motif_cols = score_homolog_motifs(chunk_df, homolog_motif_cols,
                                                                homolog_motif_col_groups, predictor_params)

            # Apply bait specificity scoring to homologous motifs
            if assign_specificities:
                chunk_df = apply_specificity_scores(chunk_df, homolog_motif_cols, predictor_params)

            # Dump current data to save memory; will be concatenated later
            print("\tCaching current chunk to save memory...")

            pd.options.mode.chained_assignment = None  # suppress SettingWithCopyWarning, which has no effect
            chunk_df.drop(seq_col, axis=1, inplace=True)
            pd.options.mode.chained_assignment = "warn" #restore to default

            temp_path = os.path.join(os.getcwd(), f"temp_df_dump_{i}.csv")
            chunk_df.to_csv(temp_path)
            cache_paths.append(temp_path)
            del chunk_df

        print("Concatenating cached dataframes...")
        cache_dfs = []
        for cache_path in cache_paths:
            df = pd.read_csv(cache_path)
            cache_dfs.append(df)
        protein_seqs_df = pd.concat(cache_dfs, ignore_index=True)

        # Save scored data
        output_path = path[:-4] + "_scored.csv"
        protein_seqs_df.to_csv(output_path)
        print(f"Saved scored motifs to {output_path}")
        output_paths.append(output_path)
        del protein_seqs_df

        # Delete temporary files
        print(f"Deleting temporary files...")
        for cache_path in cache_paths:
            os.remove(cache_path)
        print(f"Done!")

    return output_paths

def process_existing(protein_seqs_paths, keys, df_chunk_counts, topological_domains, sequences,
                     predictor_params = predictor_params, debug_caching = True):
    '''
    Helper function that processes dataframes in chunks; only used when homology_selection_mode is not "best"

    Args:
        protein_seqs_paths (list|tuple): list of paths for retrieving dataframes
        keys (list):                     corresponding keys for paths
        df_chunk_counts (list|tuple):    corresponding list of number of chunks to split each dataframe into
        topological_domains (dict):      dict of accession --> topological features list
        sequences (dict):                dict of accession --> sequence
        predictor_params (dict):         dict of user_defined parameters
        debug_caching (bool):            if set to True, pickles an interim copy of data_dfs so that it doesn't need to
                                         be repeatedly generated during debugging

    Returns:
        output_paths (list):             list of paths to processed data
    '''

    seq_col = predictor_params["seq_col"]

    # Generate ensembl_tm_dict upfront
    topo_params = predictor_params.get("topo_params")
    if isinstance(topo_params, dict):
        filter_transmembrane_helices = topo_params["filter_transmembrane_helices"]
        ensembl_tm_path = topo_params["ensembl_tm_path"]
        ensembl_tm_dict = parse_ensembl_tm(ensembl_tm_path)
    else:
        filter_transmembrane_helices = False
        ensembl_tm_path, ensembl_tm_dict = None, None

    # Perform the main scoring
    pickling_path = os.path.join(os.getcwd(), "data_dfs.pkl")
    if not debug_caching or not os.path.exists(pickling_path):
        data_dfs = []
        taxid_dfs = {}
        output_paths = []
        for key, path, chunk_count in zip(keys, protein_seqs_paths, df_chunk_counts):
            print(f"Key: {key} | Scoring path: {path}")
            current_taxid = int(key.split("_vs_")[0]) if "_vs_" in str(key) else int(key)

            # Get row count for the whole spreadsheet
            with open(path, "r", encoding="utf-8") as file:
                row_count = sum(1 for row in file) - 1

            # Load dataframe
            chunk_size = np.ceil(row_count / chunk_count)
            cache_paths = []
            for i, chunk_df in enumerate(pd.read_csv(path, chunksize=chunk_size)):
                print(f"\tProcessing chunk #{i+1}...")

                # Apply conditional matrices motif scoring
                results = score_proteins(chunk_df, predictor_params, ensembl_tm_dict, filter_transmembrane_helices,
                                         current_taxid)
                chunk_df, valid_seqs_exist, novel_motif_cols = results[:3]
                classical_motif_cols = results[10]
                all_motif_cols = novel_motif_cols + classical_motif_cols

                # Apply bait specificity scoring of discovered motifs
                assign_specificities = predictor_params["assign_specificity_scores"]
                if assign_specificities:
                    chunk_df = apply_specificity_scores(chunk_df, all_motif_cols, predictor_params)
                valid_cols = [col for col in chunk_df.columns if isinstance(col, str)]
                if len(valid_cols) > 0:
                    if not any(["_specificity_score" in col for col in valid_cols]):
                        warnings.warn(f"\tSpecificity scores were not applied to chunk {i+1} of {path}")

                # Get topology for predicted motifs
                chunk_df = predict_topology(chunk_df, all_motif_cols, predictor_params, topological_domains, sequences)

                # Delete forbidden secondary structure column
                if "forbidden_secondary_structure" in chunk_df.columns:
                    chunk_df.drop("forbidden_secondary_structure", axis=1, inplace=True)

                # Drop sequence column, which is no longer needed, as motifs have already been extracted
                pd.options.mode.chained_assignment = None  # suppress SettingWithCopyWarning, which has no effect
                chunk_df.drop(seq_col, axis=1, inplace=True)
                pd.options.mode.chained_assignment = "warn" # restore to default

                # Dump current data to save memory; will be concatenated later
                print("\tCaching current chunk to save memory...") if chunk_count > 1 else None
                temp_path = os.path.join(os.getcwd(), f"temp_df_dump_{i}.csv")
                chunk_df.to_csv(temp_path)
                cache_paths.append(temp_path)
                del chunk_df

            print("\tConcatenating cached dataframes...") if chunk_count > 1 else None
            cache_dfs = []
            for cache_path in cache_paths:
                df = pd.read_csv(cache_path)
                cache_dfs.append(df)
            data_df = pd.concat(cache_dfs, ignore_index=True)

            # Add cytosolic accessibility column
            print("\tAdding cytosolic accessibility column...")
            motif_prefixes = [col.split("_motif_topology_type")[0] for col in data_df.columns if "topology_type" in col]
            data_df = infer_cytosolic_accessibility(data_df, motif_prefixes)

            # Check if specificities were calculated
            valid_cols = [col for col in data_df.columns if isinstance(col, str)]
            if len(valid_cols) > 0:
                if not any(["_specificity_score" in col for col in valid_cols]):
                    warnings.warn(f"Specificity scores were not applied to dataframe for {path}")

            # Save scored data
            output_path = path[:-4] + "_scored.csv"
            data_df.to_csv(output_path)
            print(f"\tSaved scored motifs to {output_path}")
            output_paths.append(output_path)
            data_dfs.append(data_df)
            taxid_dfs[current_taxid] = data_df

            # Delete temporary files
            print(f"\tDeleting temporary files...")
            for cache_path in cache_paths:
                os.remove(cache_path)
            print(f"\tDone!")

        if debug_caching:
            with open(pickling_path, "wb") as f:
                pickle.dump((data_dfs, taxid_dfs, output_paths), f)

    else:
        with open(pickling_path, "rb") as f:
            data_dfs, taxid_dfs, output_paths = pickle.load(f)

    data_dict, correlated_dict, correlated_df = convert_to_json(taxid_dfs, predictor_params, correlate_homology=True,
                                                                return_data_dict=False, return_correlated_dict=False,
                                                                return_correlated_df=False)

    return output_paths, correlated_df

cwd = os.getcwd()
def main(predictor_params = predictor_params):
    '''
    Main function that integrates conditional matrices scoring and specificity scoring of discovered motifs

    Args:
        predictor_params (dict): dict of user_defined parameters

    Returns:
        protein_seqs_df (pd.DataFrame): dataframe of scored protein sequences
    '''

    # Optionally pre-parse topological domains from Uniprot instead of doing so for each chunk
    topological_domains, sequences = None, None
    if isinstance(predictor_params.get("topo_params"), dict):
        parse_topologies_upfront = predictor_params["topo_params"].get("parse_topologies_upfront")
        if parse_topologies_upfront:
            print("Parsing topologies upfront from Uniprot...")
            from uniparser import get_topological_domains
            uniprot_path = predictor_params["topo_params"]["uniprot_path"]
            base_uniprot_path = os.path.basename(uniprot_path)
            pickled_uniprot_path = base_uniprot_path.rsplit(".", 1)[0] + "_parsed.pkl"
            pickled_uniprot_path = os.path.join(cwd, pickled_uniprot_path)
            if os.path.exists(pickled_uniprot_path):
                print("\tLoading from pickled version...")
                with open(pickled_uniprot_path, "rb") as f:
                    topological_domains, sequences = pickle.load(f)
            else:
                topological_domains, sequences = get_topological_domains(path = uniprot_path)
                print("\tPickling for later re-use...")
                with open(pickled_uniprot_path, "wb") as f:
                    pickle.dump((topological_domains, sequences), f)

    # Get CSV paths with protein sequences to score
    protein_seqs_paths, df_chunk_counts = [], []
    keys = list(predictor_params["protein_seqs_paths"].keys())
    for key in keys:
        protein_seqs_paths.append(predictor_params["protein_seqs_paths"][key])
        df_chunk_count = predictor_params["df_chunks"].get(key)
        if df_chunk_count is None:
            raise Exception(f"df_chunks key mismatch: key \"{key}\" not found")
        df_chunk_counts.append(df_chunk_count)

    # Homologous motif detection is done differently if each dataframe contains data for only one species
    homolog_selection_mode = predictor_params["homology_params"]["homolog_selection_mode"]
    if homolog_selection_mode == "best":
        process_existing(protein_seqs_paths, keys, df_chunk_counts, topological_domains, sequences, predictor_params)
    else:
        output_paths = process_chunks(protein_seqs_paths, keys, df_chunk_counts, topological_domains, sequences,
                                      predictor_params)
        print(f"Combining dataframes for homolog species into one simplified dataframe...")
        combined_df = fuse_dfs(output_paths)
        taxids = [key.split("_vs_")[1] for key in keys]

        # Save combined_df
        parent_path = output_paths[0].rsplit("/",1)[0]
        combined_path = os.path.join(parent_path, "proteome_datasets_combined_scored.csv")
        combined_df.to_csv(combined_path)

        print(f"Generating dataframe with only one row per gene ID...")
        unique_df = make_gene_df(combined_df, homolog_selection_mode, taxids)
        unique_path = os.path.join(parent_path, "proteome_datasets_combined_scored_by_gene.csv")
        unique_df.to_csv(unique_path)

    print(f"Process completed!")

if __name__ == "__main__":
    main()