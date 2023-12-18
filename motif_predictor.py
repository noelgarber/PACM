# This is the workflow for predicting and analyzing motifs in proteins based on previously defined matrices.

import numpy as np
import pandas as pd
import os
import psutil
from Motif_Predictor.score_protein_motifs import score_proteins
from Motif_Predictor.specificity_score_assigner import apply_specificity_scores
from Motif_Predictor.check_conservation import evaluate_homologs
from Motif_Predictor.score_homolog_motifs import score_homolog_motifs
from Motif_Predictor.motif_topology_predictor import predict_topology
try:
    from Motif_Predictor.predictor_config_local import predictor_params
except:
    from Motif_Predictor.predictor_config import predictor_params

def main(predictor_params = predictor_params):
    '''
    Main function that integrates conditional matrices scoring and specificity scoring of discovered motifs

    Args:
        predictor_params (dict): dict of user_defined parameters

    Returns:
        protein_seqs_df (pd.DataFrame): dataframe of scored protein sequences
    '''

    # Optionally pre-parse topological domains from Uniprot instead of doing so for each chunk
    parse_topologies_upfront = predictor_params.get("parse_topologies_upfront")
    if parse_topologies_upfront:
        print("Parsing topologies upfront from Uniprot...")
        from assemble_data.parse_uniprot_topology import get_topological_domains
        uniprot_path = predictor_params["uniprot_path"]
        topological_domains, sequences = get_topological_domains(path = uniprot_path)
    else:
        topological_domains, sequences = None, None

    # Get CSV paths with protein sequences to score
    protein_seqs_path = predictor_params["protein_seqs_path"]
    if isinstance(protein_seqs_path, list):
        protein_seqs_paths = protein_seqs_path
    else:
        protein_seqs_paths = [protein_seqs_path]

    # Also get dataframe chunk sizes for memory management
    df_chunk_counts = predictor_params["df_chunks"]
    seq_col = predictor_params["seq_col"]

    for path, chunk_count in zip(protein_seqs_paths, df_chunk_counts):
        # Get row count for the whole spreadsheet
        with open(path, "r", encoding="utf-8") as file:
            row_count = sum(1 for row in file) - 1

        # Load dataframe in a memory-efficient manner
        chunk_size = np.ceil(row_count / chunk_count)
        cache_paths = []
        for i, chunk_df in enumerate(pd.read_csv(path, chunksize=chunk_size)):
            print(f"Processing chunk #{i+1} of {path}...")

            # Apply conditional matrices motif scoring
            print(f"\tAssigning motif scores...")
            results = score_proteins(chunk_df, predictor_params)
            chunk_df, novel_motif_cols, novel_score_cols, classical_motif_cols, classical_score_cols = results
            all_motif_cols = novel_motif_cols.copy()
            all_motif_cols.extend(classical_motif_cols)

            # Apply bait specificity scoring of discovered motifs
            assign_specificities = predictor_params["assign_specificity_scores"]
            if assign_specificities:
                print(f"\tAssigning specificity scores...")
                chunk_df = apply_specificity_scores(chunk_df, all_motif_cols, predictor_params)

            # Get topology for predicted motifs
            print("\tGetting motif topologies...")
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
            homology_results = evaluate_homologs(chunk_df, all_motif_cols, homolog_seq_cols)
            chunk_df, homolog_motif_cols, homolog_motif_col_groups = homology_results

            # Score homologous motifs
            print("\tScoring homologous motifs...")
            chunk_df, _, homolog_motif_cols, _ = score_homolog_motifs(chunk_df, homolog_motif_cols,
                                                                      homolog_motif_col_groups, predictor_params)

            # Apply bait specificity scoring to homologous motifs
            if assign_specificities:
                print("\tApplying specificity scores to homologous motifs...")
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
        del protein_seqs_df

        # Delete temporary files
        print(f"Deleting temporary files...")
        for cache_path in cache_paths:
            os.remove(cache_path)
        print(f"Done!")

if __name__ == "__main__":
    main()