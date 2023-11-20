# This is the workflow for predicting and analyzing motifs in proteins based on previously defined matrices.

import numpy as np
import pandas as pd
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

    # Get CSV paths with protein sequences to score
    protein_seqs_path = predictor_params["protein_seqs_path"]
    if isinstance(protein_seqs_path, list):
        protein_seqs_paths = protein_seqs_path
    else:
        protein_seqs_paths = [protein_seqs_path]

    # Also get dataframe chunk sizes for memory management
    df_chunk_counts = predictor_params["df_chunks"]

    for path, chunk_count in zip(protein_seqs_paths, df_chunk_counts):
        # Get row count for the whole spreadsheet
        with open(path, "r", encoding="utf-8") as file:
            row_count = sum(1 for row in file) - 1

        # Load dataframe in a memory-efficient manner
        chunk_size = np.ceil(row_count / chunk_count)
        chunk_dfs = []
        for i, chunk_df in enumerate(pd.read_csv(path, chunksize=chunk_size)):
            print(f"Processing chunk #{i+1} of {path}...")
            # Apply conditional matrices motif scoring
            results = score_proteins(chunk_df, predictor_params)
            chunk_df, novel_motif_cols, novel_score_cols, classical_motif_cols, classical_score_cols = results
            all_motif_cols = novel_motif_cols.copy()
            all_motif_cols.extend(classical_motif_cols)

            # Apply bait specificity scoring of discovered motifs
            chunk_df = apply_specificity_scores(chunk_df, all_motif_cols, predictor_params)

            # Get homolog seq col names
            homolog_seq_cols = []
            for col in chunk_df.columns:
                if "homolog" in col and "seq" in col:
                    homolog_seq_cols.append(col)

            # Separate dataframe by whether entries have any homologs and motifs to score
            contains_homolog = np.full(shape=len(chunk_df), fill_value=False, dtype=bool)
            for homolog_seq_col in homolog_seq_cols:
                col_contains_homolog = chunk_df[homolog_seq_col].notna()
                contains_homolog = np.logical_or(contains_homolog, col_contains_homolog.to_numpy(dtype=bool))
            contains_predicted_motif = np.full(shape=len(chunk_df), fill_value=False, dtype=bool)
            for motif_col in all_motif_cols:
                col_contains_motif = chunk_df[motif_col].notna()
                contains_predicted_motif = np.logical_or(contains_predicted_motif, col_contains_motif.to_numpy(dtype=bool))
            contains_homolog_and_motif = np.logical_and(contains_homolog, contains_predicted_motif)

            df_with_homologs = chunk_df.loc[contains_homolog_and_motif]
            df_without_homologs = chunk_df.loc[~contains_homolog_and_motif]
            del chunk_df # temporary; will be reconstructed

            # Drop homolog seq cols from df_without_homologs, since this will be done to df_with_homologs later
            for homolog_seq_col in homolog_seq_cols:
                df_without_homologs.drop(homolog_seq_col, axis=1)

            # Evaluate motif homology
            df_with_homologs, homolog_motif_cols = evaluate_homologs(df_with_homologs, all_motif_cols, homolog_seq_cols)

            # Score homologous motifs
            df_with_homologs = score_homolog_motifs(df_with_homologs, homolog_motif_cols, predictor_params)

            # Apply bait specificity scoring to homologous motifs
            df_with_homologs = apply_specificity_scores(df_with_homologs, homolog_motif_cols, predictor_params)

            # Recombine dataframes
            df_with_homologs = df_with_homologs.reset_index(drop=True)
            df_without_homologs = df_without_homologs.reset_index(drop=True)

            chunk_df = pd.concat([df_with_homologs, df_without_homologs], ignore_index=True)
            del df_with_homologs, df_without_homologs

            # Get topology for predicted motifs
            chunk_df = predict_topology(chunk_df, all_motif_cols, predictor_params)

            chunk_dfs.append(chunk_df)

        protein_seqs_df = pd.concat(chunk_dfs, axis=0)

        # Save scored data
        output_path = path[:-4] + "_scored.csv"
        protein_seqs_df.to_csv(output_path)
        print(f"Saved scored motifs to {output_path}")
        del protein_seqs_df, chunk_dfs

if __name__ == "__main__":
    main()