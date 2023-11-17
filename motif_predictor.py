# This is the workflow for predicting and analyzing motifs in proteins based on previously defined matrices.

import numpy as np
import pandas as pd
import sys
from Motif_Predictor.score_protein_motifs import score_proteins
from Motif_Predictor.specificity_score_assigner import apply_specificity_scores
from Motif_Predictor.check_conservation import evaluate_homologs
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

    # Get protein sequences to score
    protein_seqs_path = predictor_params["protein_seqs_path"]
    if isinstance(protein_seqs_path, list):
        protein_seqs_paths = protein_seqs_path
    else:
        protein_seqs_paths = [protein_seqs_path]

    for path in protein_seqs_paths:
        protein_seqs_df = pd.read_csv(path)

        # Display size of dataframe
        filename = path.rsplit("/",1)[1]
        df_size = sys.getsizeof(protein_seqs_df)
        if df_size < 1000:
            df_size_str = f"{df_size} bytes"
        elif df_size < 1000000:
            df_size = round(df_size / 1000, 1)
            df_size_str = f"{df_size} KB"
        elif df_size < 1000000000:
            df_size = round(df_size / 1000000, 1)
            df_size_str = f"{df_size} MB"
        else:
            df_size = round(df_size / 1000000000, 1)
            df_size_str = f"{df_size} GB"
        print(f"Processing {filename} as dataframe (size = {df_size_str})")

        # Apply conditional matrices motif scoring
        results = score_proteins(protein_seqs_df, predictor_params)
        protein_seqs_df, novel_motif_cols, novel_score_cols, classical_motif_cols, classical_score_cols = results
        all_motif_cols = novel_motif_cols.copy()
        all_motif_cols.extend(classical_motif_cols)

        # Apply bait specificity scoring of discovered motifs
        protein_seqs_df = apply_specificity_scores(protein_seqs_df, all_motif_cols, predictor_params)

        # Get homolog seq col names
        homolog_seq_cols = []
        for col in protein_seqs_df.columns:
            if "homolog" in col and "seq" in col:
                homolog_seq_cols.append(col)

        # Separate dataframe by whether entries have any homologs and motifs to score
        contains_homolog = np.full(shape=len(protein_seqs_df), fill_value=False, dtype=bool)
        for homolog_seq_col in homolog_seq_cols:
            col_contains_homolog = protein_seqs_df[homolog_seq_col].notna()
            contains_homolog = np.logical_or(contains_homolog, col_contains_homolog.to_numpy(dtype=bool))
        contains_predicted_motif = np.full(shape=len(protein_seqs_df), fill_value=False, dtype=bool)
        for motif_col in all_motif_cols:
            col_contains_motif = protein_seqs_df[motif_col].notna()
            contains_predicted_motif = np.logical_or(contains_predicted_motif, col_contains_motif.to_numpy(dtype=bool))
        contains_homolog_and_motif = np.logical_and(contains_homolog, contains_predicted_motif)

        df_with_homologs = protein_seqs_df.loc[contains_homolog_and_motif]
        df_without_homologs = protein_seqs_df.loc[~contains_homolog_and_motif]
        del protein_seqs_df

        # Evaluate motif homology
        df_with_homologs = evaluate_homologs(df_with_homologs, all_motif_cols, homolog_seq_cols)

        # Recombine dataframes
        for homolog_seq_col in homolog_seq_cols:
            df_without_homologs.drop(homolog_seq_col, axis=1)
        protein_seqs_df = pd.concat([df_with_homologs, df_without_homologs], axis=0)
        del df_with_homologs, df_without_homologs

        # Save scored data
        output_path = path[:-4] + "_scored.csv"
        protein_seqs_df.to_csv(output_path)
        print(f"Saved scored motifs to {output_path}")
        del protein_seqs_df

        # Get topology for predicted motifs
        '''
        protein_seqs_df = predict_topology(protein_seqs_df, all_motif_cols, predictor_params)
        topology_path = predictor_params["protein_seqs_path"][:-4] + "_with_Topology.csv"
        print(f"Saved scored motifs with topology information to {topology_path}")
        '''

if __name__ == "__main__":
    main()