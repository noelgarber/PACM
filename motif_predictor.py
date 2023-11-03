# This is the workflow for predicting and analyzing motifs in proteins based on previously defined matrices.

import numpy as np
import pandas as pd
from Motif_Predictor.score_protein_motifs import score_proteins
from Motif_Predictor.specificity_score_assigner import apply_specificity_scores
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
    protein_seqs_df = pd.read_csv(protein_seqs_path)

    # Apply conditional matrices motif scoring
    results = score_proteins(protein_seqs_df, predictor_params)
    protein_seqs_df, novel_motif_cols, novel_score_cols, classical_motif_cols, classical_score_cols = results
    all_motif_cols = novel_motif_cols.copy()
    all_motif_cols.extend(classical_motif_cols)

    # Apply bait specificity scoring of discovered motifs
    protein_seqs_df = apply_specificity_scores(protein_seqs_df, all_motif_cols, predictor_params)

    # Save scored data
    output_path = predictor_params["scored_output_path"]
    protein_seqs_df.to_csv(output_path)
    print(f"Saved scored motifs to {output_path}")

    # Get topology for predicted motifs
    protein_seqs_df = predict_topology(protein_seqs_df, motif_seq_cols, predictor_params)
    topology_path = predictor_params["protein_seqs_path"][:-4] + "_with_Topology.csv"
    print(f"Saved scored motifs with topology information to {topology_path}")

    return protein_seqs_df

if __name__ == "__main__":
    main()