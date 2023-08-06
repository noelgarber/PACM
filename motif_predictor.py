# This is the workflow for predicting and analyzing motifs in proteins based on previously defined matrices.

import numpy as np
import pandas as pd
from Motif_Predictor.score_protein_motifs import score_protein_seqs
from Motif_Predictor.specificity_score_assigner import apply_specificity_scores
from Motif_Predictor.motif_topology_predictor import predict_topology
from Motif_Predictor.predictor_config import predictor_params

def main(predictor_params = predictor_params):
    '''
    Main function that integrates conditional matrices scoring and specificity scoring of discovered motifs

    Args:
        predictor_params (dict): dict of user_defined parameters

    Returns:
        protein_seqs_df (pd.DataFrame): dataframe of scored protein sequences
    '''

    # Apply conditional matrices motif scoring
    protein_seqs_df, motif_col_names = score_protein_seqs(predictor_params)

    # Apply bait specificity scoring of discovered motifs
    compare_classical_method = predictor_params["compare_classical_method"]
    if compare_classical_method:
        motif_seq_cols = []
        for motif_col in motif_col_names:
            motif_seq_cols.append("Novel_" + motif_col)
            motif_seq_cols.append("Classical_" + motif_col)
    else:
        motif_seq_cols = motif_col_names
    protein_seqs_df = apply_specificity_scores(protein_seqs_df, motif_seq_cols, predictor_params)

    # Save scored data
    output_path = predictor_params["scored_output_path"]
    protein_seqs_df.to_csv(output_path)
    print(f"Saved scored motifs to {output_path}")

    # Get topology for predicted motifs
    protein_seqs_df = predict_topology(protein_seqs_df, motif_seq_cols, predictor_params)

    return protein_seqs_df

if __name__ == "__main__":
    main()