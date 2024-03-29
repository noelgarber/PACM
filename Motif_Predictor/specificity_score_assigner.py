# This script assigns motif specificity scores to detected motifs from protein sequences

import numpy as np
import pandas as pd
import pickle
import multiprocessing
from tqdm import trange
from functools import partial
from Motif_Predictor.predictor_config import predictor_params

def evaluate_chunk(chunk_tuple, specificity_matrix, use_specificity_weighted):
    '''
    Simple low-level scoring function that can be easily parallelized

    Args:
        chunk_tuple (tuple):                    tuple of (chunk_seqs_2d, specificity_score_col)
        specificity_matrix (SpecificityMatrix): specificity matrix object to use for scoring
        use_specificity_weighted (bool):        whether to use the weighted matrix; if false, uses unweighted

    Returns:
        results_tuple (tuple):                  tuple of (chunk_scores, specificity_score_col)
    '''

    if chunk_tuple is None:
        return (None, None)

    chunk_seqs_2d, specificity_score_col = chunk_tuple
    chunk_specificity_scores = specificity_matrix.score_motifs(chunk_seqs_2d, use_specificity_weighted)

    return (chunk_specificity_scores, specificity_score_col)

def seq_chunk_generator(protein_seqs_df, motif_cols, chunk_size = 1000):
    '''
    Generator for tuples of (seq_chunk, motif_col_idx, specificity_score_col)

    Args:
        protein_seqs_df (pd.DataFrame):  main dataframe to take data out of
        motif_cols (list):               motif sequence col names
        chunk_size (int):                size of each yielded chunk

    Yields:
        yielded_tuple (tuple):          (valid_motifs_2d[i:i+chunk_size],
                                         valid_mask[i:i+chunk_size],
                                         specificity_score_col)
    '''

    for motif_col in motif_cols:
        specificity_score_col = motif_col + "_specificity_score"

        # Score the non-blank motif sequences
        motif_seqs = protein_seqs_df[motif_col].to_numpy()
        not_nan = protein_seqs_df[motif_col].notna().to_numpy()
        not_blank = protein_seqs_df[motif_col].ne("").to_numpy()
        valid_mask = np.logical_and(not_nan, not_blank)
        valid_motifs = motif_seqs[valid_mask]
        valid_motifs_2d = np.array([list(motif) for motif in valid_motifs])

        if len(valid_motifs_2d) > 0 and valid_motifs_2d.ndim == 2:
            for i in range(0, len(valid_motifs_2d), chunk_size):
                yield (valid_motifs_2d[i:i+chunk_size], specificity_score_col)
        else:
            yield None

def apply_specificity_scores(protein_seqs_df, motif_cols, predictor_params=predictor_params):
    '''
    Main function for applying specificity scores to identified motif sequences

    Args:
        protein_seqs_df (pd.DataFrame): dataframe of protein sequences scored with conditional matrices
        motif_cols (list): 				list of columns containing motifs found by score_protein_motifs.py
        predictor_params (dict): 		dict of user-defined parameters for the predictive pipeline

    Returns:
        protein_seqs_df (pd.DataFrame): dataframe with added columns for specificity scores
    '''

    # Load the SpecificityMatrix object generated by matrix_generator.py
    specificity_matrix_path = predictor_params["specificity_matrix_path"]
    with open(specificity_matrix_path, "rb") as f:
        specificity_matrix = pickle.load(f)

    # Set up elements for parallel processing
    use_specificity_weighted = predictor_params["use_specificity_weighted"]

    partial_evaluator = partial(evaluate_chunk, specificity_matrix=specificity_matrix,
                                use_specificity_weighted=use_specificity_weighted)

    results = {}

    # Parallel processing of specificity score calculation
    chunk_size = 1000
    valid_chunk_count = 0
    for motif_col in motif_cols:
        valids = np.logical_and(protein_seqs_df[motif_col].notna().to_numpy(),
                                protein_seqs_df[motif_col].ne("").to_numpy())
        valid_chunk_count += int(np.ceil(valids.sum() / chunk_size))

    if "homolog" in motif_cols[0]:
        description = "\tScoring homologous motif specificities..."
    else:
        description  = "\tScoring motif specificities..."

    with trange(valid_chunk_count+1, desc=description) as pbar:
        pool = multiprocessing.Pool()

        for chunk_results in pool.map(partial_evaluator, seq_chunk_generator(protein_seqs_df, motif_cols, chunk_size)):
            chunk_valid_scores, specificity_score_col = chunk_results

            if chunk_valid_scores is None:
                results[specificity_score_col] = None
            elif results.get(specificity_score_col) is None:
                results[specificity_score_col] = chunk_valid_scores
            else:
                previous_scores = results[specificity_score_col]
                results[specificity_score_col] = np.concatenate([previous_scores, chunk_valid_scores])

            pbar.update()

        pool.close()
        pool.join()

        # Parse and reorder the results; assumes row order is the same as the input dataframe
        for motif_col, specificity_score_col, valid_specificity_scores in zip(motif_cols, results.keys(), results.values()):
            if valid_specificity_scores is not None:
                # Get mask for reapplying valid scores to whole column
                not_nan = protein_seqs_df[motif_col].notna().to_numpy()
                not_blank = protein_seqs_df[motif_col].ne("").to_numpy()
                valid_mask = np.logical_and(not_nan, not_blank)

                # Apply valid scores onto an expanded column using the mask
                specificity_scores = np.full(shape=len(protein_seqs_df), fill_value=np.nan, dtype=float)
                specificity_scores[valid_mask] = valid_specificity_scores

                motif_col_idx = protein_seqs_df.columns.get_loc(motif_col)
                if "homolog" in motif_col:
                    specificity_insert_idx = motif_col_idx + 9
                else:
                    specificity_insert_idx = motif_col_idx + 2 if "Classical" in motif_col else motif_col_idx + 7

                try:
                    protein_seqs_df.insert(specificity_insert_idx, specificity_score_col, specificity_scores)
                except Exception as e:
                    raise e

        pbar.update()

    return protein_seqs_df