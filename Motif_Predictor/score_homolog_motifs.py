#This script takes protein sequences and computes their motif scores based on the results of make_pairwise_matrices.py

import numpy as np
import pandas as pd
import pickle
import multiprocessing
from tqdm import trange
from functools import partial
from Matrix_Generator.ConditionalMatrix import ConditionalMatrices

# Import the user-specified params, either from a local version or the git-linked version
try:
    from Motif_Predictor.predictor_config_local import predictor_params
except:
    from Motif_Predictor.predictor_config import predictor_params

def score_motifs(seqs_2d, conditional_matrices, weights_tuple = None, standardization_coefficients = None,
                 filters = None, selenocysteine_substitute = "C", gap_substitute = "G"):
    '''
    Vectorized function to score homolog motif seqs based on the dictionary of context-aware weighted matrices

    Args:
        seqs_2d (np.ndarray):                       motif sequences to score
        conditional_matrices (ConditionalMatrices): conditional weighted matrices for scoring peptides
        weights_tuple (tuple):                      (positives_weights, suboptimals_weights, forbiddens_weights)
        standardization_coefficients (tuple)        tuple of coefficients from the model for standardizing score values
        filters (dict):                             dict of position index --> permitted residues
        selenocysteine_substitute (str):            letter to substitute for selenocysteine (U) when U is not in model
        gap_substitute (str):                       the letter to treat gaps ("X") as; default is no side chain, i.e. G

    Returns:
        total_scores (list):                       list of matching scores for each motif
    '''

    if isinstance(selenocysteine_substitute, str):
        seqs_2d[seqs_2d == "U"] = selenocysteine_substitute
    if isinstance(gap_substitute, str):
        seqs_2d[seqs_2d == "X"] = gap_substitute

    motif_length = seqs_2d.shape[1]

    # Get row indices for unique residues
    unique_residues = np.unique(seqs_2d)
    unique_residue_indices = conditional_matrices.index.get_indexer_for(unique_residues)
    if (unique_residue_indices == -1).any():
        failed_residues = unique_residues[unique_residue_indices == -1]
        raise Exception(f"residues not found by matrix indexer: {failed_residues}")

    # Get the matrix row indices for all the residues
    aa_row_indices_2d = np.ones(shape=seqs_2d.shape, dtype=int) * -1
    for unique_residue, row_index in zip(unique_residues, unique_residue_indices):
        aa_row_indices_2d[seqs_2d == unique_residue] = row_index

    # Define residues flanking either side of the residues of interest; for out-of-bounds cases, use opposite side
    flanking_left_2d = np.concatenate((seqs_2d[:, 0:1], seqs_2d[:, 0:-1]), axis=1)
    flanking_right_2d = np.concatenate((seqs_2d[:, 1:], seqs_2d[:, -1:]), axis=1)

    # Get integer-encoded chemical classes for each residue
    left_encoded_classes_2d = np.zeros(flanking_left_2d.shape, dtype=int)
    right_encoded_classes_2d = np.zeros(flanking_right_2d.shape, dtype=int)
    for member_aa, encoded_class in conditional_matrices.encoded_chemical_classes.items():
        left_encoded_classes_2d[flanking_left_2d == member_aa] = encoded_class
        right_encoded_classes_2d[flanking_right_2d == member_aa] = encoded_class
    del flanking_left_2d, flanking_right_2d

    # Find the matrix identifier number (1st dim of 3D matrix) for each encoded class, depending on seq position
    encoded_positions = np.arange(motif_length) * conditional_matrices.chemical_class_count
    left_encoded_matrix_refs = left_encoded_classes_2d + encoded_positions
    right_encoded_matrix_refs = right_encoded_classes_2d + encoded_positions
    del left_encoded_classes_2d, right_encoded_classes_2d, encoded_positions

    # Flatten the encoded matrix refs, which serve as the 1st dimension referring to 3D matrices
    left_encoded_matrix_refs_flattened = left_encoded_matrix_refs.flatten()
    right_encoded_matrix_refs_flattened = right_encoded_matrix_refs.flatten()
    del left_encoded_matrix_refs, right_encoded_matrix_refs

    # Flatten the amino acid row indices into a matching array serving as the 2nd dimension
    aa_row_indices_flattened = aa_row_indices_2d.flatten()
    del aa_row_indices_2d

    # Tile the column indices into a matching array serving as the 3rd dimension
    column_indices = np.arange(motif_length)
    column_indices_tiled = np.tile(column_indices, len(seqs_2d))

    # Define dimensions for 3D matrix indexing
    shape_2d = seqs_2d.shape
    left_dim1 = left_encoded_matrix_refs_flattened
    right_dim1 = right_encoded_matrix_refs_flattened
    dim2 = aa_row_indices_flattened
    dim3 = column_indices_tiled

    # Calculate predicted signal values
    left_positive_2d = conditional_matrices.stacked_positive_weighted[left_dim1, dim2, dim3].reshape(shape_2d)
    right_positive_2d = conditional_matrices.stacked_positive_weighted[right_dim1, dim2, dim3].reshape(shape_2d)
    positive_scores_2d = (left_positive_2d + right_positive_2d) / 2
    del left_positive_2d, right_positive_2d

    # Calculate suboptimal element scores
    left_suboptimal_2d = conditional_matrices.stacked_suboptimal_weighted[left_dim1, dim2, dim3].reshape(shape_2d)
    right_suboptimal_2d = conditional_matrices.stacked_suboptimal_weighted[right_dim1, dim2, dim3].reshape(shape_2d)
    suboptimal_scores_2d = (left_suboptimal_2d + right_suboptimal_2d) / 2
    del left_suboptimal_2d, right_suboptimal_2d

    # Calculate forbidden element scores
    left_forbidden_2d = conditional_matrices.stacked_forbidden_weighted[left_dim1, dim2, dim3].reshape(shape_2d)
    right_forbidden_2d = conditional_matrices.stacked_forbidden_weighted[right_dim1, dim2, dim3].reshape(shape_2d)
    forbidden_scores_2d = (left_forbidden_2d + right_forbidden_2d) / 2
    del left_forbidden_2d, right_forbidden_2d

    # Apply weights if a tuple of arrays of weights values were given
    if weights_tuple is not None:
        positives_weights, suboptimals_weights, forbiddens_weights = weights_tuple
        positive_scores_2d = np.multiply(positive_scores_2d, positives_weights)
        suboptimal_scores_2d = np.multiply(suboptimal_scores_2d, suboptimals_weights)
        forbidden_scores_2d = np.multiply(forbidden_scores_2d, forbiddens_weights)

    # Calculate total scores
    total_scores_2d = positive_scores_2d - suboptimal_scores_2d - forbidden_scores_2d
    del positive_scores_2d, suboptimal_scores_2d, forbidden_scores_2d
    total_scores = total_scores_2d.sum(axis=1)
    del total_scores_2d

    # Standardization of the scores
    if isinstance(standardization_coefficients, tuple) or isinstance(standardization_coefficients, list):
        coefficient_a, coefficient_b = standardization_coefficients
        total_scores = total_scores - coefficient_a
        total_scores = total_scores / coefficient_b

    # Apply filters
    if isinstance(filters, dict):
        for i, allowed_residues in filters.items():
            seqs_pass = np.full(shape=seqs_2d.shape[0], fill_value=False, dtype=bool)
            for allowed_aa in allowed_residues:
                seqs_pass = np.logical_or(seqs_pass, np.char.equal(seqs_2d[:,i], allowed_aa))
            total_scores[~seqs_pass] = 0

    return total_scores

def seqs_chunk_generator(seqs_2d, chunk_size):
    # Chunk generator; saves memory rather than loading list all at once
    for i in range(0, len(seqs_2d), chunk_size):
        yield seqs_2d[i:i+chunk_size]

def score_motifs_parallel(seqs_2d, conditional_matrices, weights_tuple = None, standardization_coefficients = None,
                          chunk_size = 1000, filters = None, selenocysteine_substitute = "C", gap_substitute = "G"):
    '''
    Parallelized function for scoring sequences using a ConditionalMatrices object

        seqs_2d (np.ndarray):                       motif sequences to score
        conditional_matrices (ConditionalMatrices): conditional weighted matrices for scoring peptides
        weights_tuple (tuple):                      (positives_weights, suboptimals_weights, forbiddens_weights)
        standardization_coefficients (tuple)        tuple of coefficients from the model for standardizing score values
        chunk_size (int):                           number of sequences per parallel processing chunk
        filters (dict):                             dict of position index --> permitted residues
        selenocysteine_substitute (str):            letter to substitute for selenocysteine (U) when U is not in model
        gap_substitute (str):                       the letter to treat gaps ("X") as; default is no side chain, i.e. G

    Returns:
        total_scores (list):                        list of matching scores for each motif
    '''

    partial_function = partial(score_motifs, conditional_matrices = conditional_matrices, weights_tuple = weights_tuple,
                               standardization_coefficients = standardization_coefficients, filters = filters,
                               selenocysteine_substitute = selenocysteine_substitute, gap_substitute = gap_substitute)

    chunk_scores = []
    pool = multiprocessing.Pool()

    chunk_count = int(np.ceil(len(seqs_2d) / chunk_size))
    with trange(chunk_count, desc="\t\tScoring current set of motifs...") as pbar:
        for scores in pool.map(partial_function, seqs_chunk_generator(seqs_2d, chunk_size)):
            chunk_scores.append(scores)
            pbar.update()

    pool.close()
    pool.join()

    scores = np.concatenate(chunk_scores)
    del chunk_scores

    return scores

def score_homolog_motifs(data_df, homolog_motif_cols, predictor_params):
    '''
    Main function for scoring homologous motifs

    Args:
        data_df (pd.DataFrame):                     main dataframe with motif sequences for host and homologs
        homolog_motif_cols (list|tuple):            col names where homolog motif sequences are stored
        predictor_params (dict):                    dictionary of parameters for scoring

    Returns:
        data_df (pd.DataFrame):                     dataframe with scores added for homolog motifs
    '''

    standardization_coefficients_path = predictor_params["standardization_coefficients_path"]
    with open(standardization_coefficients_path, "rb") as f:
        standardization_coefficients = pickle.load(f)

    weights_path = predictor_params["pickled_weights_path"]
    with open(weights_path, "rb") as f:
        weights_tuple = pickle.load(f)

    # Load ConditionalMatrices object to be used in scoring
    conditional_matrices_path = predictor_params["conditional_matrices_path"]
    with open(conditional_matrices_path, "rb") as f:
        conditional_matrices = pickle.load(f)

    # Score each column of homolog motifs
    filters = predictor_params["enforced_position_rules"]
    selenocysteine_substitute = predictor_params["selenocysteine_substitute"]
    gap_substitute = predictor_params["gap_substitute"]

    print("Scoring homologous motifs...")
    model_score_cols = []
    homolog_motif_col_count = len(homolog_motif_cols)
    for i, homolog_motif_col in enumerate(homolog_motif_cols):
        print(f"\tScoring homolog motif col ({i+1} of {homolog_motif_col_count}): {homolog_motif_col}")
        motifs = data_df[homolog_motif_col].to_list()

        valid_motifs = []
        valid_motif_indices = []
        for i, motif in enumerate(motifs):
            if isinstance(motif, str):
                if len(motif) > 0:
                    valid_motifs.append(motif)
                    valid_motif_indices.append(i)

        valid_motifs_2d = np.array([list(motif) for motif in valid_motifs])
        del valid_motifs

        valid_scores = score_motifs_parallel(valid_motifs_2d, conditional_matrices, weights_tuple,
                                             standardization_coefficients, filters = filters,
                                             selenocysteine_substitute = selenocysteine_substitute,
                                             gap_substitute = gap_substitute)
        del valid_motifs_2d

        all_scores = np.zeros(shape=len(motifs), dtype=float)
        all_scores[valid_motif_indices] = valid_scores

        model_score_col = homolog_motif_col + "_model_score"
        data_df[model_score_col] = all_scores
        model_score_cols.append(model_score_col)

    # Reorder columns so model scores are beside homolog motif sequences
    print(f"Reordering dataframe...")
    cols = list(data_df.columns)
    for cumulative_displacement, (motif_col, score_col) in enumerate(zip(homolog_motif_cols, model_score_cols)):
        insertion_idx = cols.index(motif_col) + cumulative_displacement + 1
        cols.insert(insertion_idx, score_col)

    data_df = data_df[cols]

    return data_df