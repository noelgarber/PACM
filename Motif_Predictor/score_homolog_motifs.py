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
                          chunk_size = 10000, filters = None, selenocysteine_substitute = "C", gap_substitute = "G"):
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

    for scores in pool.map(partial_function, seqs_chunk_generator(seqs_2d, chunk_size)):
        chunk_scores.append(scores)

    pool.close()
    pool.join()

    scores = np.concatenate(chunk_scores)
    del chunk_scores

    return scores

def collapse_to_best(data_df, homolog_motif_cols):
    '''
    Collapses homolog motif cols to top two matching homologs according to model score

    Args:
        data_df (pd.DataFrame):     main dataframe with motif and homolog data
        homolog_motif_cols (list):  list of col names containing homologous motifs

    Returns:
        data_df (pd.DataFrame):     dataframe with collapsed homologous motifs
        homolog_id_cols (list):     shortened list of col names where homolog ids are stored
        homolog_motif_cols (list):  shortened list of col names containing homologous motifs
        model_score_cols (list):    shortened list of col names containing homologous motif scores according to model
    '''

    # Parse columns
    motif_col_dict = {}

    for homolog_motif_col in homolog_motif_cols:
        vs_split_elements = homolog_motif_col.split("_vs_")
        source_motif_col = vs_split_elements[1].split("_matching_motif")[0] # e.g. "Novel_Motif_1"

        if motif_col_dict.get(source_motif_col) is None:
            motif_col_dict[source_motif_col] = [homolog_motif_col]
        else:
            motif_col_dict[source_motif_col].append(homolog_motif_col)

    # Pick the best homologous motif for each source motif
    total_homolog_id_cols, total_homolog_motif_cols, total_model_score_cols = [], [], []
    total_similarity_cols, total_identity_cols = [], []
    drop_cols = []
    for source_motif_col, matching_homolog_motif_cols in motif_col_dict.items():
        homolog_id_cols = [homolog_motif_col.split("_vs_")[0] for homolog_motif_col in matching_homolog_motif_cols]
        similarity_cols = [homolog_motif_col.split("_matching_motif")[0] + "_motif_similarity" for homolog_motif_col in matching_homolog_motif_cols]
        identity_cols = [homolog_motif_col.split("_matching_motif")[0] + "_motif_identity" for homolog_motif_col in matching_homolog_motif_cols]
        model_score_cols = [homolog_motif_col + "_model_score" for homolog_motif_col in matching_homolog_motif_cols]

        # Extract data to evaluate from dataframe
        homolog_ids_2d = data_df[homolog_id_cols].to_numpy(dtype="U")
        matching_homolog_motifs_2d = data_df[matching_homolog_motif_cols].to_numpy(dtype="U")
        similarities_2d = data_df[similarity_cols].to_numpy(dtype=float)
        identities_2d = data_df[identity_cols].to_numpy(dtype=float)
        scores_2d = data_df[model_score_cols].to_numpy(dtype=float)

        # Queue deletion of excess data from dataframe; best values are separately assigned
        drop_cols = drop_cols + homolog_id_cols + matching_homolog_motif_cols + model_score_cols
        drop_cols = drop_cols + similarity_cols + identity_cols

        # Get the best homologous motif cols for each row
        row_indices = np.arange(len(data_df))
        valid_mask = np.any(np.isfinite(scores_2d), axis=1)
        valid_row_indices = row_indices[valid_mask]

        valid_min_col_indices = np.nanargmax(scores_2d[valid_mask], axis=1)
        valid_best_homolog_ids = homolog_ids_2d[valid_row_indices, valid_min_col_indices]
        valid_best_homolog_motifs = matching_homolog_motifs_2d[valid_row_indices, valid_min_col_indices]
        valid_best_homolog_similarities = similarities_2d[valid_row_indices, valid_min_col_indices]
        valid_best_homolog_identities = identities_2d[valid_row_indices, valid_min_col_indices]
        valid_best_model_scores = scores_2d[valid_row_indices, valid_min_col_indices]

        best_homolog_ids = np.full(shape=len(data_df), fill_value="", dtype=valid_best_homolog_ids.dtype)
        best_homolog_motifs = np.full(shape=len(data_df), fill_value="", dtype=valid_best_homolog_motifs.dtype)
        best_homolog_similarities = np.full(shape=len(data_df), fill_value=np.nan, dtype=float)
        best_homolog_identities = np.full(shape=len(data_df), fill_value=np.nan, dtype=float)
        best_model_scores = np.full(shape=len(data_df), fill_value=np.nan, dtype=float)

        best_homolog_ids[valid_mask] = valid_best_homolog_ids
        best_homolog_motifs[valid_mask] = valid_best_homolog_motifs
        best_homolog_similarities[valid_mask] = valid_best_homolog_similarities
        best_homolog_identities[valid_mask] = valid_best_homolog_identities
        best_model_scores[valid_mask] = valid_best_model_scores

        # Assign the best homologous motifs back to the dataframe
        data_df[matching_homolog_motif_cols[0] + "_id_best"] = best_homolog_ids
        data_df[matching_homolog_motif_cols[0] + "_best"] = best_homolog_motifs
        data_df[matching_homolog_motif_cols[0] + "_similarity_best"] = best_homolog_similarities
        data_df[matching_homolog_motif_cols[0] + "_identity_best"] = best_homolog_identities
        data_df[model_score_cols[0] + "_best"] = best_model_scores

        total_homolog_id_cols.append(matching_homolog_motif_cols[0] + "_id_best")
        total_homolog_motif_cols.append(matching_homolog_motif_cols[0] + "_best")
        total_similarity_cols.append(matching_homolog_motif_cols[0] + "_similarity_best")
        total_identity_cols.append(matching_homolog_motif_cols[0] + "_identity_best")
        total_model_score_cols.append(model_score_cols[0] + "_best")

    drop_cols = list(set(drop_cols))
    data_df.drop(drop_cols, axis=1, inplace=True)

    return data_df, total_homolog_id_cols, total_homolog_motif_cols, total_model_score_cols

def score_homolog_motifs(data_df, homolog_motif_cols, predictor_params, verbose = True):
    '''
    Main function for scoring homologous motifs

    Args:
        data_df (pd.DataFrame):          main dataframe with motif sequences for host and homologs
        homolog_id_cols (list|tuple):    col names where homolog ids are stored
        homolog_motif_cols (list|tuple): col names where homolog motif sequences are stored
        predictor_params (dict):         dictionary of parameters for scoring
        verbose (bool):                  whether to display verbose progress messages

    Returns:
        data_df (pd.DataFrame):     dataframe with scores added for homolog motifs
        homolog_id_cols (list):     shortened list of col names where homolog ids are stored
        homolog_motif_cols (list):  shortened list of col names containing homologous motifs
        model_score_cols (list):    shortened list of col names containing homologous motif scores according to model
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

    # Extract all unique motif sequences for scoring
    print(f"\t\tGetting unique motif sequences...")
    motif_seqs = []
    for homolog_motif_col in homolog_motif_cols:
        col_data = data_df[homolog_motif_col].copy()
        col_data = col_data[col_data.notna()]
        col_data = col_data[col_data.ne("")]
        motif_seqs.append(col_data.to_numpy())

    motif_seqs = np.concatenate(motif_seqs)
    motif_seqs = np.unique(motif_seqs)
    motif_seqs_2d = np.array([list(motif) for motif in motif_seqs])

    # Score the unique motif sequences
    print(f"\t\tScoring {len(motif_seqs_2d)} unique motifs...")
    filters = predictor_params["enforced_position_rules"]
    selenocysteine_substitute = predictor_params["selenocysteine_substitute"]
    gap_substitute = predictor_params["gap_substitute"]
    chunk_size = predictor_params["homolog_score_chunk_size"]

    scores = score_motifs_parallel(motif_seqs_2d, conditional_matrices, weights_tuple, standardization_coefficients,
                                   chunk_size, filters, selenocysteine_substitute, gap_substitute)
    print(f"\t\tAssigning scores to dict of motif sequences...")
    scores_dict = {str(motif): score for motif, score in zip(motif_seqs, scores)}

    # Apply the scores to the dataframe as appropriate
    print(f"\t\tApplying scores to dataframe...")
    model_score_cols = []
    with trange(len(homolog_motif_cols), desc="\t\tApplying scores to dataframe...") as pbar:
        for i, homolog_motif_col in enumerate(homolog_motif_cols):
            col_scores = [scores_dict.get(motif) for motif in data_df[homolog_motif_col]]
            model_score_col = homolog_motif_col + "_model_score"
            data_df[model_score_col] = col_scores
            model_score_cols.append(model_score_col)
            pbar.update()

    # Collapse to best homolog motifs
    print(f"\t\tCollapsing to best homolog motifs...")
    data_df, homolog_id_cols, homolog_motif_cols, model_score_cols = collapse_to_best(data_df, homolog_motif_cols)

    # Reorder columns so model scores are beside homolog mostif sequences
    print(f"\t\tReordering dataframe...") if verbose else None
    cols = list(data_df.columns)
    for cumulative_displacement, (motif_col, score_col) in enumerate(zip(homolog_motif_cols, model_score_cols)):
        insertion_idx = cols.index(motif_col) + cumulative_displacement + 1
        cols.insert(insertion_idx, score_col)

    data_df = data_df[cols]

    return data_df, homolog_id_cols, homolog_motif_cols, model_score_cols