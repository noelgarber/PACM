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

def score_motifs(seqs_2d, conditional_matrices, thresholds_tuple, weights_tuple = None,
                 standardization_coefficients = None, filters = None,
                 selenocysteine_substitute = "C", gap_substitute = "G"):
    '''
    Vectorized function to score homolog motif seqs based on the dictionary of context-aware weighted matrices

    Args:
        seqs_2d (np.ndarray):                       motif sequences to score
        conditional_matrices (ConditionalMatrices): conditional weighted matrices for scoring peptides
        thresholds_tuple (tuple):                   tuple of thresholds used for score classification
        weights_tuple (tuple):                      (positives_weights, suboptimals_weights, forbiddens_weights)
        standardization_coefficients (tuple)        tuple of coefficients from the model for standardizing score values
        filters (dict):                             dict of position index --> permitted residues
        selenocysteine_substitute (str):            letter to substitute for selenocysteine (U) when U is not in model
        gap_substitute (str):                       the letter to treat gaps ("X") as; default is no side chain, i.e. G

    Returns:
        results (tuple):   tuple of (total_scores, positive_scores, suboptimal_scores, forbidden_scores, final_calls)
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

    # Calculate total and stratified scores
    positive_scores = positive_scores_2d.sum(axis=1)
    suboptimal_scores = suboptimal_scores_2d.sum(axis=1)
    forbidden_scores = forbidden_scores_2d.sum(axis=1)
    del positive_scores_2d, suboptimal_scores_2d, forbidden_scores_2d
    total_scores = positive_scores - suboptimal_scores - forbidden_scores

    # Standardization of the scores
    if isinstance(standardization_coefficients, tuple) or isinstance(standardization_coefficients, list):
        total_scores = (total_scores - standardization_coefficients[0]) / standardization_coefficients[1]
        positive_scores = (positive_scores - standardization_coefficients[2]) / standardization_coefficients[3]
        suboptimal_scores = (suboptimal_scores - standardization_coefficients[4]) / standardization_coefficients[5]
        forbidden_scores = (forbidden_scores - standardization_coefficients[6]) / standardization_coefficients[7]

    # Test scores against thresholds
    above_positives = np.greater_equal(positive_scores, thresholds_tuple[0])
    below_suboptimals = np.less_equal(suboptimal_scores, thresholds_tuple[1])
    below_forbiddens = np.less_equal(forbidden_scores, thresholds_tuple[2])
    above_totals = np.greater_equal(total_scores, thresholds_tuple[3])
    final_calls = np.logical_and(np.logical_and(above_positives, above_totals),
                                    np.logical_and(below_suboptimals, below_forbiddens))

    # Apply filters
    if isinstance(filters, dict):
        for i, allowed_residues in filters.items():
            seqs_pass = np.full(shape=seqs_2d.shape[0], fill_value=False, dtype=bool)
            for allowed_aa in allowed_residues:
                seqs_pass = np.logical_or(seqs_pass, np.char.equal(seqs_2d[:,i], allowed_aa))
            total_scores[~seqs_pass] = 0

    motifs = ["".join(seqs_2d[i]) for i in np.arange(len(seqs_2d))]
    results = (motifs, total_scores, positive_scores, suboptimal_scores, forbidden_scores, final_calls)

    return results

def seqs_chunk_generator(seqs_2d, chunk_size):
    # Chunk generator; saves memory rather than loading list all at once
    for i in range(0, len(seqs_2d), chunk_size):
        yield seqs_2d[i:i+chunk_size]

def score_motifs_parallel(seqs_2d, conditional_matrices, thresholds_tuple, weights_tuple = None,
                          standardization_coefficients = None, chunk_size = 10000, filters = None,
                          selenocysteine_substitute = "C", gap_substitute = "G"):
    '''
    Parallelized function for scoring sequences using a ConditionalMatrices object

        seqs_2d (np.ndarray):                       motif sequences to score
        conditional_matrices (ConditionalMatrices): conditional weighted matrices for scoring peptides
        thresholds_tuple (tuple):                   tuple of thresholds used for score classification
        weights_tuple (tuple):                      (positives_weights, suboptimals_weights, forbiddens_weights)
        standardization_coefficients (tuple)        tuple of coefficients from the model for standardizing score values
        chunk_size (int):                           number of sequences per parallel processing chunk
        filters (dict):                             dict of position index --> permitted residues
        selenocysteine_substitute (str):            letter to substitute for selenocysteine (U) when U is not in model
        gap_substitute (str):                       the letter to treat gaps ("X") as; default is no side chain, i.e. G

    Returns:
        total_scores (list):                        list of matching scores for each motif
    '''

    partial_function = partial(score_motifs, conditional_matrices = conditional_matrices,
                               thresholds_tuple = thresholds_tuple, weights_tuple = weights_tuple,
                               standardization_coefficients = standardization_coefficients, filters = filters,
                               selenocysteine_substitute = selenocysteine_substitute, gap_substitute = gap_substitute)

    chunk_motifs = []
    chunk_total_scores = []
    chunk_positive_scores = []
    chunk_suboptimal_scores = []
    chunk_forbidden_scores = []
    chunk_final_calls = []
    pool = multiprocessing.Pool()

    for results in pool.map(partial_function, seqs_chunk_generator(seqs_2d, chunk_size)):
        chunk_motifs.append(results[0])
        chunk_total_scores.append(results[1])
        chunk_positive_scores.append(results[2])
        chunk_suboptimal_scores.append(results[3])
        chunk_forbidden_scores.append(results[4])
        chunk_final_calls.append(results[5])

    pool.close()
    pool.join()

    chunk_motifs = np.concatenate(chunk_motifs)
    chunk_total_scores = np.concatenate(chunk_total_scores)
    chunk_positive_scores = np.concatenate(chunk_positive_scores)
    chunk_suboptimal_scores = np.concatenate(chunk_suboptimal_scores)
    chunk_forbidden_scores = np.concatenate(chunk_forbidden_scores)
    chunk_final_calls = np.concatenate(chunk_final_calls)

    output = (chunk_motifs, chunk_total_scores, chunk_positive_scores, chunk_suboptimal_scores, chunk_forbidden_scores,
              chunk_final_calls)

    return output

def score_homolog_motifs(data_df, homolog_motif_cols, homolog_motif_col_groups, predictor_params):
    '''
    Main function for scoring homologous motifs

    Args:
        data_df (pd.DataFrame):          main dataframe with motif sequences for host and homologs
        homolog_motif_cols (list|tuple): col names where homolog motif sequences are stored
        homolog_motif_col_groups (list): list of lists of grouped column names for each homologous motif
        predictor_params (dict):         dictionary of parameters for scoring

    Returns:
        data_df (pd.DataFrame):     dataframe with scores added for homolog motifs
        homolog_id_cols (list):     shortened list of col names where homolog ids are stored
        homolog_motif_cols (list):  shortened list of col names containing homologous motifs
        model_score_cols (list):    shortened list of col names containing homologous motif scores according to model
    '''

    standardization_coefficients_path = predictor_params["standardization_coefficients_path"]
    with open(standardization_coefficients_path, "rb") as f:
        standardization_coefficients = pickle.load(f)

    optimized_thresholds_path = predictor_params["optimized_thresholds_path"]
    with open(optimized_thresholds_path, "rb") as f:
        optimized_thresholds = pickle.load(f)

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

    results = score_motifs_parallel(motif_seqs_2d, conditional_matrices, optimized_thresholds, weights_tuple,
                                    standardization_coefficients, chunk_size, filters,
                                    selenocysteine_substitute, gap_substitute)

    print(f"\t\tParsing results into motif-score dicts...")
    motifs, total_scores, positive_scores, suboptimal_scores, forbidden_scores, final_calls = results
    zipped_results = zip(motifs, total_scores, positive_scores, suboptimal_scores, forbidden_scores, final_calls)
    total_dict = {}
    positive_dict = {}
    suboptimal_dict = {}
    forbidden_dict = {}
    calls_dict = {}
    combined_dict = {}
    for motif, total_score, positive, suboptimal, forbidden, call in zipped_results:
        total_dict[motif] = total_score
        positive_dict[motif] = positive
        suboptimal_dict[motif] = suboptimal
        forbidden_dict[motif] = forbidden
        calls_dict[motif] = call
        combined_dict[motif] = (total_score, positive, suboptimal, forbidden, call)

    # Apply the scores to the dataframe as appropriate
    final_homolog_motif_cols = []
    homolog_motif_cols = [col_group[0] for col_group in homolog_motif_col_groups]
    similarity_cols = [col_group[1] for col_group in homolog_motif_col_groups]
    identity_cols = [col_group[2] for col_group in homolog_motif_col_groups]
    homolog_id_cols = [homolog_motif_col.split("_vs_")[0] for homolog_motif_col in homolog_motif_cols]

    # Extract homologous motifs from dataframe
    homolog_motifs_grid = data_df[homolog_motif_cols].copy()
    data_df.drop(homolog_motif_cols, axis=1, inplace=True)

    print("\t\tOrganizing motif score information...")
    total_scores_grid = homolog_motifs_grid.applymap(lambda x: total_dict.get(x)).to_numpy(dtype=float)
    positive_scores_grid = homolog_motifs_grid.applymap(lambda x: positive_dict.get(x)).to_numpy(dtype=float)
    suboptimal_scores_grid = homolog_motifs_grid.applymap(lambda x: suboptimal_dict.get(x)).to_numpy(dtype=float)
    forbidden_scores_grid = homolog_motifs_grid.applymap(lambda x: forbidden_dict.get(x)).to_numpy(dtype=float)
    final_calls_grid = homolog_motifs_grid.applymap(lambda x: calls_dict.get(x)).to_numpy(dtype=bool)

    # Find best col indices for best homologous motifs
    selection_mode = predictor_params["homolog_selection_mode"]
    if selection_mode == "similarity":
        similarities_grid = data_df[similarity_cols].to_numpy(dtype=float)
        best_col_indices = np.nanargmax(similarities_grid, axis=1)
    elif selection_mode == "identity":
        identities_grid = data_df[identity_cols].to_numpy(dtype=float)
        best_col_indices = np.nanargmax(identities_grid, axis=1)
    elif selection_mode == "score":
        best_col_indices = np.nanargmax(total_scores_grid, axis=1)
    else:
        message = f"predictor_params[mode] was set to {selection_mode}, but must be identity, similarity, or score"
        raise ValueError(message)

    # Extract and assign best homologous motifs
    print(f"\tAssigning best homologous motifs to dataframe and removing others...")

    row_indices = np.arange(len(data_df))

    homolog_col_element, source_col_element = homolog_motif_cols[0].split("_vs_")
    homolog_col_element = homolog_col_element.rsplit("_", 1)[0]
    source_col_element = source_col_element.split("_matching_motif")[0]
    col_prefix = f"{homolog_col_element}_vs_{source_col_element}"

    homolog_ids_grid = data_df[homolog_id_cols].to_numpy(dtype="U")
    data_df.drop(homolog_id_cols, axis=1, inplace=True)
    best_homolog_ids = homolog_ids_grid[row_indices, best_col_indices]
    data_df[col_prefix + "_id_best"] = best_homolog_ids
    del homolog_ids_grid, best_homolog_ids

    best_homolog_motifs = homolog_motifs_grid.values[row_indices, best_col_indices]
    data_df[col_prefix + "_best"] = best_homolog_motifs
    del homolog_motifs_grid, best_homolog_motifs

    similarities_grid = data_df[similarity_cols].to_numpy(dtype=float)
    data_df.drop(similarity_cols, axis=1, inplace=True)
    best_similarities = similarities_grid[row_indices, best_col_indices]
    data_df[col_prefix + "_similarity_best"] = best_similarities
    del similarities_grid, best_similarities

    identities_grid = data_df[identity_cols].to_numpy(dtype=float)
    data_df.drop(identity_cols, axis=1, inplace=True)
    best_identities = identities_grid[row_indices, best_col_indices]
    data_df[col_prefix + "_identity_best"] = best_identities
    final_homolog_motif_cols.append(col_prefix + "_best")
    del identities_grid, best_identities

    best_positive_scores = positive_scores_grid[row_indices, best_col_indices]
    data_df[col_prefix + "_best_positive_model_score"] = best_positive_scores
    del positive_scores_grid, best_positive_scores

    best_suboptimal_scores = suboptimal_scores_grid[row_indices, best_col_indices]
    data_df[col_prefix + "_best_suboptimal_model_score"] = best_suboptimal_scores
    del suboptimal_scores_grid, best_suboptimal_scores

    best_forbidden_scores = forbidden_scores_grid[row_indices, best_col_indices]
    data_df[col_prefix + "_best_forbidden_model_score"] = best_forbidden_scores
    del forbidden_scores_grid, best_forbidden_scores

    best_total_scores = total_scores_grid[row_indices, best_col_indices]
    data_df[col_prefix + "_best_total_model_score"] = best_total_scores
    del total_scores_grid, best_total_scores

    best_calls = final_calls_grid[row_indices, best_col_indices]
    data_df[col_prefix + "_best_model_call"] = best_calls
    del final_calls_grid, best_calls

    return data_df, final_homolog_motif_cols