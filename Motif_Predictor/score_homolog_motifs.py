#This script takes protein sequences and computes their motif scores based on the results of make_pairwise_matrices.py

import numpy as np
import pandas as pd
import pickle
import multiprocessing
from tqdm import trange
from functools import partial
from Matrix_Generator.ConditionalMatrix import ConditionalMatrices
from general_utils.general_utils import finite_sorted_indices, add_number_suffix

# Import the user-specified params, either from a local version or the git-linked version
try:
    from Motif_Predictor.predictor_config_local import predictor_params
except:
    from Motif_Predictor.predictor_config import predictor_params

# If selected, import a parallel method for comparison
if predictor_params["compare_classical_method"]:
    from Motif_Predictor.classical_method import classical_method

def score_motifs(sequences_2d, conditional_matrices, weights_tuple = None, standardization_coefficients = None):
    '''
    Vectorized function to score homolog motif seqs based on the dictionary of context-aware weighted matrices

    Args:
        sequences_2d (np.ndarray):                  motif sequences to score
        conditional_matrices (ConditionalMatrices): conditional weighted matrices for scoring peptides
        weights_tuple (tuple):                      (positives_weights, suboptimals_weights, forbiddens_weights)
        standardization_coefficients (tuple)        tuple of coefficients from the model for standardizing score values

    Returns:
        total_scores (list):                       list of matching scores for each motif
    '''

    motif_length = sequences_2d.shape[1]

    # Get row indices for unique residues
    unique_residues = np.unique(sequences_2d)
    unique_residue_indices = conditional_matrices.index.get_indexer_for(unique_residues)
    if (unique_residue_indices == -1).any():
        failed_residues = unique_residues[unique_residue_indices == -1]
        raise Exception(f"residues not found by matrix indexer: {failed_residues}")

    # Get the matrix row indices for all the residues
    aa_row_indices_2d = np.ones(shape=sequences_2d.shape, dtype=int) * -1
    for unique_residue, row_index in zip(unique_residues, unique_residue_indices):
        aa_row_indices_2d[sequences_2d == unique_residue] = row_index

    # Define residues flanking either side of the residues of interest; for out-of-bounds cases, use opposite side
    flanking_left_2d = np.concatenate((sequences_2d[:, 0:1], sequences_2d[:, 0:-1]), axis=1)
    flanking_right_2d = np.concatenate((sequences_2d[:, 1:], sequences_2d[:, -1:]), axis=1)

    # Get integer-encoded chemical classes for each residue
    left_encoded_classes_2d = np.zeros(flanking_left_2d.shape, dtype=int)
    right_encoded_classes_2d = np.zeros(flanking_right_2d.shape, dtype=int)
    for member_aa, encoded_class in conditional_matrices.encoded_chemical_classes.items():
        left_encoded_classes_2d[flanking_left_2d == member_aa] = encoded_class
        right_encoded_classes_2d[flanking_right_2d == member_aa] = encoded_class

    # Find the matrix identifier number (1st dim of 3D matrix) for each encoded class, depending on seq position
    encoded_positions = np.arange(motif_length) * conditional_matrices.chemical_class_count
    left_encoded_matrix_refs = left_encoded_classes_2d + encoded_positions
    right_encoded_matrix_refs = right_encoded_classes_2d + encoded_positions

    # Flatten the encoded matrix refs, which serve as the 1st dimension referring to 3D matrices
    left_encoded_matrix_refs_flattened = left_encoded_matrix_refs.flatten()
    right_encoded_matrix_refs_flattened = right_encoded_matrix_refs.flatten()

    # Flatten the amino acid row indices into a matching array serving as the 2nd dimension
    aa_row_indices_flattened = aa_row_indices_2d.flatten()

    # Tile the column indices into a matching array serving as the 3rd dimension
    column_indices = np.arange(motif_length)
    column_indices_tiled = np.tile(column_indices, len(sequences_2d))

    # Define dimensions for 3D matrix indexing
    shape_2d = sequences_2d.shape
    left_dim1 = left_encoded_matrix_refs_flattened
    right_dim1 = right_encoded_matrix_refs_flattened
    dim2 = aa_row_indices_flattened
    dim3 = column_indices_tiled

    # Calculate predicted signal values
    left_positive_2d = conditional_matrices.stacked_positive_weighted[left_dim1, dim2, dim3].reshape(shape_2d)
    right_positive_2d = conditional_matrices.stacked_positive_weighted[right_dim1, dim2, dim3].reshape(shape_2d)
    positive_scores_2d = (left_positive_2d + right_positive_2d) / 2

    # Calculate suboptimal element scores
    left_suboptimal_2d = conditional_matrices.stacked_suboptimal_weighted[left_dim1, dim2, dim3].reshape(shape_2d)
    right_suboptimal_2d = conditional_matrices.stacked_suboptimal_weighted[right_dim1, dim2, dim3].reshape(shape_2d)
    suboptimal_scores_2d = (left_suboptimal_2d + right_suboptimal_2d) / 2

    # Calculate forbidden element scores
    left_forbidden_2d = conditional_matrices.stacked_forbidden_weighted[left_dim1, dim2, dim3].reshape(shape_2d)
    right_forbidden_2d = conditional_matrices.stacked_forbidden_weighted[right_dim1, dim2, dim3].reshape(shape_2d)
    forbidden_scores_2d = (left_forbidden_2d + right_forbidden_2d) / 2

    # Apply weights if a tuple of arrays of weights values were given
    if weights_tuple is not None:
        positives_weights, suboptimals_weights, forbiddens_weights = weights_tuple
        positive_scores_2d = np.multiply(positive_scores_2d, positives_weights)
        suboptimal_scores_2d = np.multiply(suboptimal_scores_2d, suboptimals_weights)
        forbidden_scores_2d = np.multiply(forbidden_scores_2d, forbiddens_weights)

    # Calculate total scores
    total_scores_2d = positive_scores_2d - suboptimal_scores_2d - forbidden_scores_2d
    total_scores = total_scores_2d.sum(axis=1)

    # Standardization of the scores
    if isinstance(standardization_coefficients, tuple) or isinstance(standardization_coefficients, list):
        coefficient_a, coefficient_b = standardization_coefficients
        total_scores = np.array(total_scores)
        total_scores = total_scores - coefficient_a
        total_scores = total_scores / coefficient_b
        total_scores = list(total_scores)

    return total_scores

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
    for homolog_motif_col in homolog_motif_cols:
        cols = list(data_df.columns)
        motifs = data_df[homolog_motif_col].to_list()
        motifs_2d = [list(motif) for motif in motifs]
        motifs_2d = np.array(motifs_2d)

        scores = score_motifs(motifs_2d, conditional_matrices, weights_tuple, standardization_coefficients)
        data_df[homolog_motif_col + "_model_score"] = scores

        col_idx = data_df.columns.get_loc(homolog_motif_col)
        cols.insert(col_idx+1, homolog_motif_col + "_model_score")
        data_df = data_df[cols]

    return data_df