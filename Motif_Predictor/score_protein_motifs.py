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

def score_sliced_protein(sequences_2d, conditional_matrices, weights_tuple = None, return_count = 3):
    '''
    Vectorized function to score amino acid sequences based on the dictionary of context-aware weighted matrices

    Args:
        sequences_2d (np.ndarray):                  unravelled peptide sequences to score
        conditional_matrices (ConditionalMatrices): conditional weighted matrices for scoring peptides
        weights_tuple (tuple):                      (positives_weights, suboptimals_weights, forbiddens_weights)

    Returns:
        output_motifs (list):                       list of motifs as strings
        output_scores (list):                       list of matching scores for each motif
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

    valid_row_indices = np.isfinite(total_scores)
    valid_seqs_2d = sequences_2d[valid_row_indices]
    valid_scores = total_scores[valid_row_indices]

    sorted_indices = np.argsort(valid_scores * -1) # multiply by -1 to get indices in descending order
    sorted_seqs_2d = valid_seqs_2d[sorted_indices]
    sorted_scores = valid_scores[sorted_indices]

    # Return the desired number of motifs and handle cases when less motifs than desired are found
    output_motifs = []
    output_scores = []
    real_output_count = return_count if return_count <= len(sorted_scores) else len(sorted_scores)
    for i in np.arange(real_output_count):
        output_motifs.append("".join(sorted_seqs_2d[i]))
        output_scores.append(sorted_scores[i])

    if len(sorted_scores) < return_count:
        for i in np.arange(real_output_count, return_count):
            output_motifs.append("")
            output_scores.append(np.nan)

    return output_motifs, output_scores

def scan_protein_seq(protein_seq, conditional_matrices, weights_tuple, predictor_params = predictor_params):
    '''

    Args:
        protein_seq (str):                          full length protein sequence to score
        conditional_matrices (ConditionalMatrices): object containing conditional weighted matrices
        weights_tuple (tuple):                      tuple of arrays of weights values
        predictor_params (dict):                    dictionary of user-defined parameters from predictor_config.py

    Returns:
        best_score, best_motif, second_best_score, second_best_motif
    '''

    # Get necessary arguments
    motif_length = predictor_params["motif_length"]
    return_count = predictor_params["return_count"]

    # Get N-term and C-term trailing residue info
    leading_glycines = np.repeat("G", predictor_params["leading_glycines"])
    trailing_glycines = np.repeat("G", predictor_params["trailing_glycines"])

    # Determine whether protein seq is valid
    valid_seq = True
    if not isinstance(protein_seq, str):
        valid_seq = False
    elif len(protein_seq) < motif_length:
        valid_seq = False
    elif "*" in protein_seq[:-1]:
        valid_seq = False

    # Extract protein sequence into overlapping motif-sized segments with step size of 1
    if valid_seq:
        # Remove stop asterisk if present
        if protein_seq[-1] == "*":
            protein_seq = protein_seq[:-1]

        seq_array = np.array(list(protein_seq))

        seq_array = np.concatenate([leading_glycines, seq_array, trailing_glycines])
        slice_indices = np.arange(len(seq_array) - motif_length + 1)[:, np.newaxis] + np.arange(motif_length)
        sliced_seqs_2d = seq_array[slice_indices]

        # Enforce position rules
        enforced_position_rules = predictor_params.get("enforced_position_rules")
        if enforced_position_rules is not None:
            for position_index, allowed_residues in enforced_position_rules.items():
                column_residues = sliced_seqs_2d[:,position_index]
                residues_allowed = np.isin(column_residues, allowed_residues)
                sliced_seqs_2d = sliced_seqs_2d[residues_allowed]

        # After rules have been enforced, remove slices with uncertain residues (X)
        missing_residues = "X" in sliced_seqs_2d
        cleaned_sliced_2d = sliced_seqs_2d.copy()
        if missing_residues:
            missing_counts = np.sum(cleaned_sliced_2d == "X", axis = 1)
            cleaned_sliced_2d = cleaned_sliced_2d[missing_counts == 0]

        # Optionally replace selenocysteine with cysteine for scoring purposes
        replace_selenocysteine = predictor_params["replace_selenocysteine"]
        if replace_selenocysteine and "U" in sliced_seqs_2d:
            cleaned_sliced_2d[cleaned_sliced_2d == "U"] = "C"

        # Calculate motif scores
        if len(cleaned_sliced_2d) > 0:
            motifs, scores = score_sliced_protein(cleaned_sliced_2d, conditional_matrices, weights_tuple, return_count)
        else:
            motifs = ["" for i in np.arange(return_count)]
            scores = [np.nan for i in np.arange(return_count)]

    else:
        motifs = ["" for i in np.arange(return_count)]
        scores = [np.nan for i in np.arange(return_count)]

    return motifs, scores

def score_proteins_chunk(df_chunk, predictor_params = predictor_params):
    '''
    Lower level function to score a chunk of protein sequences in parallel based on conditional matrices

    Args:
        df_chunk (pd.DataFrame):        chunk of protein sequences dataframe
        predictor_params (dict):        dictionary of user-defined parameters from predictor_config.py

    Returns:
        df_chunk_scored (pd.DataFrame): dataframe with protein sequences and found motifs
    '''

    # Get protein sequences to score
    protein_seqs_list = df_chunk["Sequence"].to_list()
    df_chunk_scored = df_chunk.copy()

    # Load ConditionalMatrices object to be used in scoring
    conditional_matrices_path = predictor_params["conditional_matrices_path"]
    with open(conditional_matrices_path, "rb") as f:
        conditional_matrices = pickle.load(f)

    # Generate columns for the number of motifs that will be returned per protein
    return_count = predictor_params["return_count"]
    compare_classical_method = predictor_params["compare_classical_method"]

    ordered_motifs_cols = [[] for i in np.arange(return_count)]
    ordered_scores_cols = [[] for i in np.arange(return_count)]
    classical_motifs_cols = [[] for i in np.arange(return_count)]
    classical_scores_cols = [[] for i in np.arange(return_count)]

    motif_col_names = []
    score_col_names = []
    for i in np.arange(return_count):
        suffix_number = add_number_suffix(i+1)
        motif_col_names.append(suffix_number+"_motif")
        score_col_names.append(suffix_number+"_motif_score")

    # Assemble a partial function for scoring individual proteins
    weights_path = predictor_params["pickled_weights_path"]
    with open(weights_path, "rb") as f:
        weights_tuple = pickle.load(f)

    scan_seq_partial = partial(scan_protein_seq, conditional_matrices = conditional_matrices,
                               weights_tuple = weights_tuple, predictor_params = predictor_params)

    # Loop over the protein sequences to score them
    for i, protein_seq in enumerate(protein_seqs_list):
        # Score the protein sequence using conditional matrices
        motifs, scores = scan_seq_partial(protein_seq)
        for j, (motif, score) in enumerate(zip(motifs, scores)):
            ordered_motifs_cols[j].append(motif)
            ordered_scores_cols[j].append(score)

        # Optionally score the sequence using a classical method for comparison
        if compare_classical_method:
            classical_motifs, classical_scores = classical_method(protein_seq, predictor_params)
            for j, (classical_motif, classical_score) in enumerate(zip(classical_motifs, classical_scores)):
                classical_motifs_cols[j].append(classical_motif)
                classical_scores_cols[j].append(classical_score)

    # Apply motifs and scores as columns to the dataframe
    zipped_cols = zip(ordered_motifs_cols, motif_col_names, ordered_scores_cols, score_col_names)
    for ordered_motifs_col, motif_col_name, ordered_scores_col, score_col_name in zipped_cols:
        if compare_classical_method:
            motif_col_name = "Novel_" + motif_col_name
            score_col_name = "Novel_" + score_col_name
        df_chunk_scored[motif_col_name] = ordered_motifs_col
        df_chunk_scored[score_col_name] = ordered_scores_col

    # Optionally apply classical motifs and scores as columns if they were generated
    if compare_classical_method:
        zipped_classical_cols = zip(classical_motifs_cols, motif_col_names, classical_scores_cols, score_col_names)
        for classical_motif_col, motif_col_name, classical_scores_col, score_col_name in zipped_classical_cols:
            motif_col_name = "Classical_" + motif_col_name
            score_col_name = "Classical_" + score_col_name
            df_chunk_scored[motif_col_name] = classical_motif_col
            df_chunk_scored[score_col_name] = classical_scores_col

    return df_chunk_scored

def score_proteins(protein_seqs_df, predictor_params = predictor_params):
    '''
    Upper level function to score protein sequences in parallel based on conditional matrices

    Args:
        protein_seqs_df (pd.DataFrame):   protein sequences dataframe
        predictor_params (dict):          dictionary of user-defined parameters from predictor_config.py

    Returns:
        scored_protein_df (pd.DataFrame): dataframe with protein sequences and found motifs
    '''

    chunk_size = predictor_params["chunk_size"]
    df_chunks = [protein_seqs_df.iloc[i:i + chunk_size] for i in range(0, len(protein_seqs_df), chunk_size)]

    score_chunk_partial = partial(score_proteins_chunk, predictor_params = predictor_params)

    pool = multiprocessing.Pool()
    scored_chunks = []

    with trange(len(protein_seqs_df), desc="Scoring proteins...") as pbar:
        for df_chunk_scored in pool.imap_unordered(score_chunk_partial, df_chunks):
            scored_chunks.append(df_chunk_scored)
            pbar.update()

    pool.close()
    pool.join()

    scored_protein_df = pd.concat(scored_chunks, ignore_index=True)

    return scored_protein_df