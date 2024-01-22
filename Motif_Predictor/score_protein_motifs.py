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
    from Motif_Predictor.classical_method import classical_protein_method

def score_sliced_protein(sequences_2d, conditional_matrices, return_count = 3, std_coefs = None):
    '''
    Vectorized function to score amino acid sequences based on the dictionary of context-aware weighted matrices

    Args:
        sequences_2d (np.ndarray):                  unravelled peptide sequences to score
        conditional_matrices (ConditionalMatrices): conditional weighted matrices for scoring peptides
        return_count (int):                         number of motifs to return
        std_coefs (tuple)        tuple of coefficients from the model for standardizing score values

    Returns:
        output_motifs (list):                       list of motifs as strings
        output_total_scores (list):                 list of matching total scores for each motif
        output_positive_scores (list):              list of matching positive element scores for each motif
        output_suboptimal_scores (list):            list of matching suboptimal element scores for each motif
        output_forbidden_scores (list):             list of matching forbidden element scores for each motif
    '''

    # Calculate scores using conditional matrices
    scoring_results = conditional_matrices.score_seqs_2d(sequences_2d, use_weighted = True)
    binding_scores_2d, positive_scores_2d, suboptimal_scores_2d, forbidden_scores_2d = scoring_results

    forbidden_scores_2d[:,conditional_matrices.suppress_forbidden_positions] = 0
    disqualified_forbidden = np.any(forbidden_scores_2d > 0, axis=1)

    binding_weighted_scores = binding_scores_2d.sum(axis=1)
    positive_weighted_scores = positive_scores_2d.sum(axis=1)
    suboptimal_weighted_scores = suboptimal_scores_2d.sum(axis=1)
    forbidden_scores = forbidden_scores_2d.sum(axis=1)

    # Calculate total scores
    if conditional_matrices.best_accuracy_method == "ps":
        total_scores = positive_weighted_scores - suboptimal_weighted_scores
    elif conditional_matrices.best_accuracy_method == "wps":
        total_scores = binding_weighted_scores - suboptimal_weighted_scores
    elif conditional_matrices.best_accuracy_method == "suboptimal":
        total_scores = suboptimal_weighted_scores * -1
    else:
        raise ValueError(f"conditional_matrices.best_accuracy_method is {conditional_matrices.best_accuracy_method}")

    # Standardization of the scores
    if isinstance(std_coefs, tuple) or isinstance(std_coefs, list) or isinstance(std_coefs, np.ndarray):
        total_scores = (total_scores - std_coefs[0]) / std_coefs[1]
        positive_weighted_scores = (positive_weighted_scores - std_coefs[2]) / std_coefs[3]
        suboptimal_weighted_scores = (suboptimal_weighted_scores - std_coefs[4]) / std_coefs[5]

    # Get binary predictions
    total_scores[disqualified_forbidden] = np.nan
    threshold = conditional_matrices.standardized_weighted_threshold
    predicted_calls = np.greater_equal(total_scores, threshold)

    # Sort passing motifs (which will be preferentially selected)
    passing_seqs_2d = sequences_2d[predicted_calls]
    passing_total_scores = total_scores[predicted_calls]
    passing_binding_scores = binding_weighted_scores[predicted_calls]
    passing_positive_scores = positive_weighted_scores[predicted_calls]
    passing_suboptimal_scores = suboptimal_weighted_scores[predicted_calls]
    passing_forbidden_scores = forbidden_scores[predicted_calls]

    sorted_passing_indices = np.argsort(passing_total_scores * -1) # multiply by -1 to get indices in descending order
    sorted_passing_seqs_2d = passing_seqs_2d[sorted_passing_indices]
    sorted_passing_total_scores = passing_total_scores[sorted_passing_indices]
    sorted_passing_binding_scores = passing_binding_scores[sorted_passing_indices]
    sorted_passing_positive_scores = passing_positive_scores[sorted_passing_indices]
    sorted_passing_suboptimal_scores = passing_suboptimal_scores[sorted_passing_indices]
    sorted_passing_forbidden_scores = passing_forbidden_scores[sorted_passing_indices]

    # Return the desired number of motifs into a set of lists
    output_motifs = []
    output_total_scores = []
    output_binding_scores = []
    output_positive_scores = []
    output_suboptimal_scores = []
    output_forbidden_scores = []
    output_calls = []

    # Select best motifs, first selecting from the group that pass the thresholds
    passing_count = len(sorted_passing_total_scores)
    passing_output_count = return_count if return_count <= passing_count else passing_count
    for i in np.arange(passing_output_count):
        output_motifs.append("".join(sorted_passing_seqs_2d[i]))
        output_total_scores.append(sorted_passing_total_scores[i])
        output_binding_scores.append(sorted_passing_binding_scores[i])
        output_positive_scores.append(sorted_passing_positive_scores[i])
        output_suboptimal_scores.append(sorted_passing_suboptimal_scores[i])
        output_forbidden_scores.append(sorted_passing_forbidden_scores[i])
        output_calls.append(True)

    # Handle cases where not enough score sets pass thresholds
    if passing_output_count < return_count:
        # Sort failing motifs
        failing_seqs_2d = sequences_2d[~predicted_calls]
        failing_total_scores = total_scores[~predicted_calls]
        failing_binding_scores = binding_weighted_scores[~predicted_calls]
        failing_positive_scores = positive_weighted_scores[~predicted_calls]
        failing_suboptimal_scores = suboptimal_weighted_scores[~predicted_calls]
        failing_forbidden_scores = forbidden_scores[~predicted_calls]

        sorted_failing_indices = np.argsort(failing_total_scores * -1)  # multiply by -1 to get in descending order
        sorted_failing_seqs_2d = failing_seqs_2d[sorted_failing_indices]
        sorted_failing_total_scores = failing_total_scores[sorted_failing_indices]
        sorted_failing_binding_scores = failing_binding_scores[sorted_failing_indices]
        sorted_failing_positive_scores = failing_positive_scores[sorted_failing_indices]
        sorted_failing_suboptimal_scores = failing_suboptimal_scores[sorted_failing_indices]
        sorted_failing_forbidden_scores = failing_forbidden_scores[sorted_failing_indices]
        failing_count = len(sorted_failing_total_scores)

        # Add best failed motifs to make up for not enough passing motifs existing to fill the return_count
        remaining_output_count = return_count - passing_output_count
        failing_output_count = remaining_output_count if remaining_output_count <= failing_count else failing_count
        for i in np.arange(failing_output_count):
            output_motifs.append("".join(sorted_failing_seqs_2d[i]))
            output_total_scores.append(sorted_failing_total_scores[i])
            output_binding_scores.append(sorted_failing_binding_scores[i])
            output_positive_scores.append(sorted_failing_positive_scores[i])
            output_suboptimal_scores.append(sorted_failing_suboptimal_scores[i])
            output_forbidden_scores.append(sorted_failing_forbidden_scores[i])
            output_calls.append(False)

        # If there are still not enough output motifs, supplement with blanks
        if len(output_total_scores) < return_count:
            blanks_count = return_count - len(output_total_scores)
            for i in np.arange(blanks_count):
                output_motifs.append("")
                output_total_scores.append(np.nan)
                output_binding_scores.append(np.nan)
                output_positive_scores.append(np.nan)
                output_suboptimal_scores.append(np.nan)
                output_forbidden_scores.append(np.nan)
                output_calls.append(False)

    output_lists = (output_motifs, output_total_scores, output_binding_scores, output_positive_scores,
                    output_suboptimal_scores, output_forbidden_scores, output_calls)

    return output_lists

def scan_protein_seq(protein_seq, conditional_matrices, predictor_params = predictor_params):
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
            # Get the standardization coefficients to convert raw scores to floats between 0 and 1
            standardization_coefficients_path = predictor_params["standardization_coefficients_path"]
            with open(standardization_coefficients_path, "rb") as f:
                standardization_coefficients = pickle.load(f)

            # Get the optimized thresholds for converting sets of scores to binary classifications
            optimized_thresholds_path = predictor_params["optimized_thresholds_path"]
            with open(optimized_thresholds_path, "rb") as f:
                optimized_thresholds = pickle.load(f)

            # Score the protein sequence chunks
            output_lists = score_sliced_protein(cleaned_sliced_2d, conditional_matrices, return_count,
                                                standardization_coefficients)
            motifs, total_scores = output_lists[0:2]
            binding_scores, positive_scores, suboptimal_scores, forbidden_scores, final_calls = output_lists[2:]

        else:
            # Assign blank values if there is no sequence to score
            motifs = ["" for i in np.arange(return_count)]
            total_scores = [np.nan for i in np.arange(return_count)]
            binding_scores = [np.nan for i in np.arange(return_count)]
            positive_scores = [np.nan for i in np.arange(return_count)]
            suboptimal_scores = [np.nan for i in np.arange(return_count)]
            forbidden_scores = [np.nan for i in np.arange(return_count)]
            final_calls = [False for i in np.arange(return_count)]

    else:
        # Assign blank values if sequence is not valid
        motifs = ["" for i in np.arange(return_count)]
        total_scores = [np.nan for i in np.arange(return_count)]
        binding_scores = [np.nan for i in np.arange(return_count)]
        positive_scores = [np.nan for i in np.arange(return_count)]
        suboptimal_scores = [np.nan for i in np.arange(return_count)]
        forbidden_scores = [np.nan for i in np.arange(return_count)]
        final_calls = [False for i in np.arange(return_count)]

    return (motifs, total_scores, binding_scores, positive_scores, suboptimal_scores, forbidden_scores, final_calls)

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
    seq_col = predictor_params["seq_col"]
    protein_seqs_list = df_chunk[seq_col].to_list()
    df_chunk_scored = df_chunk

    # Load ConditionalMatrices object to be used in scoring
    conditional_matrices_path = predictor_params["conditional_matrices_path"]
    with open(conditional_matrices_path, "rb") as f:
        conditional_matrices = pickle.load(f)

    # Generate columns for the number of motifs that will be returned per protein
    return_count = predictor_params["return_count"]
    compare_classical_method = predictor_params["compare_classical_method"]

    ordered_motifs_cols = [[] for i in np.arange(return_count)]
    ordered_total_scores_cols = [[] for i in np.arange(return_count)]
    ordered_binding_scores_cols = [[] for i in np.arange(return_count)]
    ordered_positive_scores_cols = [[] for i in np.arange(return_count)]
    ordered_suboptimal_scores_cols = [[] for i in np.arange(return_count)]
    ordered_forbidden_scores_cols = [[] for i in np.arange(return_count)]
    ordered_final_calls_cols = [[] for i in np.arange(return_count)]
    classical_motifs_cols = [[] for i in np.arange(return_count)]
    classical_scores_cols = [[] for i in np.arange(return_count)]

    # Generate names for the aforementioned columns
    motif_col_names = []
    total_score_col_names = []
    binding_score_col_names = []
    positive_score_col_names = []
    suboptimal_score_col_names = []
    forbidden_score_col_names = []
    final_call_col_names = []
    for i in np.arange(return_count):
        suffix_number = add_number_suffix(i+1)
        motif_col_names.append(suffix_number+"_motif")
        total_score_col_names.append(suffix_number+"_total_motif_score")
        binding_score_col_names.append(suffix_number+"_binding_motif_score")
        positive_score_col_names.append(suffix_number+"_positive_motif_score")
        suboptimal_score_col_names.append(suffix_number+"_suboptimal_motif_score")
        forbidden_score_col_names.append(suffix_number+"_forbidden_motif_score")
        final_call_col_names.append(suffix_number+"_final_call")

    # Assemble a partial function for scoring individual proteins
    weights_path = predictor_params["pickled_weights_path"]
    with open(weights_path, "rb") as f:
        weights_tuple = pickle.load(f)

    scan_seq_partial = partial(scan_protein_seq, conditional_matrices = conditional_matrices,
                               weights_tuple = weights_tuple, predictor_params = predictor_params)

    # Loop over the protein sequences to score them
    for i, protein_seq in enumerate(protein_seqs_list):
        # Score the protein sequence using conditional matrices
        protein_results = scan_seq_partial(protein_seq)
        motifs, total_scores, binding_scores = protein_results[0:3]
        positive_scores, suboptimal_scores, forbidden_scores, final_calls = protein_results[3:]
        zipped_results = zip(motifs, total_scores, binding_scores,
                             positive_scores, suboptimal_scores, forbidden_scores, final_calls)
        for j, (motif, total, binding, positive, suboptimal, forbidden, call) in enumerate(zipped_results):
            ordered_motifs_cols[j].append(motif)
            ordered_total_scores_cols[j].append(total)
            ordered_binding_scores_cols[j].append(binding)
            ordered_positive_scores_cols[j].append(positive)
            ordered_suboptimal_scores_cols[j].append(suboptimal)
            ordered_forbidden_scores_cols[j].append(forbidden)
            ordered_final_calls_cols[j].append(call)

        # Optionally score the sequence using a classical method for comparison
        if compare_classical_method:
            classical_motifs, classical_scores = classical_protein_method(protein_seq, predictor_params)
            for j, (classical_motif, classical_score) in enumerate(zip(classical_motifs, classical_scores)):
                classical_motifs_cols[j].append(classical_motif)
                classical_scores_cols[j].append(classical_score)

    # Apply motifs and scores as columns to the dataframe, and record col names
    novel_motif_headers = []
    novel_total_score_headers = []
    novel_binding_score_headers = []
    novel_positive_score_headers = []
    novel_suboptimal_score_headers = []
    novel_forbidden_score_headers = []
    novel_final_call_headers = []

    for i in np.arange(len(motif_col_names)):
        ordered_motifs_col = ordered_motifs_cols[i]
        motif_col_name = motif_col_names[i]

        ordered_total_scores_col = ordered_total_scores_cols[i]
        total_score_col_name = total_score_col_names[i]

        ordered_binding_scores_col = ordered_binding_scores_cols[i]
        binding_score_col_name = binding_score_col_names[i]

        ordered_positive_scores_col = ordered_positive_scores_cols[i]
        positive_score_col_name = positive_score_col_names[i]

        ordered_suboptimal_scores_col = ordered_suboptimal_scores_cols[i]
        suboptimal_score_col_name = suboptimal_score_col_names[i]

        ordered_forbidden_scores_col = ordered_forbidden_scores_cols[i]
        forbidden_score_col_name = forbidden_score_col_names[i]

        ordered_final_calls_col = ordered_final_calls_cols[i]
        final_call_col_name = final_call_col_names[i]

        if compare_classical_method:
            motif_col_name = f"Novel_{motif_col_name}"
            total_score_col_name = f"Novel_{total_score_col_name}"
            binding_score_col_name = f"Novel_{binding_score_col_name}"
            positive_score_col_name = f"Novel_{positive_score_col_name}"
            suboptimal_score_col_name = f"Novel_{suboptimal_score_col_name}"
            forbidden_score_col_name = f"Novel_{forbidden_score_col_name}"
            final_call_col_name = f"Novel_{final_call_col_name}"

        df_chunk_scored[motif_col_name] = ordered_motifs_col
        df_chunk_scored[total_score_col_name] = ordered_total_scores_col
        df_chunk_scored[binding_score_col_name] = ordered_binding_scores_col
        df_chunk_scored[positive_score_col_name] = ordered_positive_scores_col
        df_chunk_scored[suboptimal_score_col_name] = ordered_suboptimal_scores_col
        df_chunk_scored[forbidden_score_col_name] = ordered_forbidden_scores_col
        df_chunk_scored[final_call_col_name] = ordered_final_calls_col

        novel_motif_headers.append(motif_col_name)
        novel_total_score_headers.append(total_score_col_name)
        novel_binding_score_headers.append(binding_score_col_name)
        novel_positive_score_headers.append(positive_score_col_name)
        novel_suboptimal_score_headers.append(suboptimal_score_col_name)
        novel_forbidden_score_headers.append(forbidden_score_col_name)
        novel_final_call_headers.append(final_call_col_name)

    # Optionally apply classical motifs and scores as columns if they were generated
    classical_motif_headers = []
    classical_score_headers = []
    if compare_classical_method:
        zipped_classical_cols = zip(classical_motifs_cols, motif_col_names,
                                    classical_scores_cols, total_score_col_names)
        for classical_motif_col, motif_col_name, classical_scores_col, total_score_col_name in zipped_classical_cols:
            classical_motif_col_name = "Classical_" + motif_col_name
            classical_score_col_name = "Classical_" + total_score_col_name
            df_chunk_scored[classical_motif_col_name] = classical_motif_col
            df_chunk_scored[classical_score_col_name] = classical_scores_col

            classical_motif_headers.append(classical_motif_col_name)
            classical_score_headers.append(classical_score_col_name)

    results_tuple = (df_chunk_scored, novel_motif_headers, novel_total_score_headers, novel_binding_score_headers,
                     novel_positive_score_headers, novel_suboptimal_score_headers, novel_forbidden_score_headers,
                     novel_final_call_headers, classical_motif_headers, classical_score_headers)

    return results_tuple

def score_proteins(protein_seqs_df, predictor_params = predictor_params):
    '''
    Upper level function to score protein sequences in parallel based on conditional matrices

    Args:
        protein_seqs_df (pd.DataFrame):   protein sequences dataframe
        predictor_params (dict):          dictionary of user-defined parameters from predictor_config.py

    Returns:
        final_results (tuple): tuple of (scored_protein_df, novel_motif_cols, novel_total_score_cols,
                               novel_positive_score_cols, novel_suboptimal_score_cols, novel_forbidden_score_cols,
                               classical_motif_cols, classical_score_cols)
    '''

    chunk_size = predictor_params["chunk_size"]
    df_chunks = [protein_seqs_df.iloc[i:i + chunk_size] for i in range(0, len(protein_seqs_df), chunk_size)]

    score_chunk_partial = partial(score_proteins_chunk, predictor_params = predictor_params)

    pool = multiprocessing.Pool()
    scored_chunks = []
    novel_motif_cols, novel_score_cols, classical_motif_cols, classical_score_cols = [], [], [], []

    with trange(len(df_chunks), desc="\tScoring proteins...") as pbar:
        for results in pool.imap(score_chunk_partial, df_chunks):
            df_chunk_scored = results[0]
            scored_chunks.append(df_chunk_scored)

            novel_motif_cols, novel_total_score_cols, novel_binding_score_cols = results[1:4]
            novel_positive_score_cols, novel_suboptimal_score_cols, novel_forbidden_score_cols = results[4:7]
            novel_final_call_cols = results[7]
            classical_motif_cols, classical_score_cols = results[8:]

            pbar.update()

    pool.close()
    pool.join()

    scored_protein_df = pd.concat(scored_chunks, ignore_index=True)

    final_results = (scored_protein_df, novel_motif_cols, novel_total_score_cols, novel_binding_score_cols,
                     novel_positive_score_cols, novel_suboptimal_score_cols, novel_forbidden_score_cols,
                     novel_final_call_cols,
                     classical_motif_cols, classical_score_cols)

    return final_results