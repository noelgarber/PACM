#This script takes protein sequences and computes their motif scores based on the results of make_pairwise_matrices.py

import numpy as np
import pandas as pd
import pickle
import multiprocessing
import concurrent.futures
from tqdm import trange
from functools import partial
from Matrix_Generator.ConditionalMatrix import ConditionalMatrices

# Import the user-specified params, either from a local version or the git-linked version
try:
    from Motif_Predictor.predictor_config_local import predictor_params
except:
    from Motif_Predictor.predictor_config import predictor_params

# If selected, import a parallel method for comparison
if predictor_params["compare_classical_method"]:
    from Motif_Predictor.classical_method import classical_motif_method

def score_motifs(seqs_2d, conditional_matrices, score_addition_method, filters = None,
                 selenocysteine_substitute = "C", gap_substitute = "G", classical_func = None):
    '''
    Vectorized function to score homolog motif seqs based on the dictionary of context-aware weighted matrices

    Args:
        seqs_2d (np.ndarray):                       motif sequences to score
        conditional_matrices (ConditionalMatrices): conditional weighted matrices for scoring peptides
        score_addition_method (str):                matches matrix_params["optimization_method"]
        filters (dict):                             dict of position index --> permitted residues
        selenocysteine_substitute (str):            letter to substitute for selenocysteine (U) when U is not in model
        gap_substitute (str):                       the letter to treat gaps ("X") as; default is no side chain, i.e. G
        classical_func (function|partial):          optional function for comparing an existing classical method

    Returns:
        results (tuple):   tuple of (total_scores, positive_scores, suboptimal_scores, forbidden_scores, final_calls)
    '''

    # Substitutions for disallowed residues that are not part of the model architecture
    if isinstance(selenocysteine_substitute, str):
        seqs_2d[seqs_2d == "U"] = selenocysteine_substitute
    if isinstance(gap_substitute, str):
        seqs_2d[seqs_2d == "X"] = gap_substitute

    # Calculate scores using conditional matrices
    scoring_results = conditional_matrices.score_seqs_2d(seqs_2d, use_weighted = True)
    binding_scores_2d, positive_scores_2d, suboptimal_scores_2d, forbidden_scores_2d = scoring_results

    forbidden_scores_2d[:,conditional_matrices.suppress_forbidden_positions] = 0
    disqualified_forbidden = np.any(forbidden_scores_2d > 0, axis=1)

    binding_weighted_scores = binding_scores_2d.sum(axis=1)
    del binding_scores_2d
    positive_weighted_scores = positive_scores_2d.sum(axis=1)
    del positive_scores_2d
    suboptimal_weighted_scores = suboptimal_scores_2d.sum(axis=1)
    del suboptimal_scores_2d
    forbidden_scores = forbidden_scores_2d.sum(axis=1)
    del forbidden_scores_2d

    # Calculate total scores
    if score_addition_method == "ps":
        total_scores = positive_weighted_scores - suboptimal_weighted_scores
    elif score_addition_method == "wps":
        total_scores = binding_weighted_scores - suboptimal_weighted_scores
    elif score_addition_method == "suboptimal":
        total_scores = suboptimal_weighted_scores * -1
    else:
        raise ValueError(f"conditional_matrices.best_accuracy_method is {score_addition_method}")

    # Standardization of the scores
    binding_std_coefs = conditional_matrices.binding_standardization_coefficients
    if binding_std_coefs is not None:
        binding_weighted_scores = (binding_weighted_scores - binding_std_coefs[0]) / binding_std_coefs[1]

    std_coefs = conditional_matrices.classification_standardization_coefficients
    if std_coefs is not None:
        total_scores = (total_scores - std_coefs[0]) / std_coefs[1]
        positive_weighted_scores = (positive_weighted_scores - std_coefs[2]) / std_coefs[3]
        suboptimal_weighted_scores = (suboptimal_weighted_scores - std_coefs[4]) / std_coefs[5]

    # Get binary predictions
    total_scores[disqualified_forbidden] = np.nan
    threshold = conditional_matrices.standardized_weighted_threshold
    predicted_calls = np.greater_equal(total_scores, threshold)

    # Apply filters
    if isinstance(filters, dict):
        for i, allowed_residues in filters.items():
            seqs_pass = np.full(shape=seqs_2d.shape[0], fill_value=False, dtype=bool)
            for allowed_aa in allowed_residues:
                seqs_pass = np.logical_or(seqs_pass, np.char.equal(seqs_2d[:,i], allowed_aa))
            total_scores[~seqs_pass] = 0

    # Compare optional classical method function
    if classical_func is not None:
        classical_points_vals = classical_func(seqs_2d)
    else:
        classical_points_vals = None

    motifs = ["".join(seqs_2d[i]) for i in np.arange(len(seqs_2d))]
    results = (motifs, total_scores, binding_weighted_scores, positive_weighted_scores,
               suboptimal_weighted_scores, forbidden_scores, predicted_calls, classical_points_vals)

    return results

def seqs_chunk_generator(seqs_2d, chunk_size):
    # Chunk generator; saves memory rather than loading list all at once
    for i in range(0, len(seqs_2d), chunk_size):
        yield seqs_2d[i:i+chunk_size]

def score_motifs_parallel(seqs_2d, conditional_matrices, score_addition_method, chunk_size = 10000, filters = None,
                          selenocysteine_substitute = "C", gap_substitute = "G", classical_func = None):
    '''
    Parallelized function for scoring sequences using a ConditionalMatrices object

        seqs_2d (np.ndarray):                       motif sequences to score
        conditional_matrices (ConditionalMatrices): conditional weighted matrices for scoring peptides
        score_addition_method (str):                matches matrix_params["optimization_method"]
        chunk_size (int):                           number of sequences per parallel processing chunk
        filters (dict):                             dict of position index --> permitted residues
        selenocysteine_substitute (str):            letter to substitute for selenocysteine (U) when U is not in model
        gap_substitute (str):                       the letter to treat gaps ("X") as; default is no side chain, i.e. G
        classical_func (function|partial):          optional function for comparing an existing classical method

    Returns:
        total_scores (list):                        list of matching scores for each motif
    '''

    partial_function = partial(score_motifs, conditional_matrices = conditional_matrices,
                               score_addition_method = score_addition_method, filters = filters,
                               selenocysteine_substitute = selenocysteine_substitute, gap_substitute = gap_substitute,
                               classical_func = classical_func)

    chunk_motifs = []
    chunk_total_scores = []
    chunk_binding_scores = []
    chunk_positive_scores = []
    chunk_suboptimal_scores = []
    chunk_forbidden_scores = []
    chunk_final_calls = []
    chunk_classical_scores = []

    description = f"\tScoring {len(seqs_2d)} unique homologous motifs..."
    with trange(int(np.ceil(len(seqs_2d) / chunk_size) + 1), desc=description) as pbar:
        pool = multiprocessing.Pool()

        for results in pool.imap(partial_function, seqs_chunk_generator(seqs_2d, chunk_size)):
            chunk_motifs.append(results[0])
            chunk_total_scores.append(results[1])
            chunk_binding_scores.append(results[2])
            chunk_positive_scores.append(results[3])
            chunk_suboptimal_scores.append(results[4])
            chunk_forbidden_scores.append(results[5])
            chunk_final_calls.append(results[6])
            if results[7] is not None:
                chunk_classical_scores.append(results[7])

            pbar.update()

        pool.close()
        pool.join()

        chunk_motifs = np.concatenate(chunk_motifs)
        chunk_total_scores = np.concatenate(chunk_total_scores)
        chunk_binding_scores = np.concatenate(chunk_binding_scores)
        chunk_positive_scores = np.concatenate(chunk_positive_scores)
        chunk_suboptimal_scores = np.concatenate(chunk_suboptimal_scores)
        chunk_forbidden_scores = np.concatenate(chunk_forbidden_scores)
        chunk_final_calls = np.concatenate(chunk_final_calls)
        if len(chunk_classical_scores) > 0:
            chunk_classical_scores = np.concatenate(chunk_classical_scores)

        pbar.update()

    output = (chunk_motifs, chunk_total_scores, chunk_binding_scores, chunk_positive_scores,
              chunk_suboptimal_scores, chunk_forbidden_scores, chunk_final_calls, chunk_classical_scores)

    return output

def process_grid(grid, lookup_dict, dtype):
    return grid.applymap(lambda x: lookup_dict.get(x)).to_numpy(dtype=dtype)

def get_valid_mask(homolog_motifs_grid, filters):
    # Helper function to get homolog grid validity mask based on filters dict

    if isinstance(homolog_motifs_grid, pd.Series) or isinstance(homolog_motifs_grid, pd.DataFrame):
        homolog_motifs_grid = homolog_motifs_grid.to_numpy(dtype="U")

    motif_len = None
    for motif in homolog_motifs_grid.flatten():
        if motif != "":
            motif_len = len(motif)
            break

    homolog_motifs_grid[homolog_motifs_grid == ""] = "".join(np.repeat("Z",motif_len))

    grid_3d_shape = [homolog_motifs_grid.shape[0], homolog_motifs_grid.shape[1], motif_len]
    homolog_grid_3d = np.frombuffer(homolog_motifs_grid.astype(np.unicode_).tobytes(), dtype=np.uint32).reshape(grid_3d_shape)
    homolog_grid_3d = np.vectorize(chr)(homolog_grid_3d).astype("<U1")

    homolog_valid_grid = np.full(shape=homolog_motifs_grid.shape, fill_value=True, dtype=bool)
    for idx, allowed_residues in filters.items():
        grid_at_idx = homolog_grid_3d[:, :, idx]
        grid_at_idx_allowed = np.isin(grid_at_idx, allowed_residues)
        grid_at_idx_disallowed = ~grid_at_idx_allowed
        homolog_valid_grid[grid_at_idx_disallowed] = False

    return homolog_valid_grid

def get_best_cols(selection_grid, homolog_valid_grid):
    # Helper function to get best column indices for homolog motifs from a grid of reference values

    masked_selection_grid = selection_grid.copy()
    masked_selection_grid[~homolog_valid_grid] = -1
    masked_best_selection_vals = np.nanmax(masked_selection_grid, axis=1)
    masked_best_col_indices = np.nanargmax(masked_selection_grid, axis=1)
    if np.any(masked_best_selection_vals == -1):
        unmasked_best_col_indices = np.nanargmax(selection_grid, axis=1)
        best_col_indices = masked_best_col_indices.copy()
        masked_best_invalid = np.equal(masked_best_selection_vals, -1)
        best_col_indices[masked_best_invalid] = unmasked_best_col_indices[masked_best_invalid]
    else:
        best_col_indices = masked_best_col_indices

    return best_col_indices

def score_homolog_motifs(data_df, homolog_motif_cols, homolog_motif_col_groups, predictor_params):
    '''
    Main function for scoring homologous motifs

    Args:
        data_df (pd.DataFrame):          main dataframe with motif sequences for host and homologs
        homolog_motif_cols (list|tuple): col names where homolog motif sequences are stored
		homolog_motif_col_groups (dict): dict of host motif seq col --> grouped column names for each homologous motif
        predictor_params (dict):         dictionary of parameters for scoring

    Returns:
        data_df (pd.DataFrame):     dataframe with scores added for homolog motifs
        homolog_id_cols (list):     shortened list of col names where homolog ids are stored
        homolog_motif_cols (list):  shortened list of col names containing homologous motifs
        model_score_cols (list):    shortened list of col names containing homologous motif scores according to model
    '''

    verbose = predictor_params["homolog_scoring_verbose"]

    # Load ConditionalMatrices object to be used in scoring
    conditional_matrices_path = predictor_params["conditional_matrices_path"]
    with open(conditional_matrices_path, "rb") as f:
        conditional_matrices = pickle.load(f)

    # Extract all unique motif sequences for scoring
    print(f"\t\tGetting unique motif sequences...") if verbose else None
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
    filters = predictor_params["enforced_position_rules"]
    selenocysteine_substitute = predictor_params["selenocysteine_substitute"]
    gap_substitute = predictor_params["gap_substitute"]
    chunk_size = predictor_params["homolog_score_chunk_size"]
    score_addition_method = predictor_params["score_addition_method"]
    compare_classical_method = predictor_params["compare_classical_method"]
    classical_func = classical_motif_method if compare_classical_method else None

    results = score_motifs_parallel(motif_seqs_2d, conditional_matrices, score_addition_method, chunk_size,
                                    filters, selenocysteine_substitute, gap_substitute, classical_func)

    print(f"\t\tParsing results into motif-score dicts...") if verbose else None
    motifs, total_scores, binding_scores = results[0:3]
    positive_scores, suboptimal_scores, forbidden_scores, final_calls = results[3:7]
    classical_scores = results[7]
    if len(classical_scores) == len(total_scores):
        zipped_results = zip(motifs, total_scores, binding_scores, positive_scores,
                             suboptimal_scores, forbidden_scores, final_calls, classical_scores)
    else:
        zipped_results = zip(motifs, total_scores, binding_scores, positive_scores, suboptimal_scores, forbidden_scores,
                             final_calls, np.full(shape=len(total_scores), fill_value=np.nan))

    total_dict = {}
    binding_dict = {}
    positive_dict = {}
    suboptimal_dict = {}
    forbidden_dict = {}
    calls_dict = {}
    classical_dict = {}
    combined_dict = {}
    for motif, total_score, binding, positive, suboptimal, forbidden, call, classical in zipped_results:
        total_dict[motif] = total_score
        binding_dict[motif] = binding
        positive_dict[motif] = positive
        suboptimal_dict[motif] = suboptimal
        forbidden_dict[motif] = forbidden
        calls_dict[motif] = call
        classical_dict[motif] = classical
        combined_dict[motif] = (total_score, positive, suboptimal, forbidden, call, classical)

    # Iterate over homolog motif col groups, organized by host motif col
    row_indices = np.arange(len(data_df))
    final_homolog_motif_cols = []
    final_call_cols = []
    drop_cols = []
    selection_mode = predictor_params["homolog_selection_mode"]
    description = "\tAssigning best homologous motifs to dataframe and removing others..."
    with trange(int(21*len(homolog_motif_col_groups)), desc=description) as pbar:
        # Get the grids of scores from the dataframe
        for motif_seq_col, col_groups in homolog_motif_col_groups.items():
            homolog_motif_cols = [col_group[0] for col_group in col_groups]
            similarity_cols = [col_group[1] for col_group in col_groups]
            identity_cols = [col_group[2] for col_group in col_groups]
            homolog_id_cols = [homolog_motif_col.split("_vs_")[0] for homolog_motif_col in homolog_motif_cols]

            # Extract homologous motifs from dataframe
            homolog_motifs_grid = data_df[homolog_motif_cols].copy()
            data_df.drop(homolog_motif_cols, axis=1, inplace=True)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(homolog_motifs_grid.applymap, lambda x: total_dict.get(x)),
                    executor.submit(homolog_motifs_grid.applymap, lambda x: binding_dict.get(x)),
                    executor.submit(homolog_motifs_grid.applymap, lambda x: positive_dict.get(x)),
                    executor.submit(homolog_motifs_grid.applymap, lambda x: suboptimal_dict.get(x)),
                    executor.submit(homolog_motifs_grid.applymap, lambda x: forbidden_dict.get(x)),
                    executor.submit(homolog_motifs_grid.applymap, lambda x: calls_dict.get(x)),
                    executor.submit(homolog_motifs_grid.applymap, lambda x: classical_dict.get(x)),
                ]

                results = []
                for f in futures:
                    results.append(f.result())
                    pbar.update()

                total_scores_grid = results[0].to_numpy(dtype=float)
                binding_scores_grid = results[1].to_numpy(dtype=float)
                positive_scores_grid = results[2].to_numpy(dtype=float)
                suboptimal_scores_grid = results[3].to_numpy(dtype=float)
                forbidden_scores_grid = results[4].to_numpy(dtype=float)
                final_calls_grid = results[5].to_numpy(dtype=bool)
                classical_scores_grid = results[6].to_numpy(dtype=float)
                del results

                pbar.update()

            # Find best col indices for best homologous motifs
            homolog_valid_grid = get_valid_mask(homolog_motifs_grid, filters)

            if selection_mode == "similarity":
                similarities_grid = data_df[similarity_cols].to_numpy(dtype=float)
                best_col_indices = get_best_cols(similarities_grid, homolog_valid_grid)
            elif selection_mode == "identity":
                identities_grid = data_df[identity_cols].to_numpy(dtype=float)
                best_col_indices = get_best_cols(identities_grid, homolog_valid_grid)
            elif selection_mode == "score":
                best_col_indices = get_best_cols(total_scores_grid, homolog_valid_grid)
            else:
                message = f"mode was set to {selection_mode}, but must be identity, similarity, or score"
                raise ValueError(message)

            pbar.update()

            # Find the col prefix to use for best homologous motifs
            homolog_col_element, source_col_element = homolog_motif_cols[0].split("_vs_")
            homolog_col_element = homolog_col_element.rsplit("_", 1)[0]
            source_col_element = source_col_element.split("_matching_motif")[0]
            col_prefix = f"{homolog_col_element}_vs_{source_col_element}"
            pbar.update()

            homolog_ids_grid = data_df[homolog_id_cols].to_numpy(dtype="U")
            drop_cols.extend(homolog_id_cols)
            best_homolog_ids = homolog_ids_grid[row_indices, best_col_indices]
            data_df[col_prefix + "_id_best"] = best_homolog_ids
            del homolog_ids_grid, best_homolog_ids
            pbar.update()

            best_homolog_motifs = homolog_motifs_grid.values[row_indices, best_col_indices]
            data_df[col_prefix + "_best"] = best_homolog_motifs
            del homolog_motifs_grid, best_homolog_motifs
            pbar.update()

            similarities_grid = data_df[similarity_cols].to_numpy(dtype=float)
            data_df.drop(similarity_cols, axis=1, inplace=True)
            best_similarities = similarities_grid[row_indices, best_col_indices]
            data_df[col_prefix + "_similarity_best"] = best_similarities
            del similarities_grid, best_similarities
            pbar.update()

            identities_grid = data_df[identity_cols].to_numpy(dtype=float)
            data_df.drop(identity_cols, axis=1, inplace=True)
            best_identities = identities_grid[row_indices, best_col_indices]
            data_df[col_prefix + "_identity_best"] = best_identities
            final_homolog_motif_cols.append(col_prefix + "_best")
            del identities_grid, best_identities
            pbar.update()

            best_binding_scores = binding_scores_grid[row_indices, best_col_indices]
            data_df[col_prefix + "_best_binding_model_score"] = best_binding_scores
            del binding_scores_grid, best_binding_scores
            pbar.update()

            best_positive_scores = positive_scores_grid[row_indices, best_col_indices]
            data_df[col_prefix + "_best_positive_model_score"] = best_positive_scores
            del positive_scores_grid, best_positive_scores
            pbar.update()

            best_suboptimal_scores = suboptimal_scores_grid[row_indices, best_col_indices]
            data_df[col_prefix + "_best_suboptimal_model_score"] = best_suboptimal_scores
            del suboptimal_scores_grid, best_suboptimal_scores
            pbar.update()

            best_forbidden_scores = forbidden_scores_grid[row_indices, best_col_indices]
            data_df[col_prefix + "_best_forbidden_model_score"] = best_forbidden_scores
            del forbidden_scores_grid, best_forbidden_scores
            pbar.update()

            best_total_scores = total_scores_grid[row_indices, best_col_indices]
            data_df[col_prefix + "_best_total_model_score"] = best_total_scores
            del total_scores_grid, best_total_scores
            pbar.update()

            best_calls = final_calls_grid[row_indices, best_col_indices]
            data_df[col_prefix + "_best_model_call"] = best_calls
            final_call_cols.append(col_prefix + "_best_model_call")
            del final_calls_grid, best_calls
            pbar.update()

            if len(classical_scores) == len(total_scores):
                best_classical_scores = classical_scores_grid[row_indices, best_col_indices]
                data_df[col_prefix + "_classical_score"] = best_classical_scores
                del classical_scores_grid
            pbar.update()

    drop_cols = list(set(drop_cols))
    data_df.drop(drop_cols, axis=1, inplace=True)

    # Apply classical method if necessary
    compare_classical_method = predictor_params["compare_classical_method"]
    if compare_classical_method:
        cols = data_df.columns.copy()
        description = "\tAdding classical motif scores to best homologous motifs"
        with trange(len(final_homolog_motif_cols) + 1, desc=description) as pbar:
            for homolog_motif_col, final_call_col in zip(final_homolog_motif_cols, final_call_cols):
                insertion_idx = cols.get_loc(final_call_col) + 1
                classical_score_col = f"{homolog_motif_col}_classical_score"
                cols.insert(insertion_idx, classical_score_col)

                motif_seqs = data_df[homolog_motif_col].to_list()
                classical_motif_scores = classical_motif_method(motif_seqs)
                data_df[classical_score_col] = classical_motif_scores

                pbar.update()

            data_df = data_df[cols]
            pbar.update()

    return data_df, final_homolog_motif_cols