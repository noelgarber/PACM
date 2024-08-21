#This script takes protein sequences and computes their motif scores based on the results of make_pairwise_matrices.py

import numpy as np
import pandas as pd
import os
import hashlib
import json
import pickle
import multiprocessing
import concurrent.futures
import warnings
from tqdm import trange
from functools import partial
from Matrix_Generator.ConditionalMatrix import ConditionalMatrices
from Motif_Predictor.load_predictor_config import load_config
from Motif_Predictor.map_homologies import map_homologies

predictor_params = load_config()

# If selected, import a parallel method for comparison
if predictor_params["compare_classical_method"]:
    from Motif_Predictor.classical_method import classical_motif_method

def score_motifs(seqs_2d, conditional_matrices, score_addition_method, enforced_position_rules = None,
                 selenocysteine_substitute = "C", gap_substitute = "G", classical_func = None):
    '''
    Vectorized function to score homolog motif seqs based on the dictionary of context-aware weighted matrices

    Args:
        seqs_2d (np.ndarray):                       motif sequences to score
        conditional_matrices (ConditionalMatrices): conditional weighted matrices for scoring peptides
        score_addition_method (str):                matches matrix_params["optimization_method"]
        enforced_position_rules (dict):             dict of position index --> permitted residues
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

    # Compare optional classical method function
    classical_points_vals = classical_func(seqs_2d) if classical_func is not None else None

    # Apply filters
    if enforced_position_rules is not None:
        for position_index, allowed_residues in enforced_position_rules.items():
            column_residues = seqs_2d[:, position_index]
            residues_allowed = np.isin(column_residues, allowed_residues)
            total_scores[~residues_allowed] = np.nan
            binding_weighted_scores[~residues_allowed] = np.nan
            positive_weighted_scores[~residues_allowed] = np.nan
            suboptimal_weighted_scores[~residues_allowed] = np.nan
            forbidden_scores[~residues_allowed] = np.nan
            predicted_calls[~residues_allowed] = False
            if classical_func is not None:
                classical_points_vals[~residues_allowed] = np.nan

    motifs = ["".join(seqs_2d[i]) for i in np.arange(len(seqs_2d))]
    results = (motifs, total_scores, binding_weighted_scores, positive_weighted_scores,
               suboptimal_weighted_scores, forbidden_scores, predicted_calls, classical_points_vals)

    return results

def seqs_chunk_generator(seqs_2d, chunk_size):
    # Chunk generator; saves memory rather than loading list all at once
    for i in range(0, len(seqs_2d), chunk_size):
        yield seqs_2d[i:i+chunk_size]

def score_motifs_parallel(seqs_2d, conditional_matrices, predictor_params):
    '''
    Parallelized function for scoring sequences using a ConditionalMatrices object

        seqs_2d (np.ndarray):                       motif sequences to score
        conditional_matrices (ConditionalMatrices): conditional weighted matrices for scoring peptides
        predictor_params (dict):                    main dict of params

    Returns:
        total_scores (list):                        list of matching scores for each motif
    '''

    verbose = predictor_params["homology_params"]["homolog_scoring_verbose"]
    print(f"\t\tScoring unique motif sequences...") if verbose else None

    enforced_position_rules = predictor_params["enforced_position_rules"]
    selenocysteine_substitute = predictor_params["selenocysteine_substitute"]
    gap_substitute = predictor_params["gap_substitute"]
    chunk_size = predictor_params["homology_params"]["homolog_score_chunk_size"]
    score_addition_method = predictor_params["score_addition_method"]
    compare_classical_method = predictor_params["compare_classical_method"]
    classical_func = classical_motif_method if compare_classical_method else None

    partial_function = partial(score_motifs, conditional_matrices = conditional_matrices,
                               score_addition_method = score_addition_method,
                               enforced_position_rules = enforced_position_rules,
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

def get_valid_mask(homolog_motifs_grid, enforced_position_rules):
    # Helper function to get homolog grid validity mask based on filters dict

    if isinstance(homolog_motifs_grid, pd.Series) or isinstance(homolog_motifs_grid, pd.DataFrame):
        homolog_motifs_grid = homolog_motifs_grid.to_numpy(dtype="U")

    motif_len = None
    for motif in homolog_motifs_grid.flatten():
        if motif != "":
            motif_len = len(motif)
            break

    grid_copy = homolog_motifs_grid.copy()
    grid_copy[grid_copy == ""] = "".join(np.repeat("Z",motif_len))

    grid_3d_shape = [grid_copy.shape[0], grid_copy.shape[1], motif_len]
    homolog_grid_3d = np.frombuffer(grid_copy.astype(np.unicode_).tobytes(), dtype=np.uint32).reshape(grid_3d_shape)
    homolog_grid_3d = np.vectorize(chr)(homolog_grid_3d).astype("<U1")

    homolog_valid_grid = np.full(shape=grid_copy.shape, fill_value=True, dtype=bool)
    for idx, allowed_residues in enforced_position_rules.items():
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

def apply_classical(data_df, final_homolog_motif_cols, final_call_cols):
    # Helper function for applying the classical algorithm

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

    return data_df

def extract_unique_motifs(data_df, homolog_motif_cols, verbose = False):
    # Helper function to obtain all unique short linear motifs to be scored upfront

    print(f"\t\tGetting unique motif sequences...") if verbose else None

    motif_seqs = []
    for homolog_motif_col in homolog_motif_cols:
        col_data = data_df[homolog_motif_col].copy()
        col_data = col_data[col_data.notna()]
        col_data = col_data[col_data.ne("")]
        motif_seqs.append(col_data.to_numpy())

    motif_seqs = np.unique(np.concatenate(motif_seqs))
    motif_seqs_2d = np.array([list(motif) for motif in motif_seqs])

    return motif_seqs, motif_seqs_2d

def parse_motif_dicts(results, verbose = False):
    # Helper function that parses scored results into dictionaries of unique motif sequences and their scores

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

    dicts_output = (total_dict, binding_dict, positive_dict, suboptimal_dict, forbidden_dict,
                    calls_dict, classical_dict, combined_dict)

    return dicts_output

def generate_grids(homolog_motifs_grid, total_dict, binding_dict, positive_dict, suboptimal_dict, forbidden_dict,
                   calls_dict, classical_dict, pbar = None):
    # Helper function that parallelizes the process of generating score grids for homolog motifs; requires 7 threads

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
            if pbar is not None:
                pbar.update()

        total_scores_grid = results[0].to_numpy(dtype=float)
        binding_scores_grid = results[1].to_numpy(dtype=float)
        positive_scores_grid = results[2].to_numpy(dtype=float)
        suboptimal_scores_grid = results[3].to_numpy(dtype=float)
        forbidden_scores_grid = results[4].to_numpy(dtype=float)
        final_calls_grid = results[5].to_numpy(dtype=bool)
        classical_scores_grid = results[6].to_numpy(dtype=float)
        del results

        if pbar is not None:
            pbar.update()

    grids = [total_scores_grid, binding_scores_grid, positive_scores_grid, suboptimal_scores_grid,
             forbidden_scores_grid, final_calls_grid, classical_scores_grid]

    return grids

def score_similar_homologs(data_df, homolog_motif_col_groups, total_dict, binding_dict, positive_dict,
                           suboptimal_dict, forbidden_dict, calls_dict, classical_dict, predictor_params):
    '''
    Function for finding highly similar motifs in homologs compared to a reference species

    Args:
        data_df (pd.DataFrame):          main dataframe with motif sequences for host and homologs
        homolog_motif_col_groups (dict): dict of host motif seq col --> grouped column names for each homologous motif
        total_dict (dict):               dict of motif sequence --> total classification score
        binding_dict (dict):             dict of motif sequence --> binding score
        positive_dict (dict):            dict of motif sequence --> positive classification score
        suboptimal_dict (dict):          dict of motif sequence --> suboptimal classification score
        forbidden_dict (dict):           dict of motif sequence --> forbidden residue count score
        calls_dict (dict):               dict of motif sequence --> boolean call
        classical_dict (dict):           dict of motif sequence --> classical score
        predictor_params (dict):         dictionary of parameters for scoring

    Returns:
        data_df (pd.DataFrame):          dataframe with scores added for homolog motifs
        final_homolog_motif_cols (list): list of homolog motif column names
    '''

    # Iterate over homolog motif col groups, organized by host motif col
    row_indices = np.arange(len(data_df))
    final_homolog_motif_cols = []
    final_call_cols = []
    drop_cols = []
    selection_mode = predictor_params["homology_params"]["homolog_selection_mode"]
    description = "\tAssigning best homologous motifs to dataframe and removing others..."
    with trange(int(21 * len(homolog_motif_col_groups)), desc=description) as pbar:
        # Get the grids of scores from the dataframe
        for motif_seq_col, col_groups in homolog_motif_col_groups.items():
            homolog_motif_cols = [col_group[0] for col_group in col_groups]
            similarity_cols = [col_group[1] for col_group in col_groups]
            identity_cols = [col_group[2] for col_group in col_groups]
            homolog_id_cols = [homolog_motif_col.split("_vs_")[0] for homolog_motif_col in homolog_motif_cols]

            # Extract homologous motifs from dataframe
            homolog_motifs_grid = data_df[homolog_motif_cols].copy()
            data_df.drop(homolog_motif_cols, axis=1, inplace=True)

            # Generate score grids
            grids = generate_grids(homolog_motifs_grid, total_dict, binding_dict, positive_dict, suboptimal_dict,
                                   forbidden_dict, calls_dict, classical_dict, pbar)
            total_scores_grid, binding_scores_grid, positive_scores_grid, suboptimal_scores_grid = grids[:4]
            forbidden_scores_grid, final_calls_grid, classical_scores_grid = grids[4:]

            # Find best col indices for best homologous motifs
            homolog_valid_grid = get_valid_mask(homolog_motifs_grid, predictor_params["enforced_position_rules"])

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
            data_df[col_prefix + "_id_best"] = data_df[col_prefix + "_id_best"].fillna("")
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

    return data_df, final_homolog_motif_cols, final_call_cols

def organize_targets_chunk(pair, ref_gene_col, motif_cols, binding_cols, classification_cols, call_cols,
                           classical_motif_cols = None, classical_score_cols = None):
    # Helper function for organizing a single target dataframe into a dictionary of gene names
    target_taxid, target_df = pair

    # Generate dictionaries for novel model scores
    target_taxid_novel_scores = {}
    novel_gene_entries = {}
    for i in np.arange(len(target_df)):
        gene_id = target_df.at[i, ref_gene_col]
        best_masked_score = target_df.at[i, binding_cols[0]] if target_df.at[i, call_cols[0]] else 0
        target_taxid_novel_scores[gene_id] = best_masked_score
        entry = []
        zipped_cols = zip(motif_cols, binding_cols, classification_cols, call_cols)
        for motif_col, binding_col, classification_col, call_col in zipped_cols:
            motif = target_df.at[i, motif_col]
            classification_score = target_df.at[i, classification_col]
            binding_score = target_df.at[i, binding_col]
            call = target_df.at[i, call_col]
            masked_binding_score = binding_score if call else 0
            entry.append((motif, classification_score, binding_score, masked_binding_score, call))
        novel_gene_entries[gene_id] = entry

    # Generate dictionaries for classical model scores if another model is being compared; skipped by default
    if classical_motif_cols is not None and classical_score_cols is not None:
        target_taxid_classical_scores = {}
        classical_gene_entries = {}
        for i in np.arange(len(target_df)):
            gene_id = target_df.at[i, ref_gene_col]
            target_taxid_classical_scores[gene_id] = target_df.at[i, classical_score_cols[0]]
            entry = []
            for classical_motif_col, classical_score_col in zip(classical_motif_cols, classical_score_cols):
                motif = target_df.at[i, classical_motif_col]
                classical_score = target_df.at[i, classical_score_col]
                entry.append((motif, classical_score))
            classical_gene_entries[gene_id] = entry
    elif classical_motif_cols is not None:
        raise ValueError(f"classical_score_cols is required when classical_motif_cols is not None")
    elif classical_score_cols is not None:
        raise ValueError(f"classical_motif_cols is required when classical_score_cols is not None")
    else:
        target_taxid_classical_scores, classical_gene_entries = None, None

    output = (target_taxid, target_taxid_novel_scores, novel_gene_entries,
              target_taxid_classical_scores, classical_gene_entries)

    return output

def organize_targets(target_taxids, target_dfs, ref_gene_col, motif_cols, binding_cols, classification_cols, call_cols,
                     classical_motif_cols = None, classical_score_cols = None):
    # Helper function that organizes target dataframes into dictionaries of gene names

    target_novel_scores = {}
    target_novel_genes = {}
    target_classical_scores = {}
    target_classical_genes = {}

    func = partial(organize_targets_chunk, ref_gene_col = ref_gene_col, motif_cols = motif_cols,
                   binding_cols = binding_cols, classification_cols = classification_cols, call_cols = call_cols,
                   classical_motif_cols = classical_motif_cols, classical_score_cols = classical_score_cols)

    taxid_df_pairs = [(target_taxid, target_df) for target_taxid, target_df in zip(target_taxids, target_dfs)]

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpu_count-1)
    for result in pool.imap(func, taxid_df_pairs):
        target_taxid, target_taxid_novel_scores, novel_gene_entries = result[:3]
        target_taxid_classical_scores, classical_gene_entries = result[3:]
        target_novel_scores[target_taxid] = target_taxid_novel_scores
        target_novel_genes[target_taxid] = novel_gene_entries
        if target_taxid_classical_scores is not None and classical_gene_entries is not None:
            target_classical_scores[target_taxid] = target_taxid_classical_scores
            target_classical_genes[target_taxid] = classical_gene_entries

    pool.close()
    pool.join()

    if target_classical_scores:
        return (target_novel_scores, target_novel_genes, target_classical_scores, target_classical_genes)
    else:
        return (target_novel_scores, target_novel_genes)

def get_best_homologs(reference_df, ref_gene_col, target_dicts, target_score_dict, invert = False):
    # Helper function to make a dict of best homologs by taxid for each reference gene, based on masked binding score

    best_homolog_dict = {}
    for taxid, reference_target_homologs in target_dicts.items():
        best_homolog_dict[taxid] = {}
        if reference_target_homologs is None:
            warnings.warn(f"No reference_target_homologs dict was given for taxid {taxid}")
        else:
            for ref_gene_id in reference_df[ref_gene_col]:
                homolog_ids = reference_target_homologs.get(ref_gene_id)
                if homolog_ids is None:
                    best_homolog_dict[taxid][ref_gene_id] = None
                else:
                    homolog_masked_scores = []
                    for homolog_id in homolog_ids:
                        score = target_score_dict.get(homolog_id)
                        score = score if score is not None else -np.inf
                        score = score if not invert else -score
                        homolog_masked_scores.append(score)
                    best_homolog_idx = np.nanargmax(homolog_masked_scores)
                    best_homolog_id = homolog_ids[best_homolog_idx]
                    best_homolog_dict[taxid][ref_gene_id] = best_homolog_id

    return best_homolog_dict

def find_passing_homolog_motif(row_idx, entry, motif_col, reference_df, identity_thres, motif_length):
    # Helper function that tries to find a homologous motif that passes

    best_homolog_motif = " " * motif_length
    best_homolog_classification, best_homolog_binding, best_homolog_masked, best_homolog_call = 0, 0, 0, False
    identity = 0

    for homolog_motif, homolog_classification_score, homolog_binding_score, homolog_masked_score, homolog_call in entry:
        if homolog_call:
            ref_motif = reference_df.at[row_idx, motif_col]

            if isinstance(ref_motif, str) and isinstance(homolog_motif, str):
                identity_arr = np.equal(np.array(ref_motif), np.array(homolog_motif))
                current_identity = identity_arr.mean()
                if current_identity > 0 and current_identity < 1:
                    print(f"find_passing_homolog_motif(): Identity between {ref_motif} and {homolog_motif}: {current_identity}")
            else:
                current_identity = 0

            if current_identity > identity and current_identity >= identity_thres:
                best_homolog_motif = homolog_motif
                identity = current_identity
                best_homolog_classification = homolog_classification_score
                best_homolog_binding = homolog_binding_score
                best_homolog_masked = homolog_masked_score
                best_homolog_call = homolog_call

    output = (best_homolog_motif, best_homolog_classification,
              best_homolog_binding, best_homolog_masked, best_homolog_call, identity)

    return output

def accept_dissimilar_motif(row_idx, entry, motif_col, best_homolog_motif, best_homolog_classification,
                            best_homolog_binding, best_homolog_masked, best_homolog_call, identity, reference_df):
    # If no similar motif could be found, accept a dissimilar one

    for homolog_motif, homolog_classification_score, homolog_binding_score, homolog_masked_score, homolog_call in entry:
        ref_motif = reference_df.at[row_idx, motif_col]

        if isinstance(ref_motif, str) and isinstance(homolog_motif, str):
            identity_arr = np.equal(np.array(ref_motif), np.array(homolog_motif))
            current_identity = identity_arr.mean()
            if current_identity > 0 and current_identity < 1:
                print(f"accept_dissimilar_motif(): Identity between {ref_motif} and {homolog_motif}: {current_identity}")
        else:
            current_identity = 0

        if homolog_masked_score > best_homolog_masked:
            best_homolog_motif = homolog_motif
            identity = current_identity
            best_homolog_classification = homolog_classification_score
            best_homolog_binding = homolog_binding_score
            best_homolog_masked = homolog_masked_score
            best_homolog_call = homolog_call

    output = (best_homolog_motif, best_homolog_classification, best_homolog_binding,
              best_homolog_masked, best_homolog_call, identity)

    return output

def construct_new_data_dict(target_taxid, motif_cols, final_homolog_motif_cols, final_homolog_call_cols,
                            final_homolog_classical_motif_cols = None, include_classical = True):
    # Construct a blank dict that will contain the new column data; will be faster than repeatedly calling df.at

    for motif_col in motif_cols:
        homolog_motif_col = f"{target_taxid}_{motif_col}_homolog"
        final_homolog_motif_cols.append(homolog_motif_col)
        final_homolog_call_cols.append(f"{homolog_motif_col}_call")
        if include_classical:
            classical_homolog_motif_col = homolog_motif_col.replace("Novel", "Classical")
            final_homolog_classical_motif_cols.append(classical_homolog_motif_col)

    new_data = {}
    new_data[f"{target_taxid}_best_homolog_id"] = []
    if include_classical:
        new_data[f"{target_taxid}_classical_best_homolog_id"] = []
    for motif_col in motif_cols:
        homolog_motif_col = f"{target_taxid}_{motif_col}_homolog"
        new_data[homolog_motif_col] = []
        new_data[f"{homolog_motif_col}_identity"] = []
        new_data[f"{homolog_motif_col}_classification_score"] = []
        new_data[f"{homolog_motif_col}_binding_score"] = []
        new_data[f"{homolog_motif_col}_masked_binding_score"] = []
        new_data[f"{homolog_motif_col}_call"] = []

        if include_classical:
            classical_homolog_motif_col = homolog_motif_col.replace("Novel", "Classical")
            new_data[classical_homolog_motif_col] = []
            new_data[f"{classical_homolog_motif_col}_identity"] = []
            new_data[f"{classical_homolog_motif_col}_score"] = []

    return new_data

def assign_novel_data(new_data, best_homolog_dict, target_gene_dict, motif_cols, target_taxid, reference_df,
                      ref_gene_col, identity_thres, motif_length):
    # Iterate over the reference dataframe to find homologs for the current taxid
    for i in np.arange(len(reference_df)):
        ref_gene_id = reference_df.at[i, ref_gene_col]

        # Get results from novel model
        best_homolog_id = best_homolog_dict[target_taxid].get(ref_gene_id)
        if best_homolog_id:
            new_data[f"{target_taxid}_best_homolog_id"].append(best_homolog_id)
            entry = target_gene_dict[target_taxid].get(best_homolog_id)
            for motif_col in motif_cols:
                homolog_motif_col = f"{target_taxid}_{motif_col}_homolog"

                # Try to find a homologous motif that passes
                output = find_passing_homolog_motif(i, entry, motif_col, reference_df, identity_thres, motif_length)
                best_homolog_motif, best_homolog_classification, best_homolog_binding = output[:3]
                best_homolog_masked, best_homolog_call, identity = output[3:]

                # If no similar motif could be found, accept a dissimilar one
                if not best_homolog_call and identity == 0:
                    output = accept_dissimilar_motif(i, entry, motif_col, best_homolog_motif,
                                                     best_homolog_classification, best_homolog_binding,
                                                     best_homolog_masked, best_homolog_call, identity, reference_df)
                    best_homolog_motif, best_homolog_classification, best_homolog_binding = output[:3]
                    best_homolog_masked, best_homolog_call, identity = output[3:]

                # Assign data for current row at specified columns
                new_data[homolog_motif_col].append(best_homolog_motif)
                new_data[f"{homolog_motif_col}_identity"].append(identity)
                new_data[f"{homolog_motif_col}_classification_score"].append(best_homolog_classification)
                new_data[f"{homolog_motif_col}_binding_score"].append(best_homolog_binding)
                new_data[f"{homolog_motif_col}_masked_binding_score"].append(best_homolog_masked)
                new_data[f"{homolog_motif_col}_call"].append(best_homolog_call)
        else:
            print(f"\tNote: {best_homolog_id} could not be found in the gene_dict for {target_taxid}, "
                  f"so its values are nan") if best_homolog_id is not None else None
            new_data[f"{target_taxid}_best_homolog_id"].append("")
            for motif_col in motif_cols:
                homolog_motif_col = f"{target_taxid}_{motif_col}_homolog"
                new_data[homolog_motif_col].append(np.nan)
                new_data[f"{homolog_motif_col}_identity"].append(np.nan)
                new_data[f"{homolog_motif_col}_classification_score"].append(np.nan)
                new_data[f"{homolog_motif_col}_binding_score"].append(np.nan)
                new_data[f"{homolog_motif_col}_masked_binding_score"].append(np.nan)
                new_data[f"{homolog_motif_col}_call"].append(np.nan)

def assign_classical_data(new_data, best_classical_homolog_dict, target_classical_gene_dict, motif_cols, target_taxid,
                          reference_df, ref_gene_col, identity_thres, motif_length, classical_inverted):

    for i in np.arange(len(reference_df)):
        ref_gene_id = reference_df.at[i, ref_gene_col]
        best_classical_homolog_id = best_classical_homolog_dict[target_taxid].get(ref_gene_id)
        if best_classical_homolog_id:
            classical_entry = target_classical_gene_dict[target_taxid].get(best_classical_homolog_id)
        else:
            classical_entry = None

        new_data[f"{target_taxid}_classical_best_homolog_id"].append(best_classical_homolog_id)
        for motif_col in motif_cols:
            classical_motif_col = motif_col.replace("Novel", "Classical")
            homolog_motif_col = f"{target_taxid}_{classical_motif_col}_homolog"
            if best_classical_homolog_id:
                # Try to find a homologous motif that passes
                output = find_best_homolog_classical(i, classical_entry, classical_motif_col, reference_df,
                                                     identity_thres, motif_length)
                best_homolog_classical_motif, best_homolog_classical_score, classical_identity = output

                # If no similar motif could be found, accept a dissimilar one
                if classical_identity == 0:
                    output = accept_dissimilar_classical(i, classical_entry, classical_motif_col,
                                                         best_homolog_classical_motif, best_homolog_classical_score,
                                                         classical_identity, reference_df, classical_inverted)
                    best_homolog_classical_motif, best_homolog_classical_score, classical_identity = output

                # Assign data for current row at specified columns
                new_data[homolog_motif_col].append(best_homolog_classical_motif)
                new_data[f"{homolog_motif_col}_identity"].append(classical_identity)
                new_data[f"{homolog_motif_col}_score"].append(best_homolog_classical_score)

            else:
                new_data[homolog_motif_col].append(np.nan)
                new_data[f"{homolog_motif_col}_identity"].append(np.nan)
                new_data[f"{homolog_motif_col}_score"].append(np.nan)

def find_best_homolog_classical(row_idx, entry, motif_col, reference_df, identity_thres, motif_length):
    # Helper function that tries to find a homologous classical motif if a classical model is being compared

    best_homolog_classical_motif = " " * motif_length
    best_homolog_classical_score = 0
    classical_identity = 0

    for homolog_motif, homolog_classical_score in entry:
        ref_motif = reference_df.at[row_idx, motif_col]

        if isinstance(ref_motif, str) and isinstance(homolog_motif, str):
            identity_arr = np.equal(np.array(ref_motif), np.array(homolog_motif))
            current_identity = identity_arr.mean()
            if current_identity > 0 and current_identity < 1:
                print(f"find_passing_homolog_motif(): Identity between {ref_motif} and {homolog_motif}: {current_identity}")
        else:
            current_identity = 0

        if current_identity > classical_identity and current_identity >= identity_thres:
            best_homolog_classical_motif = homolog_motif
            classical_identity = current_identity
            best_homolog_classical_score = homolog_classical_score

    output = (best_homolog_classical_motif, best_homolog_classical_score, classical_identity)

    return output

def accept_dissimilar_classical(row_idx, entry, motif_col, best_classical_homolog_motif, best_classical_homolog_score,
                                identity, reference_df, classical_inverted = True):
    # If no similar motif could be found, accept a dissimilar one

    for homolog_motif, classical_homolog_score in entry:
        ref_motif = reference_df.at[row_idx, motif_col]

        if isinstance(ref_motif, str) and isinstance(homolog_motif, str):
            identity_arr = np.equal(np.array(ref_motif), np.array(homolog_motif))
            current_identity = identity_arr.mean()
            if current_identity > 0 and current_identity < 1:
                print(f"accept_dissimilar_motif(): Identity between {ref_motif} and {homolog_motif}: {current_identity}")
        else:
            current_identity = 0

        if classical_inverted:
            better = classical_homolog_score > best_classical_homolog_score
        else:
            better = classical_homolog_score < best_classical_homolog_score

        if better:
            best_classical_homolog_motif = homolog_motif
            best_classical_homolog_score = classical_homolog_score
            identity = current_identity

    output = (best_classical_homolog_motif, best_classical_homolog_score, identity)

    return output

def hash_args(*args, hash_len = None):
    # Generates a unique reproducible hash for an arbitrary list/tuple of args

    # Serialize the objects to a binary format
    combined_data = b""
    for obj in args:
        combined_data += pickle.dumps(obj)

    # Generate a SHA-256 hash of the serialized data
    full_hash = hashlib.sha256(combined_data).hexdigest()
    hash = full_hash[:hash_len] if hash_len is not None else full_hash

    return hash

def generate_merged_df(best_homolog_dict, target_gene_dict, motif_cols, reference_df, target_taxids, ref_gene_col,
                       best_classical_homolog_dict = None, target_classical_gene_dict = None, classical_inverted = True,
                       identity_thres = 0.3, motif_length = 15):
    # Generate a merged dataframe extracting best motifs from target species as homologs to reference species

    merged_df = reference_df.copy()
    final_homolog_motif_cols = []
    final_homolog_call_cols = []
    final_homolog_classical_motif_cols = []

    for target_taxid in target_taxids:
        print(f"\tProcessing taxid {target_taxid}...")

        # Construct a dict to contain the new column data; doing this will be faster than repeatedly calling df.at
        include_classical = best_classical_homolog_dict is not None and target_classical_gene_dict is not None
        new_data = construct_new_data_dict(target_taxid, motif_cols, final_homolog_motif_cols, final_homolog_call_cols,
                                           final_homolog_classical_motif_cols, include_classical)

        # Iterate over the reference dataframe to find homologs for the current taxid
        assign_novel_data(new_data, best_homolog_dict, target_gene_dict, motif_cols, target_taxid, reference_df,
                          ref_gene_col, identity_thres, motif_length)

        # Get results from classical model if one is being compared
        if include_classical:
            assign_classical_data(new_data, best_classical_homolog_dict, target_classical_gene_dict, motif_cols,
                                  target_taxid, reference_df, ref_gene_col, identity_thres, motif_length,
                                  classical_inverted)

        # Assign new data to merged dataframe
        for col, col_data in new_data.items():
            merged_df[col] = col_data

    return (merged_df, final_homolog_motif_cols, final_homolog_call_cols, final_homolog_classical_motif_cols)

cwd = os.getcwd()
def score_best_homologs(reference_taxid, reference_df, target_taxids, target_dfs, ref_gene_col = "ensembl_gene_id",
                        identity_thres = 0.3, motif_length = 15, classical_inverted = True, mapping_verbose = False,
                        target_homology_dicts = None):

    args_hash = hash_args(reference_taxid, reference_df, target_taxids, target_dfs, ref_gene_col,
                          identity_thres, motif_length, hash_len=8)
    pickling_path = os.path.join(cwd, f"merged_df_{args_hash}.pkl")
    if os.path.exists(pickling_path):
        with open(pickling_path, "rb") as file:
            print(f"Found pickled merged_df matching input args; reloading...")
            data = pickle.load(file)
            merged_df, final_homolog_motif_cols, final_homolog_call_cols, final_homolog_classical_motif_cols = data
    else:
        # Get a dictionary of dictionaries, i.e. target_taxid --> reference_gene --> target_homologs
        if target_homology_dicts is None:
            target_homology_dicts = map_homologies(reference_taxid, target_taxids, infer_verbose = mapping_verbose)

        # Extract column names containing motif sequences; assume they are the same in all dataframes
        novel_motif_cols = [col for col in reference_df.columns if col[:6] == "Novel_" and col[-6:] == "_motif"]
        if len(novel_motif_cols) == 0:
            novel_motif_cols = [col for col in reference_df.columns if col[-6:] == "_motif"]
            classical_motif_cols, classical_score_cols = None, None
        else:
            classical_motif_cols = [col for col in reference_df.columns if col[:10] == "Classical_" and col[-6:] == "_motif"]
            if len(classical_motif_cols) > 0:
                classical_score_cols = [col.split("_motif")[0] + "_total_motif_score" for col in classical_motif_cols]
            else:
                classical_motif_cols, classical_score_cols = None, None
        novel_binding_cols = [col.split("_motif")[0] + "_binding_motif_score" for col in novel_motif_cols]
        novel_classification_cols = [col.split("_motif")[0] + "_total_motif_score" for col in novel_motif_cols]
        novel_call_cols = [col.split("_motif")[0] + "_final_call" for col in novel_motif_cols]

        # Organize target dataframes into dictionaries of gene names
        print(f"Organizing dataframes into score and gene dictionaries...")
        organized_outputs = organize_targets(target_taxids, target_dfs, ref_gene_col, novel_motif_cols,
                                             novel_binding_cols, novel_classification_cols, novel_call_cols,
                                             classical_motif_cols, classical_score_cols)

        target_score_dict, target_gene_dict = organized_outputs[:2]
        if len(organized_outputs) == 2:
            target_classical_score_dict, target_classical_gene_dict = None, None
        elif len(organized_outputs) == 4:
            target_classical_score_dict, target_classical_gene_dict = organized_outputs[2:]
        else:
            raise ValueError(f"organized_targets() returned an unexpected number of args ({len(organized_outputs)})")

        # Make a dictionary of best homologs by taxid for each reference gene, based on masked binding score
        print("Generating dictionary of best homologs by taxid for each reference gene...")
        best_novel_homolog_dict = get_best_homologs(reference_df, ref_gene_col, target_homology_dicts, target_score_dict)
        if target_classical_score_dict:
            best_classical_homolog_dict = get_best_homologs(reference_df, ref_gene_col, target_homology_dicts,
                                                            target_classical_score_dict)
        else:
            best_classical_homolog_dict = None

        # Generate a merged dataframe extracting best motifs from target species as homologs to reference species
        print("Generating merged dataframe with homolog motif sequences...")
        merged_tuple = generate_merged_df(best_novel_homolog_dict, target_gene_dict, novel_motif_cols, reference_df,
                                          target_taxids, ref_gene_col, best_classical_homolog_dict,
                                          target_classical_gene_dict, classical_inverted, identity_thres, motif_length)
        merged_df, final_homolog_motif_cols, final_homolog_call_cols, final_homolog_classical_motif_cols = merged_tuple

        with open(pickling_path, "wb") as file:
            data = (merged_df, final_homolog_motif_cols, final_homolog_call_cols, final_homolog_classical_motif_cols)
            pickle.dump(data, file)

    return (merged_df, final_homolog_motif_cols, final_homolog_call_cols, final_homolog_classical_motif_cols)

def score_homolog_motifs(data, homolog_motif_cols = None, homolog_motif_col_groups = None,
                         predictor_params = predictor_params):
    '''
    Main function for scoring homologous motifs

    Args:
        data (pd.DataFrame|list|tuple):  main dataframe with motif sequences for host and homologs,
                                         or list of dataframes where first one is host and the rest are for homologs
        homolog_motif_cols (list|tuple): col names where homolog motif sequences are stored
        homolog_motif_col_groups (dict): dict of host motif seq col --> grouped column names for each homologous motif
        predictor_params (dict):         dictionary of parameters for scoring

    Returns:
        data_df (pd.DataFrame):     dataframe with scores added for homolog motifs
        homolog_id_cols (list):     shortened list of col names where homolog ids are stored
        homolog_motif_cols (list):  shortened list of col names containing homologous motifs
        model_score_cols (list):    shortened list of col names containing homologous motif scores according to model
    '''

    verbose = predictor_params["homology_params"]["homolog_scoring_verbose"]
    selection_mode = predictor_params["homology_params"]["homolog_selection_mode"]

    # Handle selection_mode data type requirements
    if selection_mode == "best":
        # Extract reference and target taxids
        keys = list(predictor_params["protein_seqs_paths"].keys())
        reference_taxid = keys[0]
        target_taxids = keys[1:]
    else:
        reference_taxid, target_taxids = None, None

    # Extract dataframes, assuming they are in order with reference dataframe first
    if isinstance(data, pd.DataFrame):
        if selection_mode == "best":
            raise ValueError(f"`data` is a single dataframe, but `selection_mode` is set to {selection_mode}, "
                             f"which requires a list of dataframes - one for each species being compared")
        else:
            data_df = data
            data_dfs = None
    elif isinstance(data, list) or isinstance(data, tuple):
        data_df = None
        data_dfs = data
    else:
        raise ValueError(f"`data` type is {type(data)}, but requires pd.DataFrame or list/tuple of pd.DataFrame objects")

    # Load ConditionalMatrices object to be used in scoring
    conditional_matrices_path = predictor_params["conditional_matrices_path"]
    with open(conditional_matrices_path, "rb") as f:
        conditional_matrices = pickle.load(f)

    if selection_mode == "best":
        # Search existing predictions for homologous proteins
        ref_gene_col = predictor_params["homology_params"]["ref_gene_col"]
        identity_thres = predictor_params["homology_params"]["identity_thres"]
        motif_length = predictor_params["motif_length"]
        reference_df = data_dfs[0]
        target_dfs = data_dfs[1:]
        output = score_best_homologs(reference_taxid, reference_df, target_taxids, target_dfs, ref_gene_col,
                                     identity_thres, motif_length)
        data_df, final_homolog_motif_cols, final_homolog_call_cols, final_homolog_classical_motif_cols = output
    else:
        # Score unique motif sequences
        motif_seqs, motif_seqs_2d = extract_unique_motifs(data_df, homolog_motif_cols, verbose)
        results = score_motifs_parallel(motif_seqs_2d, conditional_matrices, predictor_params)
        dicts_output = parse_motif_dicts(results, verbose)
        total_dict, binding_dict, positive_dict, suboptimal_dict, forbidden_dict = dicts_output[0:5]
        calls_dict, classical_dict, combined_score = dicts_output[5:]

        # Iterate over homolog motif col groups, organized by host motif col
        data_df, final_homolog_motif_cols, final_call_cols = score_similar_homologs(data_df, homolog_motif_col_groups,
                                                                                    total_dict, binding_dict,
                                                                                    positive_dict, suboptimal_dict,
                                                                                    forbidden_dict, calls_dict,
                                                                                    classical_dict, predictor_params)

    # Apply classical method if necessary
    if predictor_params["compare_classical_method"]:
        data_df = apply_classical(data_df, final_homolog_motif_cols, final_homolog_call_cols)

    return data_df, final_homolog_motif_cols