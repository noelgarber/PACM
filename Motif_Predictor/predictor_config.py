# This is the configuration file for running predictions on protein sequences based on previously generated matrices.

import numpy as np

'''
The following is a dictionary of required parameters for the motif prediction pipeline; please fill in with your info.

protein_seqs_path (str):         path to protein sequences as a CSV file
motif_length (int):              length of the motif being predicted; must match conditional matrices used for scoring
use_weighted (bool):             whether to use weighted or unweighted conditional matrices
convert_phospho (bool):          whether to convert phospho-residues before scoring them
use_specificity_weighted (bool): whether to use weighted or unweighted specificity matrix
return_count (int):              number of motifs to return for each protein
compare_classical_method (bool): whether to import classical_method() from classical_method.py and compare to main score
enforced_position_rules (dict):  dict of position index --> allowed residues at index; omitted indices allow all
conditional_matrices_path (str): path to ConditionalMatrices object used for sequence scoring
specificity_matrix_path (str):   path to SpecificityMatrix object used for sequence specificity scoring
core_start (int):                the index where the 'core' of a motif begins, for topology checking
core_end (int):                  the index where the 'core' of a motif ends, for topology checking
'''

protein_seqs_paths = ["/home/user/Documents/GitHub/PACM/proteome_dataset_7955_homologs.csv", 
                      "/home/user/Documents/GitHub/PACM/proteome_dataset_10116_homologs.csv",
                      "/home/user/Documents/GitHub/PACM/proteome_dataset_10090_homologs.csv",
                      "/home/user/Documents/GitHub/PACM/proteome_dataset_6239_homologs.csv",
                      "/home/user/Documents/GitHub/PACM/proteome_dataset_7227_homologs.csv",
                      "/home/user/Documents/GitHub/PACM/proteome_dataset_4932_homologs.csv",
                      "/home/user/Documents/GitHub/PACM/proteome_dataset_4896_homologs.csv",
                      "/home/user/Documents/GitHub/PACM/proteome_dataset_3702_homologs.csv"]

scored_output_paths = ["/home/user/Documents/GitHub/PACM/proteome_dataset_7955_homologs_scored.csv", 
                       "/home/user/Documents/GitHub/PACM/proteome_dataset_10116_homologs_scored.csv",
                       "/home/user/Documents/GitHub/PACM/proteome_dataset_10090_homologs_scored.csv",
                       "/home/user/Documents/GitHub/PACM/proteome_dataset_6239_homologs_scored.csv",
                       "/home/user/Documents/GitHub/PACM/proteome_dataset_7227_homologs_scored.csv",
                       "/home/user/Documents/GitHub/PACM/proteome_dataset_4932_homologs_scored.csv",
                       "/home/user/Documents/GitHub/PACM/proteome_dataset_4896_homologs_scored.csv",
                       "/home/user/Documents/GitHub/PACM/proteome_dataset_3702_homologs_scored.csv"]

df_chunks = [20,
             20,
             20,
             20,
             20,
             20,
             20,
             20]

predictor_params = {"protein_seqs_path": protein_seqs_paths,
                    "scored_output_path": scored_output_paths,
                    "df_chunks": df_chunks,
                    "seq_col": "sequence",
                    "motif_length": 15,
                    "pickled_weights_path": "/home/user/Documents/SPOT Peptide Screens/Straightened_TIFFs/Sharpened and Cleaned/Matrix_Data/conditional_weights_tuple.pkl",
                    "convert_phospho": True,
                    "assign_specificity_scores": False,
                    "use_specificity_weighted": True,
                    "return_count": 3,
                    "compare_classical_method": True,
                    "enforced_position_rules": {7: np.array(["F","Y"]),
                                                9: np.array(["D","E","S","T"]),
                                                10: np.array(["A", "V", "I", "L", "M", "F", "Y", "W",
                                                              "S", "T", "H", "N", "Q", "C", "G", "P"])},
                    "conditional_matrices_path": "/home/user/Documents/SPOT Peptide Screens/Straightened_TIFFs/Sharpened and Cleaned/Matrix_Data/conditional_matrices.pkl",
                    "standardization_coefficients_path": "/home/user/Documents/SPOT Peptide Screens/Straightened_TIFFs/Sharpened and Cleaned/Matrix_Data/standardization_coefficients.pkl",
                    "optimized_thresholds_path": "/home/user/Documents/SPOT Peptide Screens/Straightened_TIFFs/Sharpened and Cleaned/Matrix_Data/best_thresholds.pkl",
                    "specificity_matrix_path": "/home/user/Documents/SPOT Peptide Screens/Straightened_TIFFs/Sharpened and Cleaned/Matrix_Data/weighted_specificity_matrix.pkl",
                    "leading_glycines": 6,
                    "trailing_glycines": 2,
                    "replace_selenocysteine": True,
                    "selenocysteine_substitute": "C",
                    "gap_substitute": "G",
                    "homolog_score_chunk_size": 1000,
                    "homolog_selection_mode": "similarity",
                    "homolog_scoring_verbose": False,
                    "check_topology": True,
                    "parse_topologies_upfront": True,
                    "topology_trim_begin": 6,
                    "topology_trim_end": 2,
                    "topology_chunk_size": 1000,
                    "topology_verbose": False,
                    "uniprot_path": "/home/user/Documents/GitHub/PACM/uniprot_sprot_human.pkl",
                    "uniprot_refresh_time": 1,
                    "core_start": 6,
                    "core_end": 12,
                    "chunk_size": 1000}