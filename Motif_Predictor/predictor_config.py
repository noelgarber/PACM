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

predictor_params = {"protein_seqs_path": "/home/user/example_folder/Biomart_NR_110.csv",
                    "motif_length": 15,
                    "use_weighted": True,
                    "convert_phospho": True,
                    "use_specificity_weighted": True,
                    "return_count": 3,
                    "compare_classical_method": True,
                    "enforced_position_rules": {7: np.array(["F","Y"]),
                                                9: np.array(["D","E","S","T"])},
                    "conditional_matrices_path": "/home/user/example_folder/conditional_matrices.pkl",
                    "specificity_matrix_path": "/home/user/example_folder/specificity_matrix.pkl",
                    "scored_output_path": "/home/user/example_folder/biomart_scored_data.csv",
                    "check_topology": True,
                    "uniprot_refresh_time": 1,
                    "core_start": 6,
                    "core_end": 12}