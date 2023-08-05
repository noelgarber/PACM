# This is the configuration file for running predictions on protein sequences based on previously generated matrices.

'''
The following is a dictionary of required parameters for the motif prediction pipeline; please fill in with your info.

protein_seqs_path (str):         path to protein sequences as a CSV file
motif_length (int):              length of the motif being predicted; must match conditional matrices used for scoring
use_weighted (bool):             whether to use weighted or unweighted conditional matrices
return_count (int):              number of motifs to return for each protein
compare_classical_method (bool): whether to import classical_method() from classical_method.py and compare to main score
enforced_position_rules (dict):  dict of position index --> allowed residues at index; omitted indices allow all
conditional_matrices_path (str): path to ConditionalMatrices object used for sequence scoring
specificity_matrix_path (str):   path to SpecificityMatrix object used for sequence specificity scoring
'''

predictor_params = {"protein_seqs_path": "",
                    "motif_length": 15,
                    "use_weighted": True,
                    "return_count": 3,
                    "compare_classical_method": True,
                    "enforced_position_rules": {7: np.array(["F","Y"]),
                                                9: np.array(["D","E","S","T"])},
                    "conditional_matrices_path": "/home/user/example_folder/conditional_matrices.pkl",
                    "specificity_matrix_path": "/home/user/example_folder/specificity_matrix.pkl"}