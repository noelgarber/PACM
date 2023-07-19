# This is the configuration file containing all the arguments and preferences for main.py; please edit as necessary

import numpy as np

# Define the amino acid alphabet

amino_acids = ("D", "E", "R", "H", "K", "S", "T", "N", "Q", "C", "G", "P", "A", "V", "I", "L", "M", "F", "Y", "W")
amino_acids_phos = ("D", "E", "R", "H", "K", "S", "T", "N", "Q", "C", "G", "P", "A", "V", "I", "L", "M", "F", "Y", "W", "B", "J", "O") # B=pSer, J=pThr, Y=pTyr

''' ----------------------------------------------------------------------------------------------------------------
                                      SPOT Image Quantification Parameters
    ---------------------------------------------------------------------------------------------------------------- '''

''' Image quantification preferences include in image_params: 
        "use_cached_data":   whether to use pickled quantified image data from a previous run
        "cached_data_path":  the path to the cached data if use_cached_data is True; user is prompted if not given
        "output_folder":     folder to save output data and images showing detected spots; user is prompted if None
        "add_peptide_seqs":  whether to add peptide sequences to the respective quantified data points
        "peptide_seq_cols":  the cols containing peptide sequence data that should be added to quantified image data
        "save_pickled_data": whether to save pickled data for future attempts, rather than generating anew each time
        "buffer_width":      the width of the buffer zone between a defined spot and its exterior surroundings, used
                             during background signal adjustment and call index calculation '''

image_params = {"use_cached_data": True,
                "cached_data_path": "",
                "output_folder": None,
                "add_peptide_seqs": True,
                "peptide_seq_cols": ["Phos_Sequence", "No_Phos_Sequence", "BJO_Sequence"],
                "save_pickled_data": True,
                "buffer_width": None}


''' ----------------------------------------------------------------------------------------------------------------
                                    Conditional Weighted Matrix Configuration
    ---------------------------------------------------------------------------------------------------------------- '''

''' Amino acid chemical characteristics dict must encompass one instance of every amino acid, split into lists and 
    assigned to keys representing side chain chemical characteristics '''

aa_charac_dict = {"Acidic": ["D", "E"],
                  "Basic": ["K", "R"],
                  "SerThr": ["S", "T", "B", "J"],
                  "Aromatic": ["F", "Y", "W", "O"],
                  "Aliphatic": ["A", "V", "I", "L", "M"],
                  "Polar": ["N", "Q", "H", "C"],
                  "Proline": ["P"],
                  "Glycine": ["G"]}

''' Amino acid equivalence dict (more closely related), used for constructing the forbidden residues matrix; 
    operates on the premise of "if this residue is forbidden, those residues are also likely to be forbidden" '''

aa_equivalence_dict = {"D": ("D", "E"),
                       "E": ("D", "E"),
                       "R": ("R", "K"),
                       "K": ("R", "K"),
                       "H": ("N", "Q", "R", "K"),
                       "S": ("S", "T", "B", "J"),
                       "T": ("S", "T", "B", "J"),
                       "B": ("S", "T", "B", "J"),
                       "J": ("S", "T", "B", "J"),
                       "F": ("F", "Y", "O"),
                       "Y": ("F", "Y", "O"),
                       "O": ("F", "Y", "O"),
                       "W": ("W"),
                       "A": ("A", "V", "G"),
                       "V": ("A", "V", "I", "L"),
                       "I": ("V", "I", "L", "M"),
                       "L": ("V", "I", "L", "M"),
                       "M": ("I", "L"),
                       "N": ("N", "Q", "H", "C"),
                       "Q": ("N", "Q", "H", "C"),
                       "C": ("N", "Q", "H", "C"),
                       "P": ("P"),
                       "G": ("G", "A")}

''' General parameters included in general_params: 
        "motif_length":           length of the peptide motif for which matrices are being generated
        "output_folder":          folder to save matrices and scored data into; user is prompted if None
        "make_calls":             whether to make calls on whether peptides pass based on scoring
        "aa_charac_dict":         the aforementioned amino acid side chain characteristics dictionary
        "convert_phospho":        whether to convert phospho-residues in peptide before constructing matrices; 
                                  phospho-residues are denoted as pSer=B, pThr=J, pTyr=O
        "position_thresholds":    range of values for optimization, as a list, for thresholding points values '''

general_params = {"motif_length": 15,
                  "output_folder": "",
                  "make_calls": True,
                  "aa_charac_dict": aa_charac_dict,
                  "convert_phospho": True,
                  "position_thresholds": np.array([0.4,0.8,0.0,0.6,0.2])}

''' Input data-specific parameters included in data_params: 
        "bait":                   bait protein name for generating conditional matrices; if None, best is used
        "bait_signal_col_marker": keyword denoting whether a column in the input dataframe has bait signal values
        "best_signal_col":        name of column in input dataframe that has best signal values, when "bait" is None
        "bait_pass_col":          name of column with pass/fail calls for peptide binding
        "pass_str":               string found in bait_pass_col that denotes a positive, e.g. "Yes"
        "seq_col":                name of column with peptide sequences
        "dest_score_col":         name of the destination column where motif score values will be assigned '''

data_params = {"bait": None,
               "bait_signal_col_marker": "Background-Adjusted_Standardized_Signal",
               "best_signal_col": "Max_Bait_Background-Adjusted_Mean",
               "bait_pass_col": "One_Passes",
               "pass_str": "Yes",
               "seq_col": "BJO_Sequence",
               "dest_score_col": "SLiM_Score"}

''' Conditional matrix generation parameters included in matrix_params: 
        "thresholds_points_dict": dict of signal thresholds and associated points values, if mode is "thresholds"
        "points_assignment_mode": if "continuous", points are assigned according to a rational function that puts the 
                                  highest weight on midrange binders and a moderate weight on high-end binders; 
                                  if "thresholds", points are assigned according to thresholds_points_dict
        "amino_acids":            amino acid alphabet to use, as a list of single-letter codes
        "include_phospho":        whether to include phospho-residues in matrices; collapses to non-phospho if False
        "min_members":            the minimum number of peptides belonging to a type-position rule for a conditional
                                  matrix to be constructed; for cases where not enough peptides are in source data, 
                                  the conditional matrix defaults to a standard matrix using all passing peptides
        "clear_filtering_column": whether to clear the column to which a conditional matrix's rule refers
        "penalize_negatives":     whether to use negative peptides to decrement the matrix for disfavoured residues
        "use_sigmoid":            whether to scale matrix values using a sigmoid function
        "sigmoid_strength":       the strength of the sigmoid function scaling; defaults to 1
        "sigmoid_inflection":     threshold where matrix values are scaled larger when above or smaller when below
        "position_weights":       array of weights reflecting the relative score contributions of each position '''

matrix_params = {"thresholds_points_dict": None,
                 "points_assignment_mode": "continuous",
                 "amino_acids": amino_acids_phos,
                 "include_phospho": False,
                 "min_members": 10,
                 "clear_filtering_column": False,
                 "penalize_negatives": True,
                 "use_sigmoid": True,
                 "sigmoid_strength": 0.5,
                 "sigmoid_inflection": 0.3,
                 "position_weights": np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                               0.5, 3, 3, 3, 3, 0, 0.5,
                                               0.25, 0])}


''' ----------------------------------------------------------------------------------------------------------------
                                    Conditional Weighted Matrix Configuration
    ---------------------------------------------------------------------------------------------------------------- '''

''' Comparator information in comparator_info: 
        "comparator_set_1": list of baits to pool as the first comparator when assessing binding specificity
        "comparator_set_2": list of baits to pool as the second comparator as above; if None, user is prompted
        "seq_col":          name of column containing sequences to use for specificity matrix construction
        "bait_pass_col":    name of column containing pass/fail information for whether peptides bind to the baits
        "pass_str":         string representing a pass in bait_pass_col, e.g. "Yes" '''

comparator_info = {"comparator_set_1": None,
                   "comparator_set_2": None,
                   "seq_col": "BJO_Sequence",
                   "bait_pass_col": "One_Passes",
                   "pass_str": "Yes"}

''' If optimizing weights, possible_weights is a list of arrays of possible weight values for each matrix position '''

possible_weights = [np.array([0.0, 1.0]),
                    np.array([0.0, 1.0]),
                    np.array([0.0, 1.0]),
                    np.array([0.0, 1.0]),
                    np.array([0.0, 1.0]),
                    np.array([0.0, 1.0, 2.0]),
                    np.array([0.0, 1.0, 2.0, 3.0]),
                    np.array([0.0, 1.0, 2.0, 3.0]),
                    np.array([0.0, 1.0, 2.0, 3.0]),
                    np.array([0.0, 1.0, 2.0, 3.0]),
                    np.array([0.0, 1.0, 2.0, 3.0]),
                    np.array([0.0, 0.5]),
                    np.array([0.0, 1.0, 2.0]),
                    np.array([0.0, 0.5]),
                    np.array([0.0, 0.5])]

''' Specificity matrix generation parameters contained in specificity_params: 
        "thresholds":            comma-delimited tuple of log2fc thresholds, in descending order, as floats
        "matching_points":       corresponding comma_delimited tuple of points associated with log2fc thresholds
        "include_phospho":       whether to include separate phospho-residues during matrix construction
        "predefined_weights":    array of predefined position weights to apply to the specificity matrix if not optimizing
        "optimize_weights":      whether to optimize position weights
        "possible_weights":      possible weights values to permute for optimization 
        "output_folder":         the folder to save the specificity matrix and scored output data into 
        "chunk_size":            parallel processing chunk size for weights optimization if optimize_weights is True '''

specificity_params = {"motif_length": 15,
                      "thresholds": (1.0, 0.5, -0.5, -1.0),
                      "matching_points": (2.0, 1.0, -1.0, -2.0),
                      "include_phospho": False,
                      "predefined_weights": None,
                      "optimize_weights": True,
                      "possible_weights": possible_weights,
                      "output_folder": "",
                      "chunk_size": 1000}