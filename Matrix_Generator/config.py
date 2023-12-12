# This is the configuration file containing all the arguments and preferences for matrix_generator.py; please edit as necessary

import numpy as np

# Define the amino acid alphabet

amino_acids = ("D", "E", "R", "H", "K", "S", "T", "N", "Q", "C", "G", "P", "A", "V", "I", "L", "M", "F", "Y", "W")
amino_acids_phos = ("D", "E", "R", "H", "K", "S", "T", "N", "Q", "C", "G", "P", "A", "V", "I", "L", "M", "F", "Y", "W", "B", "J", "O") # B=pSer, J=pThr, Y=pTyr

''' ----------------------------------------------------------------------------------------------------------------
                                      SPOT Image Quantification Parameters
    ---------------------------------------------------------------------------------------------------------------- '''

''' Image quantification preferences include in image_params: 
        "use_cached_data":                    whether to use pickled quantified image data from a previous run
        "cached_data_path":                   the path to the cached data if use_cached_data is True
        "output_folder":                      folder to save output data and images showing detected spots
        "add_peptide_seqs":                   whether to add peptide sequences to the respective quantified data points
        "peptide_seq_cols":                   cols with peptide sequences that should be added to quantified image data
        "save_pickled_data":                  whether to save pickled data for future attempts
        "buffer_width":                       width of the buffer zone between a defined spot and its exterior 
                                              surroundings, used during background signal adjustment
        "tiff_paths":                         list of directories containing groups of TIFF images to be quantified; 
                                              the image file names must be: 
                                                    [probe_name]_Copy[replicate_number]_Scan[scan_order_number].tif
                                              where probe_name is the bait probe name, 
                                              replicate_number is the technical replicate number, 
                                              and scan_order_number is the number representing the position in the order 
                                              that the  baits were applied to the blots between stripping cycles
        "pixel_encoding_base":                logarithm base for pixel encoding; if set to 1, linear encoding is assumed
        "add_peptide_names":                  whether to add peptide names
        "peptide_names_paths":                list of paths to peptide names CSV files matching the TIFF image folders
        "processed_image_paths":              paths where outlined and processed images will be saved for each input dir
        "grid_dimensions":                    for each path, list of number of spots wide by number of spots high
        "circle_index_threshold":             call index threshold used for making positive/negative calls 
        "last_valid_coords":                  list of last valid alphanumeric spot coords where data ends in each set
        "ordered_probe_names":                list of bait probe names in the order they should appear in the output df 
        "control_probe_name":                 name of the negative control probe, e.g. Secondary-only or Mock 
        "control_multiplier":                 multiplier for control signal values to test against bait signal values 
        "standardize_within_datasets":        whether to standardize dataframes within themselves, between baits
        "intra_dataset_controls":             list of controls to use for intra-dataset standardization
        "max_bait_mean_col":                  column name where max bait mean signal values will be assigned
        "standardize_between_datasets":       whether to standardize dataframes between each other
        "inter_dataset_control":              control to use for inter-dataset standardization
        "enforce_positive_control_multiple":  whether to enforce a minimum multiple of the positive control that other 
                                              peptides must exceed to be considered significant
        "positive_control":                   control to use if enforcing significant hits to be multiples of a control
        "positive_control_multiple":          the multiple of the positive control signal that is considered the lowest 
                                              allowed to be considered significant'''

image_params = {"use_cached_data": False,
                "cached_data_path": None,
                "output_folder": "/home/user/example_folder/SPOT Peptide Screens/Straightened_TIFFs/Sharpened and Cleaned/Image_Data",
                "add_peptide_seqs": True,
                "peptide_seq_cols": ["Phos_Sequence", "No_Phos_Sequence", "BJO_Sequence"],
                "save_pickled_data": True,
                "buffer_width": 2,
                "tiff_paths": ["/home/user/example_folder/SPOT Peptide Screens/Straightened_TIFFs/Sharpened and Cleaned/1_ExampleA",
                               "/home/user/example_folder/SPOT Peptide Screens/Straightened_TIFFs/Sharpened and Cleaned/2_ExampleB"],
                "pixel_encoding_base": 1,
                "add_peptide_names": True,
                "multiline_cols": False,
                "peptide_names_paths": ["/home/user/example_folder/SPOT Peptide Screens/Straightened_TIFFs/Sharpened and Cleaned/1_ExampleA_Coordinate_Peptide_Names_and_Sequences.csv",
                                        "/home/user/example_folder/SPOT Peptide Screens/Straightened_TIFFs/Sharpened and Cleaned/2_ExampleB_Coordinate_Peptide_Names_and_Sequences.csv"],
                "processed_image_paths": ["/home/user/example_folder/SPOT Peptide Screens/Straightened_TIFFs/Sharpened and Cleaned/Image_Data/1_ExampleA",
                                          "/home/user/example_folder/SPOT Peptide Screens/Straightened_TIFFs/Sharpened and Cleaned/Image_Data/2_ExampleB"],
                "grid_dimensions": [[28, 6], [28, 4]],
                "circle_index_threshold": 1.4,
                "last_valid_coords": ["F1", "D8"],
                "ordered_probe_names": ["Secondary-only", "GHI3", "DEF2", "ABC1"],
                "control_probe_name": "Secondary-only",
                "control_multiplier": 5,
                "standardize_within_datasets": True,
                "intra_dataset_controls": ["OSBP"],
                "max_bait_mean_col": "Max_Bait_Background-Adjusted_Mean",
                "standardize_between_datasets": True,
                "inter_dataset_control": "OSBP",
                "enforce_positive_control_multiple": True,
                "positive_control": "OSBP",
                "positive_control_multiple": 0.05}


''' ----------------------------------------------------------------------------------------------------------------
                                    Conditional Weighted Matrix Configuration
    ---------------------------------------------------------------------------------------------------------------- '''

''' Amino acid chemical characteristics dict must encompass one instance of every amino acid, split into lists and 
    assigned to keys representing side chain chemical characteristics '''

aa_charac_dict = {"Acidic": ["D", "E"],
                  "Basic": ["K", "R"],
                  "ST": ["S", "T", "B", "J"],
                  "Phenyl": ["F", "Y", "W", "O"],
                  "Aliphatic": ["A", "V", "I", "L", "M"],
                  "Polar": ["N", "Q", "H", "C"],
                  "Proline": ["P"],
                  "Glycine": ["G"],
                  "Indole": ["W"]}

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
                  "output_folder": "/home/user/example_folder/SPOT Peptide Screens/Straightened_TIFFs/Sharpened and Cleaned/Matrix_Data",
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
        "optimize_weights":       whether to permute weights to get optimally accurate results; takes a long time
        "possible_weights":       possible weights values to permute for optimization 
        "chunk_size":             parallel processing chunk size for weights optimization if optimize_weights is True
        "position_weights":       array of weights reflecting the relative score contributions of each position
        "forbidden_threshold":    minimum number of peptides with a putative forbidden residue before the residue is 
                                  considered forbidden
        "slice_scores_subsets":   array of sequence subsets to be scored and thresholded separately; sum must be equal 
                                  to motif length; as an example, array[5,2,4,1,1,2] slices DEDDENEFFDAPEII as 
                                  DEDDE NE FFDA P E II '''

matrix_params = {"thresholds_points_dict": None,
                 "points_assignment_mode": "continuous",
                 "amino_acids": amino_acids_phos,
                 "include_phospho": False,
                 "min_members": 20,
                 "barnard_alpha": 0.2,
                 "suboptimal_points_mode": "counts",
                 "min_aa_entries": 4,
                 "replace_forbidden": True,
                 "optimize_weights": True,
                 "possible_weights": None,
                 "chunk_size": 1000,
                 "position_weights": np.array([0.5,0.5,0.5,0.5,0.5,
                                               1,2,
                                               3,3,3,3,
                                               0,
                                               1,
                                               0,0]),
                 "objective_mode": "accuracy",
                 "forbidden_threshold": 3}


''' ----------------------------------------------------------------------------------------------------------------
                                         Specificity Matrix Configuration
    ---------------------------------------------------------------------------------------------------------------- '''

''' Comparator information in comparator_info: 
        "comparator_set_1": list of baits to pool as the first comparator when assessing binding specificity
        "comparator_set_2": list of baits to pool as the second comparator as above; if None, user is prompted
        "seq_col":          name of column containing sequences to use for specificity matrix construction
        "bait_pass_col":    name of column containing pass/fail information for whether peptides bind to the baits
        "pass_str":         string representing a pass in bait_pass_col, e.g. "Yes" '''

comparator_info = {"comparator_set_1": ["GHI3"],
                   "comparator_set_2": ["ABC1", "DEF2"],
                   "seq_col": "BJO_Sequence",
                   "bait_pass_col": "One_Passes",
                   "pass_str": "Yes"}

''' Specificity matrix generation parameters contained in specificity_params: 
        "motif_length":              the length of the motif being studied
        "include_phospho":           whether to include separate phospho-residues during matrix construction
        "predefined_weights":        array of predefined position weights to apply to the specificity matrix
        "optimize_weights":          whether to optimize position weights
        "control_peptide_index":     row index of the positive control peptide to use for thresholding
        "control_peptide_threshold": percentage of control peptide max signal that a peptide must exceed before being 
                                     allowed to contribute to matrix-building
        "output_folder":             the folder to save the specificity matrix and scored output data into 
        "chunk_size":                parallel processing chunk size for weights optimization if optimize_weights is True
        "ignore_positions":          position indices (0-indexed) that should have their weights forcibly set to zero
        "max_bait_mean_col":         column name containing max mean signal from the top bait for a given peptide
        "plus_threshold":            positive log2fc threshold to be considered "specific"
        "minus_threshold":           negative log2fc threshold to be considered "specific"
        "fit_mode":                  can be either "f1" (fits weights to f1-score), "mcc" (fits to MCC), or "accuracy"
        "standardize_matrix":        whether to standardize matrix values per col; set to True if applying weights
        "matrix_alpha":              p-value threshold for log2fc values before a nonzero value can be added to the 
                                     matrix; this is generally much more permissive than significance testing, and 
                                     serves only to remove obvious noise '''

specificity_params = {"motif_length": 15,
                      "include_phospho": False,
                      "predefined_weights": None,
                      "optimize_weights": True,
                      "optimize_separately": False,
                      "control_peptide_index": 0,
                      "control_peptide_threshold": 0.3,
                      "output_folder": "/home/user/example_folder/SPOT Peptide Screens/Straightened_TIFFs/Sharpened and Cleaned/Matrix_Data",
                      "chunk_size": 1000,
                      "ignore_positions": (14,),
                      "max_bait_mean_col": "Max_Bait_Background-Adjusted_Mean",
                      "plus_threshold": 1.0,
                      "minus_threshold": -1.0,
                      "fit_mode": "f1",
                      "standardize_matrix": True,
                      "matrix_alpha": 0.25}