# This is the configuration file containing all the arguments and preferences for main.py; please edit as necessary

from general_utils.general_vars import amino_acids_phos

''' ----------------------------------------------------------------------------------------------------------------
                                    Conditional Weighted Matrix Configuration
    
    Structure of aa_charac_dict: 
        chemical_characteristic: list of member amino acids
        Example: 
            "Acidic": ["D","E"]
    
    Structure of general_params: 
        "motif_length":           length of the peptide motif for which matrices are being generated
        "output_folder":          folder to save matrices and scored data into; user is prompted if None
        "make_calls":             whether to make calls on whether peptides pass based on scoring
        "aa_charac_dict":         the aforementioned amino acid side chain characteristics dictionary
        "convert_phospho":        whether to convert phospho-residues in peptide before constructing matrices; 
                                  phospho-residues are denoted as pSer=B, pThr=J, pTyr=O
        "position_thresholds":    range of values for optimization, as a list, for thresholding points values
    
    Structure of data_params: 
        "bait":                   bait protein name for generating conditional matrices; if None, best is used
        "bait_signal_col_marker": keyword denoting whether a column in the input dataframe has bait signal values
        "best_signal_col":        name of column in input dataframe that has best signal values, when "bait" is None
        "bait_pass_col":          name of column with pass/fail calls for peptide binding
        "pass_str":               string found in bait_pass_col that denotes a positive, e.g. "Yes"
        "seq_col":                name of column with peptide sequences
        "dest_score_col":         name of the destination column where motif score values will be assigned
    
    Structure of matrix_params: 
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
        
    ---------------------------------------------------------------------------------------------------------------- '''

aa_charac_dict = {"Acidic": ["D", "E"],
                  "Basic": ["K", "R"],
                  "ST": ["S", "T", "B", "J"],
                  "Aromatic": ["F", "Y", "W", "O"],
                  "Aliphatic": ["A", "V", "I", "L", "M"],
                  "Polar": ["N", "Q", "H", "C"],
                  "Proline": ["P"],
                  "Glycine": ["G"]
}

general_params = {"motif_length": 15,
                  "output_folder": None,
                  "make_calls": True,
                  "aa_charac_dict": aa_charac_dict,
                  "convert_phospho": True,
                  "position_thresholds": np.array([0.4,0.8,0.0,0.6,0.2])}

data_params = {"bait": None,
               "bait_signal_col_marker": "Background-Adjusted_Standardized_Signal",
               "best_signal_col": "Max_Bait_Background-Adjusted_Mean",
               "bait_pass_col": "One_Passes",
               "pass_str": "Yes",
               "seq_col": "BJO_Sequence",
               "dest_score_col": "SLiM_Score"}

matrix_params = {"thresholds_points_dict": None,
                 "points_assignment_mode": "continuous",
                 "amino_acids": amino_acids_phos,
                 "include_phospho": False,
                 "min_members": 10,
                 "clear_filtering_column": False,
                 "penalize_negatives": True,
                 "use_sigmoid": False,
                 "sigmoid_strength": 1,
                 "sigmoid_inflection": 0.5}

