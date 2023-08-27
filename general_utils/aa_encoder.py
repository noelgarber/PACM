'''
This is a set of dictionaries of amino acids, by single-letter code, describing their chemical characteristics for
purposes of encoding and exposing to neural networks
'''

import numpy as np
from sklearn.preprocessing import StandardScaler

'''
Kyte-Doolittle hydrophobicity scale
    --> Positive values are hydrophobic, and negative values are hydrophilic
    --> Source: Kyte J, Doolittle RF. J Mol Biol. 1982 May 5;157(1):105-32.
    --> pSer (B) was assigned as -4.0, slightly less hydrophobic than Asp
    --> pThr (J) was assigned as -3.5, slightly more hydrophobic than pSer because of an extra methyl group
    --> pTyr (J) was assigned as -2.0, since phosphate is very hydrophilic, but the phenyl group is very hydrophobic
'''
kyte_doolittle_dict = {"I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
                       "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
                       "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
                       "B": -4.0, "J": -3.5, "O": -2.0}

'''
Huang-Nau conformational flexibility scale
    --> This dict contains the quenching frequency values, which are correlated with end-to-end collision frequency
    --> Values for Cys, Met, Tyr, & Trp were estimated based on other amino acids, but not directly measured
    --> Values for pSer, pThr, pTyr were estimated based on the assumption that phosphate causes decreased flexibility
    --> Source: https://doi.org/10.1002/anie.200250684
'''
huang_nau_dict = {"G": 39.0, "S": 25.0, "D": 19.0, "N": 20.0, "A": 18.0, "T": 11.0, "L": 10.0, "F": 7.6,
                  "E": 7.4, "Q": 7.2, "H": 4.8, "R": 4.6, "K": 2.8, "V": 3.0, "I": 2.3, "P": 0.1,
                  "C": 20.0, "M": 4.0, "Y": 6.0, "W": 2.0,
                  "B": 8.0, "J": 5.0, "O": 2.0}

'''
Side chain charges, including partial values for residues where two species exist
    --> Source of side chain pKa values: https://www.vanderbilt.edu/AnS/Chemistry/Rizzo/stuff/AA/AminoAcids.html
    --> Cys has a side chain pKa of ~8.37, so at pH 7.4, ~9.68% is deprotonated as a negatively charged thiolate
    --> His has a side chain pKa of about ~6.04, so at pH 7.4, ~4.18% will be protonated (positive)
    --> Phosphate has pKa values of about 1.2 and 6.5, so at pH 7.4, ~100% will have the first oxygen charged, 
        and ~88.8% will have the second oxygen also charged
    --> Sec (selenocysteine) has a side chain pKa of about ~5.43, so at pH 7.4, ~98.9% is deprotonated (negative)
'''
charge_dict = {"D": -1.0, "E": -1.0, "C": -0.0968,
               "K": 1.0, "R": 1.0, "H": 0.0418,
               "B": -1.8882, "J": -1.8882, "O": -1.8882,
               "A": 0.0, "F": 0.0, "G": 0.0, "I": 0.0, "L": 0.0, "M": 0.0, "N": 0.0,
               "P": 0.0, "Q": 0.0, "S": 0.0, "T": 0.0, "V": 0.0, "W": 0.0, "Y": 0.0}

'''
Molecular weights of each amino acid side chain, calculated from atomic masses taken from the University of Alberta: 
https://www.chem.ualberta.ca/~massspec/atomic_mass_abund.pdf
'''
mol_weight_dict = {"A": 15.023475, "C": 46.995546, "D": 59.013305, "E": 73.028955, "F": 91.054775, "G": 1.007825,
                   "H": 81.045273, "I": 57.070425, "K": 72.081324, "L": 57.070425, "M": 75.026846, "N": 58.029289,
                   "P": 42.04695, "Q": 72.044939, "R": 100.087472, "S": 31.01839, "T": 45.03404, "V": 43.054775,
                   "W": 130.065674, "Y": 107.04969, "B": 110.984722, "J": 125.000372, "O": 187.016022}

'''Relative, arbitrary propensities for pi-stacking (and other aromatic interactions) by various residues'''
stacking_dict = {"A": 0.0, "C": 0.0, "D": 0.0, "E": 0.0, "F": 1.0, "G": 0.0, "H": 0.25, "I": 0.0,
                 "K": 0.0, "L": 0.0, "M": 0.0, "N": 0.0, "P": 0.0, "Q": 0.0, "R": 0.0, "S": 0.0,
                 "T": 0.0, "V": 0.0, "W": 0.5, "Y": 0.75, "B": 0.0, "J": 0.0, "O": 0.0}

'''
Lengths of alkyl spacers between the terminal functional group of a side chain and the peptide backbone
    --> For branched side chains, the longest path is taken
'''
chain_length_dict = {"A": 1, "C": 1, "D": 1, "E": 2, "F": 1, "G": 0, "H": 1, "I": 3, "K": 4, "L": 3, "M": 2, "N": 1,
                     "P": 3, "Q": 2, "R": 3, "S": 1, "T": 2, "V": 2, "W": 1, "Y": 1, "B": 1, "J": 2, "O": 1}

# Compositional dictionaries

phosphate_groups = {"B": 1, "J": 1, "O": 1,
                    "S": 0, "T": 0, "Y": 0,
                    "R": 0, "H": 0, "K": 0, "D": 0, "E": 0,
                    "N": 0, "Q": 0, "C": 0, "G": 0, "P": 0,
                    "A": 0, "V": 0, "I": 0, "L": 0, "M": 0, "F": 0, "W": 0}
alkyl_groups = {"A": 1, "V": 3, "I": 4, "L": 4, "M": 3,
                "F": 1, "Y": 1, "W": 1,
                "S": 1, "T": 2, "N": 1, "Q": 2, "C": 1, "P": 3, "G": 0,
                "R": 3, "H": 1, "K": 4, "D": 1, "E": 2,
                "B": 1, "J": 2, "O": 1}
alkyl_branches = {"V": 1, "I": 1, "L": 1, "T": 1, "J": 1,
                  "A": 0, "M": 0, "F": 0, "Y": 0, "W": 0, "S": 0, "N": 0, "Q": 0, "C": 0, "G": 0,
                  "P": 0, "R": 0, "H": 0, "K": 0, "D": 0, "E": 0, "B": 0, "O": 0}
longest_chain = {"A": 1, "V": 2, "I": 3, "L": 3, "M": 4, "F": 5, "Y": 6, "W": 7,
                 "S": 2, "T": 2, "N": 3, "Q": 4, "C": 2, "G": 0, "P": 3,
                 "R": 6, "H": 4, "K": 5, "D": 2, "E": 3,
                 "B": 4, "J": 4, "O": 8}

carbon_count = {"R": 4, "H": 4, "K": 4, "D": 2, "E": 3,
                "S": 1, "T": 2, "N": 2, "Q": 3, "C": 1, "G": 0, "P": 3,
                "A": 1, "V": 3, "I": 4, "L": 4, "M": 3, "F": 7, "Y": 7, "W": 9,
                "B": 1, "J": 2, "O": 7}
nitrogen_count = {"R": 3, "H": 2, "K": 1, "D": 0, "E": 0,
                  "S": 0, "T": 0, "N": 1, "Q": 1, "C": 0, "G": 0, "P": 0,
                  "A": 0, "V": 0, "I": 0, "L": 0, "M": 0, "F": 0, "Y": 0, "W": 1,
                  "B": 0, "J": 0, "O": 0}
oxygen_count = {"D": 2, "E": 2, "S": 1, "T": 1, "N": 1, "Q": 1, "Y": 1,
                "R": 0, "H": 0, "K": 0, "C": 0, "G": 0, "P": 0, "A": 0,
                "V": 0, "I": 0, "L": 0, "F": 0, "W": 0, "M": 0,
                "B": 4, "J": 4, "O": 4}
sulfur_count = {"M": 1, "C": 1,
                "R": 0, "H": 0, "K": 0, "D": 0, "E": 0, "S": 0, "T": 0, "N": 0, "Q": 0,
                "G": 0, "P": 0, "A": 0, "V": 0, "I": 0, "L": 0, "F": 0, "Y": 0, "W": 0,
                "B": 0, "J": 0, "O": 0}
nitrogen_ring_count = {"H": 1, "W": 1,
                       "R": 0, "K": 0, "D": 0, "E": 0, "S": 0, "T": 0, "N": 0, "Q": 0, "M": 0,
                       "G": 0, "C": 0, "P": 0, "A": 0, "V": 0, "I": 0, "L": 0, "F": 0, "Y": 0,
                       "B": 0, "J": 0, "O": 0}
aromatic_ring_count = {"F": 1, "Y": 1, "W": 1, "O": 1,
                       "R": 0, "H": 0, "K": 0, "D": 0, "E": 0, "S": 0, "T": 0, "N": 0, "Q": 0, "G": 0,
                       "P": 0, "C": 0, "A": 0, "V": 0, "I": 0, "L": 0, "M": 0, "B": 0, "J": 0}

# Perform scaling and construct a dictionary of dictionaries
chemical_characteristics = {"kyte-doolittle": kyte_doolittle_dict,
                            "huang_nau": huang_nau_dict,
                            "charges": charge_dict,
                            "mol_weights": mol_weight_dict,
                            "stacking_scores": stacking_dict,
                            "chain_lengths": chain_length_dict,
                            "phosphate_groups": phosphate_groups,
                            "alkyl_groups": alkyl_groups,
                            "alkyl_branches": alkyl_branches,
                            "longest_chain": longest_chain,
                            "carbon_count": carbon_count,
                            "nitrogen_count": nitrogen_count,
                            "oxygen_count": oxygen_count,
                            "sulfur_count": sulfur_count,
                            "nitrogen_ring_count": nitrogen_ring_count,
                            "aromatic_ring_count": aromatic_ring_count}

# Scale the lookup dicts; this serves a similar function to StandardScaler

scaled_chemical_characteristics = {}
for characteristic, lookup_dict in chemical_characteristics.items():
    max_abs_value = max(abs(value) for value in lookup_dict.values())
    scaled_lookup_dict = {amino_acid: value / max_abs_value for amino_acid, value in lookup_dict.items()}
    scaled_chemical_characteristics[characteristic] = scaled_lookup_dict

def encode_aa(amino_acid, use_scaled_values = True):
    '''
    Simple function that returns a tuple of encoded values representing an amino acid's chemical characteristics

    Args:
        amino_acid (str):         single-letter code representing the amino acid to be encoded; accepts standard 20 amino acids
                                  plus B=pSer, J=pThr, O=pTyr
        use_scaled_values (bool): whether to use scaled or original values

    Returns:
        encoded_list (list): a list of values, according to the following indices:
                                0:  Kyte-Doolittle hydrophobicity score
                                1:  Huang-Nau conformational flexibility scale
                                2:  Charge
                                3:  Molecular weight
                                4:  Stacking score
                                5:  Chain length
                                6:  Phosphate group count
                                7:  Alkyl carbon count (CH, CH2, CH3)
                                8:  Alkyl branch count
                                9:  Longest chain length
                                10: Carbon count
                                11: Nitrogen count
                                12: Oxygen count
                                13: Sulfur count
                                14: Nitrogen-containing ring count
                                15: Aromatic ring count
    '''

    encoded_list = []
    for dict_name in chemical_characteristics.keys():
        if use_scaled_values:
            lookup_dict = scaled_chemical_characteristics.get(dict_name)
        else:
            lookup_dict = chemical_characteristics.get(dict_name)

        value = lookup_dict.get(amino_acid)
        if value is None:
            raise ValueError(f"encode_aa error: could not find {amino_acid} in {dict_name}")

        encoded_list.append(value)

    return encoded_list

def encode_seq(sequence, scaling = True):
    '''
    General function for returning a list of encoded tuples representing amino acids in the inputted sequence

    Args:
        sequence (str): the amino acid sequence as a string of single-letter codes
        scaling (bool): whether to perform scaling to ensure that all parameters have similar value ranges

    Returns:
        encoded_seq (np.ndarray):  2D array of encoded values for each amino acid in the sequence; equal in length to
                                   the inputted sequence string
    '''

    encoded_seq = []
    for amino_acid in sequence:
        encoded_list = encode_aa(amino_acid = amino_acid, use_scaled_values = scaling)
        encoded_seq.append(encoded_list)

    encoded_seq = np.array(encoded_seq)

    return encoded_seq

def encode_seqs_2d(sequences_2d, scaling = True):
    '''
    Semi-vectorized function that encodes a set of fixed-length sequences represented by a 2D array

    Args:
        sequences_2d (np.ndarray): 2D array where each row is a peptide represented by an array of amino acids
        scaling (bool):            whether to scale the features; useful for neural network applications

    Returns:
        features_matrix (np.ndarray): arr of shape (seqs_2d.shape[0], len(chemical_characteristics) * seqs_2d.shape[1])
        characteristic_count (int):   number of chemical characteristics that are encoded per position
    '''

    feature_matrix_list = []
    characteristics_dicts = scaled_chemical_characteristics if scaling else chemical_characteristics
    characteristic_count = len(characteristics_dicts)

    unique_residues = np.unique(sequences_2d)
    for characteristic_dict in characteristics_dicts.values():
        encoded_characteristic_2d = np.full(shape=sequences_2d, fill_value=np.nan, dtype=float)
        for aa in unique_residues:
            encoded_characteristic_2d[sequences_2d == aa] = characteristic_dict[aa]
        feature_matrix_list.append(encoded_characteristic_2d)

    feature_matrix = np.hstack(feature_matrix_list)

    return feature_matrix, characteristic_count