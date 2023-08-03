# This is a script describing the ForbiddenMatrix class and associated reject_seqs method for disallowed sequences

import numpy as np
import pandas as pd
import os
from scipy.stats import barnard_exact
from general_utils.general_utils import unravel_seqs, check_seq_lengths
from general_utils.matrix_utils import make_empty_matrix
try:
    from Matrix_Generator.config_local import matrix_params, aa_equivalence_dict
except:
    from Matrix_Generator.config import matrix_params, aa_equivalence_dict

class ForbiddenMatrix:
    # Class containing a boolean position matrix describing whether certain amino acids are forbidden based on position
    def __init__(self, motif_length, sequences, passes_bools, aa_equivalence_dict = aa_equivalence_dict,
                 matrix_params = matrix_params, verbose = False):
        '''
        Function for initializing the forbidden matrix based on source peptide data

        Args:
            motif_length (int):                the length of the motif being assessed
            sequences (np.ndarray):            array of peptide sequences of equal length matching motif_length
            passes_bools (np.ndarray):         array of bools representing whether the peptides are positive or not
            aa_equivalence_dict (dict):        dict of amino acid --> (tuple of functionally highly similar amino acids)
            matrix_params (dict):              dictionary of conditional matrix params described in config.py
            verbose (bool):                    whether to display status information during processing
        '''

        self.motif_length = motif_length

        # Construct the empty matrix dataframe
        amino_acids = matrix_params.get("amino_acids")
        self.matrix_df = make_empty_matrix(motif_length, amino_acids, dtype=bool)

        # Extract and unravel sequences
        check_seq_lengths(sequences, motif_length)  # check that all seqs in seq_col are the same length
        convert_phospho = not matrix_params.get("include_phospho")
        self.convert_phospho = convert_phospho
        sequences_2d = unravel_seqs(sequences, motif_length, convert_phospho)

        # Divide sequences based on whether they pass or not
        positive_sequences_2d = sequences_2d[passes_bools]
        negative_sequences_2d = sequences_2d[~passes_bools]

        # Iterate over matrix cols to assign boolean values (True = forbidden)
        positive_count = positive_sequences_2d.shape[0]
        negative_count = negative_sequences_2d.shape[0]

        for col_number in np.arange(positive_sequences_2d.shape[1]):
            positive_masked_col = positive_sequences_2d[:, col_number]
            negative_masked_col = negative_sequences_2d[:, col_number]

            position_forbidden_calls = np.full(shape=len(self.matrix_df.index), fill_value=False, dtype=bool)

            for i, aa in enumerate(self.matrix_df.index):
                # Count the number of occurrences of this aa in positive and negative peptides
                aa_positive_count = np.sum(positive_masked_col == aa)
                aa_negative_count = np.sum(negative_masked_col == aa)

                if aa_negative_count > 0 and aa_positive_count == 0:
                    # Get the functional identity group members for the amino acid in question
                    group_members = aa_equivalence_dict.get(aa)

                    # Count the number of occurrences in positive and negative peptides at this position
                    group_positive_count = np.sum(np.in1d(positive_masked_col, np.array(group_members, dtype="U")))
                    group_negative_count = np.sum(np.in1d(negative_masked_col, np.array(group_members, dtype="U")))

                    nongroup_positive_count = positive_count - group_positive_count
                    nongroup_negative_count = negative_count - group_negative_count

                    # Test whether enrichment of aa in negatives is significant compared to non-group aas
                    contingency_table = [[aa_positive_count, nongroup_positive_count],
                                         [aa_negative_count, nongroup_negative_count]]
                    p_value = barnard_exact(contingency_table).pvalue

                    if p_value <= 0.2:
                        # If there's a 4-in-5 chance that the residue is forbidden, consider it significantly forbidden
                        print(f"#{col_number}={aa} is forbidden at p={p_value}") if verbose else None
                        position_forbidden_calls[i] = True

                    else:
                        # If the aa is only found in negative peptides, but fails the Fisher exact test, test aa group
                        group_contingency_table = [[group_positive_count, nongroup_positive_count],
                                                   [group_negative_count, nongroup_negative_count]]
                        group_p_value = barnard_exact(group_contingency_table).pvalue
                        if group_p_value <= 0.2:
                            # If there's a 1-in-5 chance the aa's group is forbidden, consider that aa forbidden
                            forbidden_threshold = matrix_params["forbidden_threshold"]
                            if aa_negative_count >= forbidden_threshold:
                                if verbose:
                                    print(f"#{col_number}={aa} is forbidden (p={p_value}, group p={group_p_value})")
                                position_forbidden_calls[i] = True

            self.matrix_df.iloc[:, col_number] = position_forbidden_calls

        # Declare the index and array version
        self.index = self.matrix_df.index
        self.matrix_array = self.matrix_df.to_numpy()

    def predict_seqs(self, input_seqs):
        # Method that returns an array of bools matching the input array of peptide sequences

        if not isinstance(input_seqs, np.ndarray):
            input_seqs = np.array(input_seqs, dtype="U")

        input_seqs_2d = unravel_seqs(input_seqs, self.motif_length, self.convert_phospho)
        matrix_indices_2d = np.full(shape=input_seqs_2d.shape, fill_value=np.nan, dtype=int)

        unique_residues = np.unique(input_seqs_2d)
        unique_residue_indices = self.index.get_indexer_for(unique_residues)
        if np.any(unique_residue_indices == -1):
            raise IndexError("ForbiddenMatrix.predict_seqs() error: input_seqs has residues absent from the matrix")

        for aa, aa_index in zip(unique_residues, unique_residue_indices):
            matrix_indices_2d[input_seqs_2d == aa] = aa_index

        # Get boolean calls for forbidden residues present in each column
        forbidden_calls_2d = np.full(shape=input_seqs_2d.shape, fill_value=False, dtype=bool)
        for col_index in np.arange(self.matrix_array.shape[1]):
            col_aa_indices = matrix_indices_2d[:, col_index]
            col_bools = self.matrix_array[col_aa_indices, col_index]
            forbidden_calls_2d[:, col_index] = col_bools

        forbidden_calls = forbidden_calls_2d.any(axis=1)

        return forbidden_calls

    def save(self, output_folder):
        # Method to save the matrix dataframe to a specified output folder

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, "forbidden_residues_matrix.csv")
        output_matrix = self.matrix_df.astype(str)
        output_matrix.to_csv(output_path)