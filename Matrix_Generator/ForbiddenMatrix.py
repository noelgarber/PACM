# This is a script describing the ForbiddenMatrix class and associated reject_seqs method for disallowed sequences

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from general_utils.general_utils import unravel_seqs, check_seq_lengths
from general_utils.matrix_utils import make_empty_matrix

class ForbiddenMatrix:
    # Class containing a boolean position matrix describing whether certain amino acids are forbidden based on position
    def __init__(self, motif_length, source_df, residue_charac_dict,
                 data_params = data_params, matrix_params = matrix_params):
        '''
        Function for initializing the forbidden matrix based on source peptide data

        Args:
            motif_length (int):                the length of the motif being assessed
            source_df (pd.DataFrame):   dataframe containing peptide-protein binding data
            residue_charac_dict: dict of amino acid chemical characteristics
            data_params (dict):                dictionary of conditional matrix data-specific params from config.py
            matrix_params (dict):              dictionary of conditional matrix params described in config.py
        '''

        self.motif_length = motif_length

        # Construct the empty matrix dataframe
        amino_acids = matrix_params.get("amino_acids")
        self.matrix_df = make_empty_matrix(motif_length, amino_acids, dtype=bool)

        # Extract and unravel sequences
        seq_col = data_params.get("seq_col")
        sequences = source_df[seq_col].to_numpy()
        check_seq_lengths(source_df[seq_col], motif_length)  # check that all seqs in seq_col are the same length
        convert_phospho = not matrix_params.get("include_phospho")
        self.convert_phospho = convert_phospho
        sequences_2d = unravel_seqs(sequences, motif_length, convert_phospho)

        # Get boolean calls for whether each peptide binds to the bait(s) or not
        pass_str = data_params.get("pass_str")
        pass_col = data_params.get("bait_pass_col")
        pass_values = source_df[pass_col].to_numpy()
        pass_calls = pass_values == pass_str

        # Divide sequences based on whether they pass or not
        positive_sequences_2d = sequences_2d[pass_calls]
        negative_sequences_2d = sequences_2d[~pass_calls]

        # Iterate over matrix cols to assign boolean values (True = forbidden)
        positive_count = positive_sequences_2d.shape[0]
        negative_count = negative_sequences_2d.shape[1]

        for col_number in np.arange(positive_sequences_2d.shape[1]):
            positive_masked_col = positive_sequences_2d[:, col_number]
            negative_masked_col = negative_sequences_2d[:, col_number]

            position_forbidden_calls = np.full(shape=len(self.matrix_df.index), fill_value=False, dtype=bool)

            for i, aa in enumerate(self.matrix_df.index):
                aa_positive_count = np.sum(positive_masked_col == aa)
                other_positive_count = positive_count - aa_positive_count
                aa_negative_count = np.sum(negative_masked_col == aa)
                other_negative_count = negative_count - aa_negative_count

                if aa_negative_count > 0 and aa_positive_count == 0:
                    # Test whether the nonzero negative aa count is due to chance
                    contingency_table = [[aa_positive_count, other_positive_count],
                                         [aa_negative_count, other_negative_count]]
                    odds_ratio, p_value = fisher_exact(contingency_table)

                    # Set the threshold to 0.1, since we are interested only in using this as a filter
                    if p_value <= 0.1:
                        position_forbidden_calls[i] = True

            self.matrix_df.iloc[:, col_number] = position_forbidden_calls

        # Declare the index and array version
        self.index = self.matrix_df.index
        self.matrix_array = self.matrix_df.to_numpy()

    def predict_seqs(self, input_seqs):
        # Method that returns an array of bools matching the input array of peptide sequences

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