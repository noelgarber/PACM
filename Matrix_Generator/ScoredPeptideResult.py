# Defines the ScoredPeptideResult class, which represents peptide scoring results from ConditionalMatrices objects

import numpy as np
import pandas as pd
import warnings
try:
    from Matrix_Generator.config_local import aa_charac_dict
except:
    from Matrix_Generator.config import aa_charac_dict

class ScoredPeptideResult:
    '''
    Class that represents the result of scoring peptides using ConditionalMatrices.score_peptides()
    '''
    def __init__(self, seqs_2d, slice_scores_subsets, positive_scores_2d, suboptimal_scores_2d, forbidden_scores_2d):
        '''
        Initialization function to generate the score values and assign them to self

        Args:
            seqs_2d  (np.ndarray):             2D array of single letter code amino acids, where each row is a peptide
            slice_scores_subsets (np.ndarray): array of span lengths in the motif to stratify scores by; e.g. if it is
                                               [6,7,2], then subset scores are derived for positions 1-6, 7:13, & 14:15
            positive_scores_2d (np.ndarray):   standardized predicted signal values for each residue for each peptide
            suboptimal_scores_2d (np.ndarray): suboptimal element scores for each residue for each peptide
            forbidden_scores_2d (np.ndarray):  forbidden element scores for each residue for each peptide
        '''

        # Check validity of slice_scores_subsets
        if slice_scores_subsets is not None:
            if slice_scores_subsets.sum() != positive_scores_2d.shape[1]:
                raise ValueError(f"ScoredPeptideResult error: slice_scores_subsets sum ({slice_scores_subsets.sum()}) "
                                 f"does not match axis=1 shape of 2D score arrays ({positive_scores_2d.shape[1]})")

        # Assign constituent sequences to self
        self.sequences_2d = seqs_2d

        # Assign predicted signals score values
        self.positive_scores_2d = positive_scores_2d
        self.positive_scores_raw = positive_scores_2d.sum(axis=1)
        divisor = self.positive_scores_raw.max() * self.positive_scores_2d.shape[1]
        self.positive_scores_adjusted = self.positive_scores_raw / divisor

        # Assign suboptimal element score values
        self.suboptimal_scores_2d = suboptimal_scores_2d
        self.suboptimal_scores = suboptimal_scores_2d.sum(axis=1)

        # Assign forbidden element score values
        self.forbidden_scores_2d = forbidden_scores_2d
        self.forbidden_scores = forbidden_scores_2d.sum(axis=1)

        # Assign sliced score values if slice_scores_subsets was given
        self.slice_scores_subsets = slice_scores_subsets
        self.slice_scores()
        self.stack_scores()

        # Encode source residues with integer values; these will be used for NN training
        self.encode_residues(aa_charac_dict)

    def slice_scores(self):
        # Function that generates scores based on sliced subsets of peptide sequences

        self.score_cols = ["Positive_Score_Adjusted", "Suboptimal_Element_Score", "Forbidden_Element_Score"]

        if self.slice_scores_subsets is not None:
            end_position = 0
            self.sliced_positive_scores = []
            self.sliced_suboptimal_scores = []
            self.sliced_forbidden_scores = []

            for subset in self.slice_scores_subsets:
                start_position = end_position
                end_position += subset
                suffix_str = str(start_position) + "-" + str(end_position)

                subset_positive_scores = self.positive_scores_2d[:, start_position:end_position + 1].sum(axis=1)
                self.sliced_positive_scores.append(subset_positive_scores)
                self.score_cols.append("Positive_Score_" + suffix_str)

                subset_suboptimal_scores = self.suboptimal_scores_2d[:, start_position:end_position + 1].sum(axis=1)
                self.sliced_suboptimal_scores.append(subset_suboptimal_scores)
                self.score_cols.append("Suboptimal_Score_" + suffix_str)

                subset_forbidden_scores = self.forbidden_scores_2d[:, start_position:end_position + 1].sum(axis=1)
                self.sliced_forbidden_scores.append(subset_forbidden_scores)
                self.score_cols.append("Forbidden_Score_" + suffix_str)

        else:
            warnings.warn(RuntimeWarning("slice_scores_subsets was not given, so scores have not been sliced"))
            self.sliced_positive_scores = None
            self.sliced_suboptimal_scores = None
            self.sliced_forbidden_scores = None

    def stack_scores(self):
        '''
        Helper function that constructs a 2D array of scores values as columns
        '''

        scores = [self.positive_scores_adjusted, self.suboptimal_scores, self.forbidden_scores]

        if self.slice_scores_subsets is not None:
            for positive_scores_slice in self.sliced_positive_scores:
                scores.append(positive_scores_slice)
            for suboptimal_scores_slice in self.sliced_suboptimal_scores:
                scores.append(suboptimal_scores_slice)
            for forbidden_scores_slice in self.sliced_forbidden_scores:
                scores.append(forbidden_scores_slice)

        # Stack the scores and also use them to construct a dataframe
        self.stacked_scores = np.stack(scores).T
        self.scored_df = pd.DataFrame(self.stacked_scores, columns = self.score_cols)

    def encode_residues(self, aa_charac_dict = aa_charac_dict):
        '''
        Function that creates an encoded representation of sequences by chemical group
        '''

        self.charac_arrays = {}
        for charac, member_list in aa_charac_dict.items():
            is_member = np.isin(self.sequences_2d, member_list)
            self.charac_arrays[charac] = is_member

        self.stacked_encoded_characs = np.hstack(list(self.charac_arrays.values()))