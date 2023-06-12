# This workflow includes image processing, quantification, standardization, concatenation, and matrix-building

import numpy as np
import os
import pickle
from Matrix_Generator.standardize_and_concatenate import main_workflow as standardized_concatenate
from Matrix_Generator.make_pairwise_matrices import make_pairwise_matrices
from general_utils.general_utils import input_number

def get_data(output_folder=None, add_peptide_seqs=True,
             peptide_seq_cols=["Phos_Sequence", "No_Phos_Sequence", "BJO_Sequence"], verbose=False):
    # Get output folder if not provided
    if output_folder is None:
        user_output_folder = input("Enter the folder for saving data, or leave blank to use the working directory:  ")
        if user_output_folder != "":
            output_folder = user_output_folder
        else:
            output_folder = os.getcwd()

    # Get the standardized concatenated dataframe containing all of the quantified peptide spot data
    print("Processing and standardizing the SPOT image data...") if verbose else None
    data_df, percentiles_dict = standardized_concatenate(predefined_batch = True, add_peptide_seqs = add_peptide_seqs,
                                                         peptide_seq_cols = peptide_seq_cols)
    reindexed_data_df = data_df.reset_index(drop = False)
    reindexed_data_df.to_csv(os.path.join(output_folder, "standardized_and_concatenated_data.csv"))

    return reindexed_data_df, percentiles_dict

def main(slim_length, minimum_members = None, thres_tuple = None, points_tuple = None, always_allowed_dict = None,
         position_weights = None, output_folder = None, add_peptide_seqs = True,
         peptide_seq_cols = ["Phos_Sequence", "No_Phos_Sequence", "BJO_Sequence"],
         sequence_col = "No_Phos_Sequence", significance_col = "One_Passes",
         optimize_weights = True, position_copies = None, use_cached_data = False, verbose = True):

    if use_cached_data:
        cached_data_path = input("Enter path to pickled data:  ")
        with open(cached_data_path, "rb") as f:
            reindexed_data_df, percentiles_dict = pickle.load(f)

    # Get output folder if not provided
    if output_folder is None:
        user_output_folder = input("Enter the folder to output the matrices and scored data into, or leave blank to use the working directory:  ")
        if user_output_folder != "":
            output_folder = user_output_folder
        else:
            output_folder = os.getcwd()

    else:
        reindexed_data_df, percentiles_dict = get_data(output_folder = output_folder, add_peptide_seqs = add_peptide_seqs,
                                                       peptide_seq_cols = peptide_seq_cols, verbose = verbose)

        save_pickled_data = input("Save data as pickled tuple of (reindexed_data_df, percentiles_dict)? (Y/N)  ")
        if save_pickled_data == "Y":
            save_path = input("Enter path to output folder: ")
            save_path = os.path.join(save_path, "cached_ml_training_data.pkl")
            with open(save_path, "wb") as f:
                pickle.dump((reindexed_data_df, percentiles_dict), f)

    # Generate pairwise position-weighted matrices
    scored_data_df, pred_val_dict = make_pairwise_matrices(reindexed_data_df, percentiles_dict = percentiles_dict,
                                                           slim_length = slim_length, minimum_members = minimum_members,
						                                   thres_tuple = thres_tuple, points_tuple = points_tuple,
                                                           always_allowed_dict = always_allowed_dict,
                                                           position_weights = position_weights,
                                                           output_folder = output_folder, sequence_col = sequence_col,
                                                           significance_col = significance_col, make_calls = True,
                                                           optimize_weights = optimize_weights,
                                                           position_copies = position_copies, verbose = True)

        
    return scored_data_df, pred_val_dict

# If the script is executed directly, invoke the main workflow
if __name__ == "__main__":
    slim_length = input_number("Please enter the length of the short linear motif being studied:  ", "int")

    # Default position copies; this is arbitrary and will depend on the characteristics of your motif of interest
    position_copies = {0: 4,
                       1: 1,
                       2: 1,
                       3: 1,
                       4: 1,
                       5: 1,
                       6: 1,
                       7: 1,
                       8: 1,
                       9: 1,
                       10: 2}

    use_cached = input("Use cached pickled data from a previous run? (Y/N)  ")
    if use_cached == "Y":
        main(slim_length = slim_length, optimize_weights = True, position_copies = position_copies, use_cached_data = True)
    else:
        main(slim_length = slim_length, optimize_weights = True, position_copies = position_copies, use_cached_data = False)