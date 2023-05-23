# This workflow includes image processing, quantification, standardization, concatenation, and matrix-building

import os
from Matrix_Generator.standardize_and_concatenate import main_workflow as standardized_concatenate
from Matrix_Generator.make_pairwise_matrices import make_pairwise_matrices
from general_utils.general_utils import input_number

def main(slim_length, minimum_members = None, thres_tuple = None, points_tuple = None, always_allowed_dict = None,
         position_weights = None, output_folder = None, add_peptide_seqs = True,
         peptide_seq_cols = ["Phos_Sequence", "No_Phos_Sequence", "BJO_Sequence"],
         sequence_col = "No_Phos_Sequence", significance_col = "One_Passes"):

    # Get output folder if not provided
    if output_folder is None:
        user_output_folder = input(
            "Enter the folder to output the matrices and scored data into, or leave blank to use the working directory:  ")
        if user_output_folder != "":
            output_folder = user_output_folder
        else:
            output_folder = os.getcwd()

    # Get the standardized concatenated dataframe containing all of the quantified peptide spot data
    data_df, percentiles_dict = standardized_concatenate(predefined_batch = True, add_peptide_seqs = add_peptide_seqs,
                                                         peptide_seq_cols = peptide_seq_cols)
    reindexed_data_df = data_df.reset_index(drop = False)
    reindexed_data_df.to_csv(os.path.join(output_folder, "standardized_and_concatenated_data.csv"))

    # Generate pairwise position-weighted matrices
    scored_data_df = make_pairwise_matrices(dens_df = reindexed_data_df, percentiles_dict = percentiles_dict,
                                            slim_length = slim_length, minimum_members = minimum_members,
                                            thres_tuple = thres_tuple, points_tuple = points_tuple,
                                            always_allowed_dict = always_allowed_dict,
                                            position_weights = position_weights, output_folder = output_folder,
                                            sequence_col = sequence_col, significance_col = significance_col)

    return scored_data_df

# If the script is executed directly, invoke the main workflow
if __name__ == "__main__":
    slim_length = input_number("Please enter the length of the short linear motif being studied:  ", "int")
    main(slim_length = slim_length)