# This workflow includes image processing, quantification, standardization, concatenation, and matrix-building

import numpy as np
import os
import pickle
import warnings

from Matrix_Generator.standardize_and_concatenate import main_workflow as standardized_concatenate

from Matrix_Generator.make_pairwise_matrices import main as make_pairwise_matrices
from Matrix_Generator.make_pairwise_matrices import default_data_params, default_matrix_params, default_general_params

from Matrix_Generator.make_specificity_matrices import main as make_specificity_matrix
from Matrix_Generator.make_specificity_matrices import default_comparator_info, default_specificity_params

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

# Define the default parameters for main
default_data = {"slim_length": None,
                "position_copies": position_copies,
                "minimum_members": None,
                "thres_tuple": None,
                "points_tuple": None,
                "always_allowed_dict": None,
                "position_weights": None,
                "peptide_seq_cols": ["Phos_Sequence", "No_Phos_Sequence", "BJO_Sequence"],
                "sequence_col": "No_Phos_Sequence",
                "significance_col": "One_Passes"}

default_params = {"output_folder": None,
                  "specificity_points_mode": "discrete",
                  "add_peptide_seqs": True,
                  "optimize_weights": True,
                  "use_cached_data": False}


default_image_params = {"output_folder": "",
                        "add_peptide_seqs": True,
                        "peptide_seq_cols": ["Phos_Sequence", "No_Phos_Sequence", "BJO_Sequence"],
                        "save_pickled_data": True}

def main(image_params = None, general_params = None, data_params = None, matrix_params = None, comparator_info = None,
         specificity_params = None, use_cached_data = False, generate_specificity_matrix = True, verbose = True):

    # Quantify SPOT peptide binding data
    if use_cached_data:
        cached_data_path = input("Enter path to pickled data:  ")
        with open(cached_data_path, "rb") as f:
            data_df, percentiles_dict = pickle.load(f)
    else:
        # Define necessary arguments for getting data
        image_params = image_params or default_image_params.copy()
        add_peptide_seqs, peptide_seq_cols = image_params.get("add_peptide_seqs"), image_params.get("peptide_seq_cols")
        output_folder = image_params.get("output_folder")
        if output_folder == "" or output_folder is None:
            output_folder = input("Enter the folder to output data to:  ")

        # Obtain and quantify the data
        data_df, percentiles_dict = get_data(output_folder, add_peptide_seqs, peptide_seq_cols, verbose)

        # Optionally save pickled quantified data for future runs
        save_pickled_data = image_params.get("save_pickled_data")
        if save_pickled_data:
            save_path = os.path.join(output_folder, "cached_ml_training_data.pkl")
            with open(save_path, "wb") as f:
                pickle.dump((data_df, percentiles_dict), f)

    # Generate pairwise position-weighted matrices
    general_params = general_params or default_general_params.copy()
    data_params = data_params or default_data_params.copy()
    matrix_params = matrix_params or default_matrix_params.copy()

    pairwise_results = make_pairwise_matrices(data_df, general_params, data_params, matrix_params, verbose)
    scored_data_df, conditional_matrix_weights, weighted_matrices_dict, motif_statistics = pairwise_results
    if not generate_specificity_matrix:
        return scored_data_df, conditional_matrix_weights, weighted_matrices_dict, motif_statistics

    # Get specificity matrix comparator info and ensure consistency with data_params
    comparator_info = comparator_info or default_comparator_info.copy()
    comparator_info["seq_col"] = data_params.get("seq_col")
    comparator_info["bait_pass_col"] = data_params.get("bait_pass_col")
    comparator_info["pass_str"] = data_params.get("pass_str")

    # Get specificity matrix params and ensure consistency with matrix_params
    specificity_params = specificity_params or default_specificity_params.copy()
    specificity_params["position_copies"] = general_params["position_copies"]

    # Generate specificity matrix and back-calculate scores
    specificity_results = make_specificity_matrix(scored_data_df, comparator_info, specificity_params)
    specificity_score_values, specificity_weighted_matrix, linear_coef, linear_intercept, model_r2 = specificity_results
    scored_data_df["Specificity_Score"] = specificity_score_values

    linear_intercept_signed = "+"+str(linear_intercept) if linear_intercept >= 0 else "-"+str(linear_intercept)
    linear_equation = "y=" + str(linear_coef) + "x" + linear_intercept_signed

    specificity_statistics = {"coefficient": linear_coef, "intercept": linear_intercept,
                              "equation": linear_equation, "r2": model_r2}

    return (scored_data_df, best_conditional_weights, weighted_matrices_dict, motif_statistics, specificity_weighted_matrix, specificity_statistics)


# If the script is executed directly, invoke the main workflow
if __name__ == "__main__":
    data = default_data.copy()
    params = default_params.copy()

    slim_length = input_number("Please enter the length of the short linear motif being studied:  ", "int")
    data["slim_length"] = slim_length

    use_cached = input("Use cached pickled data from a previous run? (Y/N)  ")
    if use_cached == "Y":
        params["use_cached_data"] = True

    main(data, params, verbose = True)