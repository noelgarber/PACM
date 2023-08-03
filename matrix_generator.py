# This workflow includes image processing, quantification, standardization, concatenation, and matrix-building

import os
import pickle
from Matrix_Generator.standardize_and_concatenate import main_workflow as standardized_concatenate
from Matrix_Generator.make_pairwise_matrices import main as make_pairwise_matrices
from Matrix_Generator.make_specificity_matrices import main as make_specificity_matrix
try:
    from Matrix_Generator.config_local import *
except:
    from Matrix_Generator.config import *

def get_data(image_params = image_params, output_folder = None, verbose = False):
    # Helper function to get quantified SPOT peptide array image data

    # Get output folder if not provided
    if output_folder is None:
        user_output_folder = input("Enter the folder for saving data, or leave blank to use the working directory:  ")
        output_folder = user_output_folder if user_output_folder != "" else os.getcwd()

    # Get the standardized concatenated dataframe containing all of the quantified peptide spot data
    print("Processing and standardizing the SPOT image data...") if verbose else None
    data_df, percentiles_dict = standardized_concatenate(image_params)
    reindexed_data_df = data_df.reset_index(drop = False)
    reindexed_data_df.to_csv(os.path.join(output_folder, "standardized_and_concatenated_data.csv"))

    return reindexed_data_df, percentiles_dict

def main(image_params = image_params, general_params = general_params, data_params = data_params,
         matrix_params = matrix_params, comparator_info = comparator_info, specificity_params = specificity_params,
         generate_context_matrices = True, generate_specificity_matrix = True, verbose = True):
    '''
    Main function for quantifying source data, generating context-aware matrices, and generating the specificity matrix

    Args:
        image_params (dict):                as defined in config.py
        general_params (dict):              as defined in config.py
        data_params (dict):                 as defined in config.py
        matrix_params (dict):               as defined in config.py
        comparator_info (dict):             as defined in config.py
        specificity_params (dict):          as defined in config.py
        use_cached_data (bool):             whether to use cached quantified data from a previous run
        generate_context_matrices (bool):   whether to generate context-aware position-weighted matrices for
                                            overall motif prediction
        generate_specificity_matrix (bool): whether to generate a specificity matrix
        verbose (bool):                     whether to display additional information in the command line

    Returns:
        results_tuple (tuple):              (scored_data_df,
                                            best_score_threshold, weighted_matrices_dict, best_fdr, best_for,
                                            specificity_weighted_matrix, specificity_statistics)
    '''

    use_cached_data = image_params.get("use_cached_data")

    # Define output folders
    if not image_params.get("output_folder") and not use_cached_data:
        image_params["output_folder"] = input("Enter the folder to output image quantification data to:  ")

    if not general_params.get("output_folder"):
        general_params["output_folder"] = input("Enter the folder to output position-weighted matrices to:  ")

    # Quantify SPOT peptide binding data
    if use_cached_data:
        cached_data_path = image_params.get("cached_data_path")
        if cached_data_path is None:
            cached_data_path = input("Enter the path to the cached image data:  ")
        with open(cached_data_path, "rb") as f:
            data_df, percentiles_dict = pickle.load(f)
    else:
        # Obtain and quantify the data
        image_output_folder = image_params.get("output_folder")
        data_df, percentiles_dict = get_data(image_params, image_output_folder, verbose)

        # Optionally save pickled quantified data for future runs
        save_pickled_data = image_params.get("save_pickled_data")
        if save_pickled_data:
            save_path = os.path.join(image_output_folder, "cached_ml_training_data.pkl")
            with open(save_path, "wb") as f:
                pickle.dump((data_df, percentiles_dict), f)

    if not generate_context_matrices and not generate_specificity_matrix:
        return data_df

    # Generate pairwise position-weighted matrices

    general_params["percentiles_dict"] = percentiles_dict

    if generate_context_matrices:
        pairwise_results = make_pairwise_matrices(data_df, general_params, data_params, matrix_params)
        best_fdr, best_for, best_score_threshold, scored_data_df = pairwise_results
        if not generate_specificity_matrix:
            scored_data_df.to_csv(os.path.join(general_params.get("output_folder"), "final_scored_data.csv"))
    else:
        scored_data_df = data_df.copy()

    # Generate specificity matrix and associated results as a SpecificityMatrix object
    if generate_specificity_matrix:
        specificity_matrix = make_specificity_matrix(scored_data_df, comparator_info, specificity_params, save = True)

    if generate_context_matrices and generate_specificity_matrix:
        return (scored_data_df, best_score_threshold, best_fdr, best_for, specificity_matrix)
    elif generate_context_matrices:
        return (scored_data_df, best_score_threshold, best_fdr, best_for)
    elif generate_specificity_matrix:
        return (scored_data_df, specificity_matrix)


# If the script is executed directly, invoke the main workflow
if __name__ == "__main__":
    generate_context_matrices = input("Generate context-aware position-weighted matrices? (Y/N)  ") == "Y"
    generate_specificity_matrix = input("Generate specificity position-weighted matrix? (Y/N)  ") == "Y"
    main(image_params, general_params, data_params, matrix_params, comparator_info, specificity_params,
         generate_context_matrices, generate_specificity_matrix)