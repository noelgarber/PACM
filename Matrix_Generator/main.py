# This workflow includes image processing, quantification, standardization, concatenation, and matrix-building

import os
import pickle
from Matrix_Generator.config import *
from Matrix_Generator.standardize_and_concatenate import main_workflow as standardized_concatenate
from Matrix_Generator.make_pairwise_matrices import main as make_pairwise_matrices
from Matrix_Generator.make_specificity_matrices import main as make_specificity_matrix

def get_data(output_folder = None, add_peptide_seqs = True,
             peptide_seq_cols = ["Phos_Sequence", "No_Phos_Sequence", "BJO_Sequence"], buffer_width = None,
             verbose = False):
    # Get output folder if not provided
    if output_folder is None:
        user_output_folder = input("Enter the folder for saving data, or leave blank to use the working directory:  ")
        output_folder = user_output_folder if user_output_folder != "" else os.getcwd()

    # Get the standardized concatenated dataframe containing all of the quantified peptide spot data
    print("Processing and standardizing the SPOT image data...") if verbose else None
    data_df, percentiles_dict = standardized_concatenate(predefined_batch = True, add_peptide_seqs = add_peptide_seqs,
                                                         peptide_seq_cols = peptide_seq_cols,
                                                         buffer_width = buffer_width)
    reindexed_data_df = data_df.reset_index(drop = False)
    reindexed_data_df.to_csv(os.path.join(output_folder, "standardized_and_concatenated_data.csv"))

    return reindexed_data_df, percentiles_dict

# Define the default image quantification parameters

def main(image_params = image_params, general_params = general_params, data_params = data_params,
         matrix_params = matrix_params, comparator_info = comparator_info, specificity_params = specificity_params,
         generate_context_matrices = True, generate_specificity_matrix = True, verbose = True):
    '''
    Main function for quantifying source data, generating context-aware matrices, and generating the specificity matrix

    Args:
        image_params (dict):                SPOT peptide image quantification parameters for deriving binding data
                                                --> "output_folder" (str): the folder path where data should be saved
                                                --> "add_peptide_seqs" (bool): must be True when building matrices
                                                --> "peptide_seq_cols" (list): col names, e.g. "BJO_Sequence"
                                                --> "save_pickled_data" (bool): whether to pickle data for future re-use
        general_params (dict):              as defined in make_pairwise_matrices.py
        data_params (dict):                 as defined in make_pairwise_matrices.py
        matrix_params (dict):               as defined in make_pairwise_matrices.py
        comparator_info (dict):             as defined in make_specificity_matrices.py
        specificity_params (dict):          as defined in make_specificity_matrices.py
        use_cached_data (bool):             whether to use cached quantified data from a previous run
        generate_context_matrices (bool):   whether to generate context-aware position-weighted matrices for
                                            overall motif prediction
        generate_specificity_matrix (bool): whether to generate a specificity matrix
        verbose (bool):                     whether to display additional information in the command line

    Returns:
        results_tuple (tuple):              (scored_data_df, best_conditional_weights, weighted_matrices_dict, motif_statistics, specificity_weighted_matrix, specificity_statistics)
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
        # Define necessary arguments for getting data
        add_peptide_seqs, peptide_seq_cols = image_params.get("add_peptide_seqs"), image_params.get("peptide_seq_cols")
        buffer_width = image_params.get("buffer_width")

        # Obtain and quantify the data
        image_output_folder = image_params.get("output_folder")
        data_df, percentiles_dict = get_data(image_output_folder, add_peptide_seqs, peptide_seq_cols, buffer_width, verbose)

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
        position_thresholds = general_params.get("position_thresholds")
        position_thresholds_str = ",".join(position_thresholds.astype(str))
        use_default_thresholds = input(f"For generating context-aware matrices, use default possible threshold values during optimization ({position_thresholds_str})? (Y/N)  ") == "Y"
        if not use_default_thresholds:
            position_thresholds = input("Enter comma-delimited possible threshold values:  ").split(",")
            general_params["position_thresholds"] = position_thresholds

        pairwise_results = make_pairwise_matrices(data_df, general_params, data_params, matrix_params)
        best_fdr, best_for, best_residue_thresholds, scored_data_df = pairwise_results

    # Generate specificity matrix and back-calculate scores
    if not generate_context_matrices:
        scored_data_df = data_df.copy()
    specificity_results = make_specificity_matrix(scored_data_df, comparator_info, specificity_params)

    scored_data_df = specificity_results[0]
    specificity_weights = specificity_results[1]
    specificity_weighted_matrix = specificity_results[3]
    specificity_statistics = {"equation": specificity_results[4], "coefficient": specificity_results[5],
                              "intercept": specificity_results[6], "r2": specificity_results[7]}

    # Save data that has not been saved already
    if generate_context_matrices or generate_specificity_matrix:
        scored_data_df.to_csv(os.path.join(general_params.get("output_folder"), "final_scored_data.csv"))
    if generate_specificity_matrix:
        specificity_weighted_matrix.to_csv(os.path.join(general_params.get("output_folder"), "specificity_weighted_matrix.csv"))

    # Display final report in the command line
    print("--------------------------------------------------------------------")
    print("                       Final Analysis Report                        ")
    if generate_context_matrices:
        print("                     -------------------------                      ")
        print("Context-aware position-weighted matrix residue thresholds:", best_residue_thresholds)
        print(f"Detected motif statistics: FDR = {best_fdr}, FOR = {best_for}")
    if generate_specificity_matrix:
        print("                     -------------------------                      ")
        print("Specificity position-weighted matrix weights:", specificity_weights)
        print("Specificity statistics: ")
        print(specificity_statistics)
    print("--------------------------------------------------------------------")


    if generate_context_matrices and generate_specificity_matrix:
        return (scored_data_df, best_residue_thresholds, best_fdr, best_for,
                specificity_weights, specificity_weighted_matrix, specificity_statistics)
    elif generate_context_matrices:
        return (scored_data_df, best_residue_thresholds, best_fdr, best_for)
    elif generate_specificity_matrix:
        return (scored_data_df, specificity_weights, specificity_weighted_matrix, specificity_statistics)


# If the script is executed directly, invoke the main workflow
if __name__ == "__main__":
    generate_context_matrices = input("Generate context-aware position-weighted matrices? (Y/N)  ") == "Y"
    generate_specificity_matrix = input("Generate specificity position-weighted matrix? (Y/N)  ") == "Y"
    main(image_params, general_params, data_params, matrix_params, comparator_info, specificity_params,
         generate_context_matrices, generate_specificity_matrix)