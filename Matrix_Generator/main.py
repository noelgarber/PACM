# This workflow includes image processing, quantification, standardization, concatenation, and matrix-building

import numpy as np
import os
import pickle
import warnings

from Matrix_Generator.standardize_and_concatenate import main_workflow as standardized_concatenate

from Matrix_Generator.make_pairwise_matrices import main as make_pairwise_matrices
from Matrix_Generator.make_pairwise_matrices import default_general_params, default_data_params, default_matrix_params

from Matrix_Generator.make_specificity_matrices import main as make_specificity_matrix
from Matrix_Generator.make_specificity_matrices import default_comparator_info, default_specificity_params

from general_utils.general_utils import input_number, list_inputter

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

# Define the default image quantification parameters

default_image_params = {"output_folder": "",
                        "add_peptide_seqs": True,
                        "peptide_seq_cols": ["Phos_Sequence", "No_Phos_Sequence", "BJO_Sequence"],
                        "save_pickled_data": True}

def main(image_params = None, general_params = None, data_params = None, matrix_params = None, comparator_info = None,
         specificity_params = None, use_cached_data = False, generate_context_matrices = True,
         generate_specificity_matrix = True, verbose = True):
    '''
    Main function for quantifying source data, generating context-aware matrices, and generating the specificity matrix

    Args:
        image_params (dict):                SPOT peptide image quantification parameters for deriving binding data
                                                --> "output_folder" (str): the folder path where data should be saved
                                                --> "add_peptide_seqs" (bool): must be True when building matrices
                                                --> "peptide_seq_cols" (list): col names, e.g. "BJO_Sequence"
                                                --> "save_pickled_data" (bool): whether to pickle data for future re-use
        general_params (dict):              general parameters as defined in make_pairwise_matrices:
                                                Auto-generated params: "percentiles_dict", "always_allowed_dict"
                                                Given in default_general_params: "aa_charac_dict"
                                                Required always: "slim_length", "optimize_weights", "output_folder"
                                                Required if optimizing weights: "position_copies"
                                                Required if not optimizing weights: "position_weights", "make_calls"
        data_params (dict):                 parameters describing source data, as defined in make_pairwise_matrices:
                                                --> bait (str): the bait to use for matrix generation; defaults to best if left blank
                                                --> bait_signal_col_marker (str): keyword that marks columns in source_dataframe that
                                                     contain signal values; required only if bait is given
                                                --> best_signal_col (str): column name with best signal values; used if bait is None
                                                --> bait_pass_col (str): column name with pass/fail information
                                                --> pass_str (str): the string representing a pass in bait_pass_col, e.g. "Yes"
                                                --> seq_col (str): column name containing peptide sequences as strings
        matrix_params (dict):               dictionary of parameters that affect matrix-building behaviour, used in matrix-building:
                                                --> thresholds_points_dict (dict): dictionary where threshold_value --> points_value
                                                --> included_residues (list): the residues included for the current type-position rule
                                                --> amino_acids (tuple): the alphabet of amino acids to use when constructing the matrix
                                                --> min_members (int): the minimum number of peptides that must follow the current
                                                    type-position rule for the matrix to be built
                                                --> position_for_filtering (int): the position for the type-position rule being assessed
                                                --> clear_filtering_column (bool): whether to set values in the filtering column to zero
        comparator_info (dict):             parameters describing comparator groups used for specificity analysis:
                                                --> comparator_set_1 (list-like): 1st set of pooled baits; if blank, the user is prompted
                                                --> comparator_set_2 (list-like): 2nd set of pooled baits; if blank, the user is prompted
                                                --> seq_col (str): the name of the column in the source dataframe containing peptide seqs
                                                --> bait_pass_col (str): same as in data_params
                                                --> pass_str (str): same as in data_params
        specificity_params (dict):          dictionary of parameters that affect specificity matrix-building behaviour:
                                                --> thresholds (list-like): log2fc threshold values; if absent, the user is prompted
                                                --> matching_points (list_like): matching points for the thresholds; prompted if absent
                                                --> include_phospho (bool): whether to include phospho residues in the matrix, or collapse them
                                                --> predefined_weights (list_like): if not optimizing weights, these are the position weights
                                                --> optimize_weights (bool): whether to optimize weights by attempting to maximize linear R2
                                                --> position_copies (dict): dict used for generating permuted weights; sum of values must equal motif length
        use_cached_data (bool):             whether to use cached quantified data from a previous run
        generate_context_matrices (bool):   whether tp generate context-aware position-weighted matrices for overall motif prediction
        generate_specificity_matrix (bool): whether to generate a specificity matrix
        verbose (bool):                     whether to display additional information in the command line

    Returns:
        results_tuple (tuple):              (scored_data_df, best_conditional_weights, weighted_matrices_dict, motif_statistics, specificity_weighted_matrix, specificity_statistics)
    '''

    # Define a consistent output folder to be used everywhere
    image_params = image_params or default_image_params.copy()
    general_params = general_params or default_general_params.copy()
    image_output_folder = image_params.get("output_folder")
    matrix_output_folder = general_params.get("output_folder")
    if not image_output_folder and not matrix_output_folder:
        output_folder = input("Enter the folder to output data to:  ")
    elif not image_output_folder:
        output_folder = matrix_output_folder
        image_params["output_folder"] = matrix_output_folder
    else:
        output_folder = image_output_folder
        general_params["output_folder"] = image_output_folder

    # Quantify SPOT peptide binding data
    if use_cached_data:
        cached_data_path = input("Enter path to pickled data:  ")
        with open(cached_data_path, "rb") as f:
            data_df, percentiles_dict = pickle.load(f)
    else:
        # Define necessary arguments for getting data
        add_peptide_seqs, peptide_seq_cols = image_params.get("add_peptide_seqs"), image_params.get("peptide_seq_cols")

        # Obtain and quantify the data
        data_df, percentiles_dict = get_data(output_folder, add_peptide_seqs, peptide_seq_cols, verbose)

        # Optionally save pickled quantified data for future runs
        save_pickled_data = image_params.get("save_pickled_data")
        if save_pickled_data:
            save_path = os.path.join(output_folder, "cached_ml_training_data.pkl")
            with open(save_path, "wb") as f:
                pickle.dump((data_df, percentiles_dict), f)

    # Generate pairwise position-weighted matrices
    general_params["percentiles_dict"] = percentiles_dict
    general_params["output_folder"] = output_folder  # ensure same folder is used

    data_params = data_params or default_data_params.copy()
    matrix_params = matrix_params or default_matrix_params.copy()

    if generate_context_matrices:
        pairwise_results = make_pairwise_matrices(data_df, general_params, data_params, matrix_params, verbose)
        scored_data_df, conditional_matrix_weights, weighted_matrices_dict, motif_statistics = pairwise_results
        if not generate_specificity_matrix:
            return (scored_data_df, conditional_matrix_weights, weighted_matrices_dict, motif_statistics)
    elif not generate_specificity_matrix:
        return data_df

    # Get specificity matrix comparator info and ensure consistency with data_params
    comparator_info = comparator_info or default_comparator_info.copy()
    comparator_info["seq_col"] = data_params.get("seq_col")
    comparator_info["bait_pass_col"] = data_params.get("bait_pass_col")
    comparator_info["pass_str"] = data_params.get("pass_str")

    # Get specificity matrix params and ensure consistency with matrix_params
    specificity_params = specificity_params or default_specificity_params.copy()
    specificity_params["position_copies"] = general_params["position_copies"]

    # Generate specificity matrix and back-calculate scores
    if not generate_context_matrices:
        scored_data_df = data_df.copy()
    specificity_results = make_specificity_matrix(scored_data_df, comparator_info, specificity_params)

    specificity_score_values, specificity_weighted_matrix, linear_coef, linear_intercept, model_r2 = specificity_results
    scored_data_df["Specificity_Score"] = specificity_score_values

    linear_intercept_signed = "+"+str(linear_intercept) if linear_intercept >= 0 else "-"+str(linear_intercept)
    linear_equation = "y=" + str(linear_coef) + "x" + linear_intercept_signed

    specificity_statistics = {"coefficient": linear_coef, "intercept": linear_intercept,
                              "equation": linear_equation, "r2": model_r2}

    if generate_context_matrices and generate_specificity_matrix:
        return (scored_data_df, conditional_matrix_weights, weighted_matrices_dict, motif_statistics, specificity_weighted_matrix, specificity_statistics)
    elif generate_specificity_matrix:
        return (scored_data_df, specificity_weighted_matrix, specificity_statistics)

# If the script is executed directly, invoke the main workflow
if __name__ == "__main__":
    # Define image quantification parameters
    image_params = default_image_params.copy()
    use_cached = input("Use cached pickled data from a previous run? (Y/N)  ")
    use_cached_data = use_cached == "Y"
    if not use_cached_data:
        save_pickled = input("Save pickled data from this run for future use? (Y/N)  ")
        save_pickled_data = save_pickled == "Y"
        image_params["save_pickled_data"] = save_pickled_data
        print("For adding sequences to quantified data, the following columns are expected:", image_params.get("peptide_seq_cols"))
        different_seq_cols = input("Use different sequence columns? (Y/N)  ")
        if different_seq_cols == "Y":
            print("Enter sequence columns one at a time.")
            seq_cols = list_inputter("Next col name:  ")
            image_params["peptide_seq_cols"] = seq_cols

    # Define general, source data, and matrix-building params for context-aware matrix generation
    generate_context_matrices = input("Generate context-aware position-weighted matrices? (Y/N)  ") == "Y"
    if generate_context_matrices:
        general_params = default_general_params.copy()
        general_params["slim_length"] = input_number("Please enter the length of the short linear motif being studied:  ", "int")
        optimize_weights = input("Optimize context-aware matrix weights? (Y/N)  ") == "Y"
        general_params["optimize_weights"] = optimize_weights
        if not optimize_weights:
            weights_list = input("Enter a comma-delimited list of predefined weights:  ")
            weights_array = np.array(weights_list.split(",")).astype(float)
            general_params["position_weights"] = weights_array
        data_params = default_data_params.copy()
        matrix_params = default_matrix_params.copy()
    else:
        general_params = None
        optimize_weights = False
        data_params = None
        matrix_params = None

    # Define comparator info and specificity params for generating the specificity position-weighted matrix
    generate_specificity_matrix = input("Generate specificity position-weighted matrix? (Y/N)  ") == "Y"
    if generate_specificity_matrix:
        comparator_info = default_comparator_info.copy()
        specificity_params = default_specificity_params.copy()
        optimize_specificity_weights = input("Optimize specificity matrix weights? (Y/N)  ")
        optimize_specificity_weights = optimize_specificity_weights == "Y"
        specificity_params["optimize_weights"] = optimize_specificity_weights
        if not optimize_specificity_weights:
            specificity_weights_list = input("Enter a comma-delimited list of predefined weights:  ")
            specificity_weights_array = np.array(specificity_weights_list.split(",")).astype(float)
            specificity_params["predefined_weights"] = specificity_weights_array
    else:
        comparator_info = None
        specificity_params = None

    # Execute the main function
    main(image_params, general_params, data_params, matrix_params, comparator_info, specificity_params,
         use_cached_data, generate_context_matrices, generate_specificity_matrix)