import os
import yaml
import numpy as np

def load_config(verbose = False):
    # Load configuration; use local version if present
    cwd = os.getcwd()
    if not "Motif_Predictor" in cwd:
        cwd = os.path.join(cwd.split("PACM")[0], "PACM/Motif_Predictor")
    local_path = os.path.join(cwd, "predictor_params_local.yaml")
    default_path = os.path.join(cwd, "predictor_params.yaml")

    if os.path.exists(local_path):
        print(f"Found local config file at {local_path}") if verbose else None
        config_path = local_path
    else:
        print(f"Using general config file at {default_path}") if verbose else None
        config_path = default_path

    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    # Extract main dict
    predictor_params = config["predictor_params"]

    # Insert other dicts as values in the main dict
    predictor_params["protein_seqs_paths"] = config["protein_seqs_paths"]
    predictor_params["df_chunks"] = config["df_chunks"]
    predictor_params["topo_params"] = config["topo_params"]
    predictor_params["homology_params"] = config["homology_params"]
    predictor_params["alphafold_params"] = config["alphafold_params"]

    # Convert arrays to numpy arrays
    similarity_position_weights = np.array(predictor_params["homology_params"]["similarity_position_weights"])
    predictor_params["homology_params"]["similarity_position_weights"] = similarity_position_weights
    for key, value in predictor_params['enforced_position_rules'].items():
        predictor_params["enforced_position_rules"][key] = np.array(value)

    return predictor_params