import os
import yaml
import numpy as np

# Load configuration; use local version if present
cwd = os.getcwd()
local_path = os.path.join(os.getcwd(), "predictor_params_local.yaml")
default_path = os.path.join(os.getcwd(), "predictor_params.yaml")

def load_config(local_path = local_path, default_path = default_path):
    if os.path.exists(local_path):
        print(f"Found local config file at {local_path}")
        config_path = local_path
    else:
        print(f"Using general config file at {default_path}")
        config_path = default_path

    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    # Extract main dict
    predictor_params = config["predictor_params"]

    # Insert other dicts as values in the main dict
    predictor_params["protein_seqs_paths"] = config["protein_seqs_paths"]
    predictor_params["df_chunks"] = config["df_chunks"]
    predictor_params["homology_params"] = config["homology_params"]
    predictor_params["alphafold_params"] = config["alphafold_params"]

    # Convert arrays to numpy arrays
    similarity_position_weights = np.array(predictor_params["homology_params"]["similarity_position_weights"])
    predictor_params["homology_params"]["similarity_position_weights"] = similarity_position_weights
    for key, value in predictor_params['enforced_position_rules'].items():
        predictor_params["enforced_position_rules"][key] = np.array(value)

    return predictor_params