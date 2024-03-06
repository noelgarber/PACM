import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def main(path, similarity_col, homolog_id_col, likelihood_cols = None, strength_cols = None, call_cols = None):
    '''
    Plots associations between similarity scores and change in model scores

    Args:
        path (str):                        path to CSV containing the data
        similarity_col (str):              column in the CSV containing similarity scores
        homolog_id_col (str):              column in the CSV containing homolog IDs (if existing)
        likelihood_cols (list|tuple|None): tuple of (likelihood_host_col, likelihood_homolog_col)
        strength_cols (list|tuple|None):   tuple of (strength_host_col, strength_homolog_col)
        call_cols (list|tuple|None):       tuple of (host_call_col, homolog_call_col)

    Returns:
        None
    '''

    df = pd.read_csv(path)

    homolog_exists = np.logical_and(df[homolog_id_col].notna(), df[homolog_id_col].ne(""))
    df.drop(np.where(~homolog_exists)[0], axis=0, inplace=True)

    similarities = df[similarity_col].to_numpy()
    likelihood_deltas = df[likelihood_cols[1]].to_numpy() - df[likelihood_cols[0]].to_numpy()
    strength_deltas = df[strength_cols[1]].to_numpy() - df[strength_cols[0]].to_numpy()

    host_calls = df[call_cols[0]].to_numpy(dtype=bool)
    homolog_calls = df[call_cols[1]].to_numpy(dtype=bool)

    both_pass = np.logical_and(host_calls, homolog_calls)
    host_only = np.logical_and(host_calls, ~homolog_calls)
    homolog_only = np.logical_and(~host_calls, homolog_calls)
    neither_pass = np.logical_and(~host_calls, ~homolog_calls)

    similarities_both_pass = similarities[both_pass]
    similarities_host_only = similarities[host_only]
    similarities_homolog_only = similarities[homolog_only]
    similarities_neither_pass = similarities[neither_pass]

    likelihood_deltas_both_pass = likelihood_deltas[both_pass]
    likelihood_deltas_host_only = likelihood_deltas[host_only]
    likelihood_deltas_homolog_only = likelihood_deltas[homolog_only]
    likelihood_deltas_neither_pass = likelihood_deltas[neither_pass]

    strength_deltas_both_pass = strength_deltas[both_pass]
    strength_deltas_host_only = strength_deltas[host_only]
    strength_deltas_homolog_only = strength_deltas[homolog_only]
    strength_deltas_neither_pass = strength_deltas[neither_pass]

    both_count = len(similarities_both_pass)
    host_only_count = len(similarities_host_only)
    homolog_only_count = len(similarities_homolog_only)
    neither_count = len(similarities_neither_pass)

    colors = ["green", "blue", "purple", "red"]
    labels = [f"Both (n={both_count})", f"Host only (n={host_only_count})",
              f"Homolog only (n={homolog_only_count})", f"Neither (n={neither_count})"]
    handles = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]

    # Plot likelihood deltas vs. similarities
    plt.figure(figsize=(12,8))
    for i in reversed(np.arange(np.max([both_count, host_only_count, homolog_only_count, neither_count]))):
        if i < both_count:
            plt.scatter(likelihood_deltas_both_pass[i], similarities_both_pass[i], color="green", alpha=0.5)
        if i < host_only_count:
            plt.scatter(likelihood_deltas_host_only[i], similarities_host_only[i], color="blue", alpha=0.5)
        if i < homolog_only_count:
            plt.scatter(likelihood_deltas_homolog_only[i], similarities_homolog_only[i], color="purple", alpha=0.5)
        if i < neither_count:
            plt.scatter(likelihood_deltas_neither_pass[i], similarities_neither_pass[i], color="red", alpha=0.5)

    plt.xlabel("Δ Model Score (Homolog - Host)", fontsize=18)
    plt.ylabel("FFAT Core Similarity (BLOSUM62, normalized)", fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(handles=handles, fontsize=14)
    plt.show()

    # Plot strength deltas vs. similarities
    plt.figure(figsize=(12,8))
    for i in reversed(np.arange(np.max([both_count, host_only_count, homolog_only_count, neither_count]))):
        if i < both_count:
            plt.scatter(strength_deltas_both_pass[i], similarities_both_pass[i], color="green", alpha=0.5)
        if i < host_only_count:
            plt.scatter(strength_deltas_host_only[i], similarities_host_only[i], color="blue", alpha=0.5)
        if i < homolog_only_count:
            plt.scatter(strength_deltas_homolog_only[i], similarities_homolog_only[i], color="purple", alpha=0.5)
        if i < neither_count:
            plt.scatter(strength_deltas_neither_pass[i], similarities_neither_pass[i], color="red", alpha=0.5)

    plt.xlabel("Δ Binding Score (Homolog - Host)", fontsize=18)
    plt.ylabel("FFAT Core Similarity (BLOSUM62, normalized)", fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(handles=handles, fontsize=14)
    plt.show()

if __name__ == "__main__":
    with open("visualize_homology_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    path = config["settings"]["csv_path"]

    similarity_col = config["settings"]["similarity_col"]
    homolog_id_col = config["settings"]["homolog_id_col"]

    likelihood_host_col = config["settings"]["host_likelihood_col"]
    likelihood_homolog_col = config["settings"]["homolog_likelihood_col"]
    likelihood_cols = (likelihood_host_col, likelihood_homolog_col)

    strength_host_col = config["settings"]["host_binding_col"]
    strength_homolog_col = config["settings"]["homolog_binding_col"]
    strength_cols = (strength_host_col, strength_homolog_col)

    call_host_col = config["settings"]["host_call_col"]
    call_homolog_col = config["settings"]["homolog_call_col"]
    call_cols = (call_host_col, call_homolog_col)

    main(path, similarity_col, homolog_id_col, likelihood_cols, strength_cols, call_cols)