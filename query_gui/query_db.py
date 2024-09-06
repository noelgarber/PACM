import numpy as np
import pandas as pd
from tifffile import imwrite
import os
import json
import pickle
import tkinter as tk
from Motif_Predictor.load_predictor_config import load_config
from query_gui.gene_query_popup import GeneQueryPopup
from query_gui.motif_mapper import render_text, MotifDomainMap

predictor_params = load_config(verbose=True)
default_db_path = predictor_params["db_params"]["db_path"]
cwd = os.getcwd()

def load_db(db_path = default_db_path):
    pkl_path = db_path.replace(".json", ".pkl")
    if os.path.exists(pkl_path):
        print(f"Loading pickled database, please wait...")
        with open(pkl_path, "rb") as file:
            data_dict = pickle.load(file)
    else:
        print(f"Loading JSON database, please wait...")
        with open(db_path, "r") as json_file:
            data_dict = json.load(json_file)
    print(f"\tLoaded.")

    correlated_db_path = db_path.split(".json")[0] + "_with_homologs.json"
    correlated_pkl_path = correlated_db_path.replace(".json", ".pkl")
    if os.path.exists(correlated_pkl_path):
        print(f"Loading pickled correlated homolog database, please wait...")
        with open(correlated_pkl_path, "rb") as file:
            correlated_dict = pickle.load(file)
        print(f"\tLoaded.")
    elif os.path.exists(correlated_db_path):
        print(f"Loading correlated homolog JSON database, please wait...")
        with open(correlated_db_path, "r") as json_file:
            correlated_dict = json.load(json_file)
        print(f"\tLoaded.")
    else:
        correlated_dict = None

    return data_dict, correlated_dict

def get_protein_lengths(predictor_params = predictor_params, save_pkl = True):
    # Get a dictionary of protein isoform lengths

    ref_protein_col = predictor_params["homology_params"]["ref_protein_col"]
    ref_seq_col = predictor_params["seq_col"]
    protein_seqs_paths = predictor_params["protein_seqs_paths"]
    motif_length = predictor_params["motif_length"]

    protein_lengths = {}
    for taxid, path in protein_seqs_paths.items():
        df = pd.read_csv(path)
        protein_ids = df[ref_protein_col].to_list()
        protein_seqs = df[ref_seq_col].to_list()
        for protein_id, seq in zip(protein_ids, protein_seqs):
            if isinstance(seq, str):
                protein_lengths[protein_id] = len(seq)

    if save_pkl:
        # Save pickled version; this will be used for a standalone package that relies on this dictionary
        protein_lengths_path = os.path.join(cwd, "protein_lengths_dict.pkl")
        with open(protein_lengths_path, "wb") as file:
            pickle.dump(protein_lengths, file)

    return protein_lengths, motif_length

def extract_protein_gene_dict(data_dict):
    protein_gene_dict = {}
    gene_name_id_dict = {}
    gene_id_name_dict = {}

    for taxid, taxid_data_dict in data_dict.items():
        for ref_gene_id, ref_gene_dict in taxid_data_dict.items():
            for key in ref_gene_dict.keys():
                if key == "gene_name":
                    gene_name = ref_gene_dict.get(key)
                    gene_id_name_dict[ref_gene_id] = gene_name
                    if gene_name_id_dict.get(gene_name) is None:
                        gene_name_id_dict[gene_name] = [ref_gene_id]
                    else:
                        gene_name_id_dict[gene_name].append(ref_gene_id)
                else:
                    protein_id = key
                    protein_gene_dict[protein_id] = ref_gene_id

    return protein_gene_dict, gene_name_id_dict, gene_id_name_dict

def generate_novel_motif_map(gene_id, protein_id, protein_len, motif_len, query_dict, scaling_factor=1.0,
                             min_thickness_ratio=0.01, color_ranges=None, opacity_range=(0,1),
                             display=False, save=True):
    # Function to generate a MotifDomainMap object for a protein with detected motifs, color-coded by strength

    motif_domain_map = None
    
    gene_results = query_dict.get(gene_id)
    if gene_results is not None:
        gene_name = gene_results.get("gene_name")
        protein_results = gene_results.get(protein_id)
        if protein_results is not None:
            motif_domain_map = MotifDomainMap(protein_len, scaling_factor, protein_id)

            novel_results = protein_results.get("novel")
            for novel_num_motif, motif_vals_dict in novel_results.items():
                start = int(motif_vals_dict.get("start"))
                score = motif_vals_dict.get("masked_binding_score")
                specificity_score = motif_vals_dict.get("specificity_score")
                seq = motif_vals_dict.get("sequence")
                if score > 0:
                    motif_domain_map.add_motif(start, seq, score, specificity_score, motif_len, min_thickness_ratio,
                                               tick_outline=5, color_ranges=color_ranges, opacity_range=opacity_range,
                                               legend_placement="bottom")

            if display:
                motif_domain_map.show()
            if save:
                map_path = os.path.join(cwd, f"{gene_name}_{protein_id}_motif_map.tif")
                motif_domain_map.save(map_path)

    return motif_domain_map

def merge_motif_maps(motif_domain_maps, gene_name, title_fontsize = 48, top_padding = 80, sep_padding = 60):
    # Merge maps into one image for the gene of interest

    merged_scaling_factor = 0
    max_width = max([motif_domain_map.get_arr().shape[1] for motif_domain_map in motif_domain_maps.values()])
    combined_height = sum([motif_domain_map.get_arr().shape[0] for motif_domain_map in motif_domain_maps.values()])
    for protein_id, motif_domain_map in motif_domain_maps.items():
        scaling_factor = motif_domain_map.scaling_factor
        sep_px = round(sep_padding * scaling_factor)
        combined_height += sep_px
        if scaling_factor > merged_scaling_factor:
            merged_scaling_factor = scaling_factor

    top_padding = round(top_padding * merged_scaling_factor)
    combined_height += top_padding

    # Generate the stacked images
    merged_img = np.ones(shape=(combined_height, max_width, 3), dtype=float)
    top = top_padding
    for protein_id, motif_domain_map in motif_domain_maps.items():
        scaling_factor = motif_domain_map.scaling_factor
        sep_px = round(sep_padding * scaling_factor)

        bottom = top + motif_domain_map.get_arr().shape[0] + sep_px
        left = 0
        right = motif_domain_map.get_arr().shape[1]
        arr = motif_domain_map.get_arr().copy()

        padded_arr = np.ones(shape=(arr.shape[0] + sep_px, arr.shape[1], arr.shape[2]), dtype=float)
        padded_arr[sep_px:,:,:] = arr
        merged_img[top:bottom, left:right, :] = padded_arr

        top = bottom
    
    # Add the title
    title = f"{gene_name} Motifs by Protein Isoform"
    scaled_title_fontsize = round(title_fontsize * merged_scaling_factor)
    title_arr = render_text(title, scaled_title_fontsize, use_bold=True)
    title_top = 0
    title_bottom = title_top + title_arr.shape[0]
    title_left = round(merged_img.shape[1] / 2) - round(title_arr.shape[1] / 2)
    title_right = title_left + title_arr.shape[1]
    merged_img[title_top:title_bottom, title_left:title_right, :] = title_arr

    return merged_img

def print_entry(gene_id, protein_id, query_dict):
    gene_results = query_dict.get(gene_id)
    if gene_results is not None:
        gene_name = gene_results.get("gene_name")
        print(f"----- Results for {gene_name} ({gene_id}) -----")
        protein_results = gene_results.get(protein_id)
        if protein_results is not None:
            print(f"\tNovel model results for isoform {protein_id}: ")
            novel_results = protein_results.get("novel")
            for key, value in novel_results.items():
                print(f"\t\t{key}:")
                for var_key, var_val in value.items():
                    print(f"\t\t\t{var_key}: {var_val}")
            classical_results = protein_results.get("classical")
            if classical_results is not None:
                print(f"\tClassical model results for isoform {protein_id}: ")
                for key, value in classical_results.items():
                    print(f"\t\t{key}:")
                    for var_key, var_val in value.items():
                        print(f"\t\t\t{var_key}: {var_val}")

def entry_to_file(path, gene_protein_ids, query_dict):
    lines = ["Query Results\n",
             "\n"]

    multiple_genes = len(gene_protein_ids) > 1
    for gene_id, protein_ids in gene_protein_ids.items():
        gene_results = query_dict.get(gene_id)
        if gene_results is not None:
            gene_name = gene_results.get("gene_name")
            lines.append(f"Gene: {gene_name} ({gene_id})\n")
            lines.append("\n")

            for protein_id in protein_ids:
                protein_results = gene_results.get(protein_id)
                if protein_results is not None:
                    isoform_desc = f"\tIsoform {protein_id}\n"
                    lines.append(isoform_desc)
                    lines.append("\n")

                    novel_results = protein_results.get("novel")
                    if len(novel_results) > 0:
                        lines.append(f"\t\tNovel model results:\n")
                        lines.append("\n")

                    for key, value in novel_results.items():
                        lines.append(f"\t\t\t{key}:\n")
                        for var_key, var_val in value.items():
                            lines.append(f"\t\t\t\t{var_key}: {var_val}\n")
                        lines.append("\n")

                    classical_results = protein_results.get("classical")
                    classical_results_count = len(classical_results) if classical_results is not None else 0
                    if classical_results_count > 0:
                        lines.append(f"\t\tClassical algorithm results:\n")
                        lines.append("\n")

                        for key, value in classical_results.items():
                            lines.append(f"\t\t\t{key}:\n")
                            for var_key, var_val in value.items():
                                lines.append(f"\t\t\t\t{var_key}: {var_val}\n")
                        lines.append("\n")

            if multiple_genes:
                lines.append("-" * 80)
                lines.append("\n")

    with open(path, "w") as file:
        file.writelines(lines)

    return lines

def user_prompt():
    # Initializes a GeneQueryPopup class to collect gene query info from the user

    root = tk.Tk()
    root.withdraw()  # Hide the root window
    popup = GeneQueryPopup(root)
    root.wait_window(popup)  # Wait until the popup window is closed

    try:
        query_data = popup.query_data.copy()

        gene_name = query_data.get("gene_name")
        gene_id = query_data.get("gene_id")
        protein_id = query_data.get("protein_id")
        query_taxid = query_data.get("query_taxid")
        homology_taxids = query_data.get("homology_taxids")

        root.destroy()
        valid_input_given = True

    except:
        root.destroy()
        gene_name, gene_id, protein_id, query_taxid, homology_taxids = None, None, None, None, None
        valid_input_given = False

    user_inputs = (gene_name, gene_id, protein_id, query_taxid, homology_taxids)

    return user_inputs, valid_input_given

def parse_user_input(protein_id, gene_id, gene_name, query_dict, protein_gene_dict, gene_name_id_dict = None,
                     gene_id_name_dict = None, warn_no_results = True, warn_multiple = True):
    '''
    Parses user input depending on what kind of identifier the user provided.

    Args:
        protein_id (str):         User input from Ensembl protein ID field
        gene_id (str):            User input from Ensembl gene ID field
        gene_name (str):          User input from gene name field (must be Ensembl name)
        query_dict (dict):        Dictionary of data, with or without homology information
        protein_gene_dict (dict): Dictionary of protein IDs and their corresponding gene IDs
        gene_name_id_dict (dict): Dictionary of gene names to corresponding gene IDs
        gene_id_name_dict (dict): Dictionary of gene IDs to corresponding gene names
        warn_no_results (bool):   Whether to print a warning message if no results were found for the identifier
        warn_multiple (bool):     Whether to print a warning message if a given gene name matches more than one gene

    Returns:
        gene_protein_ids (dict):  Dictionary of gene IDs and corresponding lists of protein IDs
        gene_name (str):          Gene name
        valid_input_given (bool): Whether the user has provided valid input
    '''

    valid_input_given = True

    if protein_id:
        gene_id = protein_gene_dict.get(protein_id)
        gene_protein_ids = {gene_id: [protein_id]}

    elif gene_id:
        gene_name = gene_id_name_dict.get(gene_id) if gene_id_name_dict is not None else None
        protein_ids = list(query_dict[gene_id].keys()) if query_dict.get(gene_id) is not None else []
        protein_ids.remove("gene_name")
        gene_protein_ids = {gene_id: protein_ids}

    elif gene_name:
        if gene_name_id_dict is None:
            raise ValueError(f"gene_name_id_dict was given as None, but must be given when the input is a gene name.")
        gene_ids = gene_name_id_dict.get(gene_name)
        if gene_ids is None:
            if warn_no_results:
                print(f"No results found.")
        elif len(gene_ids) > 1:
            if warn_multiple:
                print(f"Caution: \"{gene_name}\" matches multiple Ensembl gene IDs; showing results for each of them.")

        gene_protein_ids = {}
        for gene_id in gene_ids:
            print(f"Current gene ID: {gene_id}")
            protein_ids = list(query_dict[gene_id].keys()) if query_dict.get(gene_id) is not None else []
            if "gene_name" in protein_ids:
                protein_ids.remove("gene_name")
            gene_protein_ids[gene_id] = protein_ids

    else:
        gene_protein_ids = {}
        valid_input_given = False

    return gene_protein_ids, gene_name, valid_input_given

def select_db(data_dict, correlated_dict, query_taxid = None):
    '''
    Selects the database to use (with or without homolog information)

    Args:
        data_dict (dict):            Base dictionary (no homology info)
        correlated_dict (dict):      Data dictionary containing base info and homology information
        query_taxid (int):           Base taxonomic identifier given by the user

    Returns:
        query_dict (dict):      Either data_dict or correlated_dict, depending on which is valid
        search_homologs (bool): Whether to search homologs; corresponds to whether the correlated_dict was returned
    '''

    search_homologs = isinstance(correlated_dict, dict)
    if not search_homologs:
        query_dict = data_dict.get(int(query_taxid))
        if query_dict is None:
            query_dict = data_dict[str(query_taxid)]
    else:
        query_dict = correlated_dict

    return query_dict, search_homologs

def prompt_for_results(user_inputs, query_dict, motif_len, search_homologs, protein_lengths_dict = None,
                       names_to_ids = {}, ids_to_names = {}, print_to_terminal = True):
    '''
    Main function for prompting the user and collecting the results.

    Args:
        user_inputs (tuple):         Tuple of (gene_name, gene_id, protein_id, query_taxid, homology_taxids)
        query_dict (dict):           Dictionary of data, with or without homology information
        motif_len (int):             Length of the motif being analyzed
        search_homologs (bool):      Whether to search for homologous motifs
        protein_lengths_dict (dict): Dictionary of protein IDs and their corresponding lengths, needed for drawing maps
        names_to_ids (dict):         Dictionary of gene names to corresponding Ensembl gene IDs
        ids_to_names (dict):         Dictionary of Ensembl gene IDs to corresponding gene names

    Returns:
        valid_input_given (bool):    Whether the prompt received valid information that was parsed.
    '''

    # Create the main window to get the IDs and other information, then parse it
    gene_name, gene_id, protein_id, query_taxid, homology_taxids = user_inputs
    gene_protein_ids, gene_name, valid_input_given = parse_user_input(protein_id, gene_id, gene_name, query_dict,
                                                                      protein_gene_dict, names_to_ids, ids_to_names)

    # Search the database
    motif_domain_maps = {}
    if valid_input_given:
        text_results_path = os.path.join(cwd, f"{gene_name}_motif_results.txt")
        entry_to_file(text_results_path, gene_protein_ids, query_dict)

        for gene_id, protein_ids in gene_protein_ids.items():
            print(f"Current gene ID: {gene_id}")
            for protein_id in protein_ids:
                if print_to_terminal:
                    print_entry(gene_id, protein_id, query_dict)
                if protein_lengths_dict is not None:
                    protein_len = protein_lengths_dict.get(protein_id)
                    motif_domain_map = generate_novel_motif_map(gene_id, protein_id, protein_len, motif_len, query_dict)
                    motif_domain_maps[protein_id] = motif_domain_map

        merged_map_path = os.path.join(cwd, f"{gene_name}_merged_motif_maps.tif")
        merged_maps = merge_motif_maps(motif_domain_maps, gene_name)
        imwrite(merged_map_path, merged_maps)

    return motif_domain_maps, valid_input_given

if __name__ == "__main__":
    # Load the database
    data_dict, correlated_dict = load_db()
    protein_gene_dict, gene_name_id_dict, gene_id_name_dict = extract_protein_gene_dict(data_dict)
    protein_lengths, motif_len = get_protein_lengths()

    while True:
        user_inputs, valid_input_given = user_prompt()
        if valid_input_given:
            query_taxid = user_inputs[3]
            query_dict, search_homologs = select_db(data_dict, correlated_dict, query_taxid)
            motif_domain_maps, valid_input_given = prompt_for_results(user_inputs, query_dict, motif_len,
                                                                      search_homologs, protein_lengths,
                                                                      gene_name_id_dict, gene_id_name_dict)
        if not valid_input_given:
            break