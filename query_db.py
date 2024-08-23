import numpy as np
import pandas as pd
from tifffile import imread, imwrite, imshow
import matplotlib.pyplot as plt
import os
import json
import pickle
import tkinter as tk
from tkinter import messagebox
from Motif_Predictor.load_predictor_config import load_config

predictor_params = load_config(verbose=True)
default_db_path = predictor_params["db_params"]["db_path"]
cwd = os.getcwd()

def load_db(db_path = default_db_path):
    pkl_path = db_path.replace(".json", ".pkl")
    if os.path.exists(pkl_path):
        print(f"Loading {pkl_path}")
        with open(pkl_path, "rb") as file:
            data_dict = pickle.load(file)
    else:
        print(f"Loading {db_path}")
        with open(db_path, "r") as json_file:
            data_dict = json.load(json_file)

    correlated_db_path = db_path.split(".json")[0] + "_with_homologs.json"
    correlated_pkl_path = correlated_db_path.replace(".json", ".pkl")
    if os.path.exists(correlated_pkl_path):
        print(f"Loading {correlated_pkl_path}")
        with open(correlated_pkl_path, "rb") as file:
            correlated_dict = pickle.load(file)
    elif os.path.exists(correlated_db_path):
        print(f"Loading {correlated_db_path}")
        with open(correlated_db_path, "r") as json_file:
            correlated_dict = json.load(json_file)
    else:
        correlated_dict = None

    return data_dict, correlated_dict

def get_protein_lengths(predictor_params = predictor_params):
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

    return protein_lengths, motif_length

def extract_protein_gene_dict(data_dict):
    protein_gene_dict = {}
    gene_name_id_dict = {}

    for taxid, taxid_data_dict in data_dict.items():
        for ref_gene_id, ref_gene_dict in taxid_data_dict.items():
            for key in ref_gene_dict.keys():
                if key == "gene_name":
                    gene_name = ref_gene_dict.get(key)
                    if gene_name_id_dict.get(gene_name) is None:
                        gene_name_id_dict[gene_name] = [ref_gene_id]
                    else:
                        gene_name_id_dict[gene_name].append(ref_gene_id)
                else:
                    protein_id = key
                    protein_gene_dict[protein_id] = ref_gene_id

    return protein_gene_dict, gene_name_id_dict

default_species_taxids = [(9606, "Homo sapiens", "Human"),
                          (10090, "Mus musculus", "Mouse"),
                          (10116, "Rattus norvegicus", "Rat"),
                          (7955, "Danio rerio", "Zebrafish"),
                          (6239, "Caenorhabditis elegans", None),
                          (4932, "Saccharomyces cerevisiae", "Budding yeast"),
                          (4896, "Schizosaccharomyces pombe", "Fission yeast"),
                          (3702, "Arabidopsis thaliana", "Thale cress")]

class GeneQueryPopup(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Gene Query Input")

        # Initialize row counter
        self.current_row = 0

        # Labels and Entry fields
        self.create_label_entry("Ensembl Gene Name:", "gene_name")
        self.create_label_entry("Ensembl Gene ID:", "gene_id")
        self.create_label_entry("Ensembl Protein ID:", "protein_id")

        # TODO Dropdowns here
        self.create_label_entry("Query TaxID or Species (default = 9606):", "query_taxid", default="9606")
        self.create_label_entry("Homology TaxIDs or Species (comma-separated):", "homology_taxids")

        # Submit button
        submit_button = tk.Button(self, text="Submit", command=self.submit)
        submit_button.grid(row=self.current_row, column=1, pady=10)

        # Make dict for converting query to taxid if not directly given
        self.species_taxid_dict = {}
        for taxid, species_name, common_name in default_species_taxids:
            self.get_taxid_species(taxid, species_name, common_name)

    def create_label_entry(self, label_text, variable_name, default=""):
        label = tk.Label(self, text=label_text)
        label.grid(row=self.current_row, column=0, padx=10, pady=5, sticky=tk.W)
        entry = tk.Entry(self)
        entry.grid(row=self.current_row, column=1, padx=10, pady=5, sticky=tk.W)
        entry.insert(0, default)
        setattr(self, variable_name, entry)
        self.current_row += 1

    def get_taxid_species(self, taxid, species_name, common_name = None):
        species_name = species_name.capitalize().replace("_", " ")
        self.species_taxid_dict[species_name] = taxid

        genus, species = species_name.split(" ")
        abbreviated_name = f"{genus[0]}. {species}"
        abbreviated_name = abbreviated_name.capitalize()
        self.species_taxid_dict[abbreviated_name] = taxid

        if common_name is not None:
            common_name = common_name.capitalize()
            self.species_taxid_dict[common_name] = taxid

    def submit(self):
        gene_name = self.gene_name.get().strip()
        gene_id = self.gene_id.get().strip()
        protein_id = self.protein_id.get().strip()
        query_taxid = self.query_taxid.get().strip()

        homology_taxids = self.homology_taxids.get().strip()
        if len(homology_taxids) > 0:
            homology_taxids = homology_taxids.split(",")
            homology_taxids = [int(taxid) for taxid in homology_taxids]
        else:
            homology_taxids = []

        # Validate the required fields
        if not gene_name and not gene_id and not protein_id:
            messagebox.showerror("Input Error", "You must provide at least one of Ensembl Gene Name, Gene ID, or Protein ID.")
            return

        if not query_taxid:
            messagebox.showerror("Input Error", "Query TaxID is required.")
            return

        # If taxid fields were given as species names, convert them to numerical TaxIDs
        try:
            query_taxid_int = int(query_taxid)
        except:
            query_taxid = self.species_taxid_dict.get(query_taxid)

        if homology_taxids:
            for i, homolog_taxid in enumerate(homology_taxids):
                try:
                    homolog_taxid_int = int(homolog_taxid)
                except:
                    homolog_taxid = self.species_taxid_dict.get(homolog_taxid)
                    homology_taxids[i] = homolog_taxid

        # Assign data to self
        self.query_data = {
            "gene_name": gene_name,
            "gene_id": gene_id,
            "protein_id": protein_id,
            "query_taxid": query_taxid,
            "homology_taxids": homology_taxids
        }

        self.destroy()  # Close the popup if input is valid

class MotifDomainMap:
    def __init__(self, total_residues, height = 40, width = 2000):
        self.total_residues = total_residues
        self.arr = np.zeros(shape=(height, width, 4), dtype=float)
        midline_distance_from_edge = int((3/8) * height)
        midline_thickness = int((1/4) * height)
        midline_coords = (midline_distance_from_edge, midline_distance_from_edge + midline_thickness)
        self.arr[midline_coords[0]:midline_coords[1], :, 3] = 1 # set horizontal midline to opaque black

    def interpolate_color(self, color1, color2, t):
        t = np.clip(t, 0, 1)
        interpolated_color = color1 + t * (color2 - color1)
        return interpolated_color

    def add_motif(self, motif_start, motif_score, motif_len, min_thickness_ratio = 0.005, bottom_color = None, top_color = None):
        if motif_score > 0:
            distance_from_left = round((motif_start / self.total_residues) * self.arr.shape[1])
            motif_width = round(motif_len / self.total_residues)
            min_thickness = round(min_thickness_ratio * self.arr.shape[1])
            if motif_width >= min_thickness:
                motif_line_coords = (distance_from_left, distance_from_left + motif_width)
            else:
                delta_width = min_thickness - motif_width
                motif_line_coords = (distance_from_left - round(delta_width / 2),
                                     distance_from_left + motif_width + round(delta_width / 2))

            if bottom_color is None:
                bottom_color = np.array([0.75, 0.75, 0.75, 1.0])
            if top_color is None:
                top_color = np.array([0.0, 0.5, 1.0, 1.0])
            interpolated_color = self.interpolate_color(bottom_color, top_color, t=motif_score)

            self.arr[:, motif_line_coords[0]:motif_line_coords[1], :] = interpolated_color

    def show(self):
        imshow(self.arr)
        plt.show()

    def save(self, path):
        imwrite(path, self.arr)

def generate_novel_motif_map(gene_id, protein_id, protein_len, motif_len, query_dict, height = 40, width = 2000,
                             min_thickness_ratio = 0.005, display = True, save = True):
    # Function to generate a MotifDomainMap object for a protein with detected motifs, color-coded by strength

    motif_domain_map = None
    
    gene_results = query_dict.get(gene_id)
    if gene_results is not None:
        gene_name = gene_results.get("gene_name")
        protein_results = gene_results.get(protein_id)
        if protein_results is not None:
            motif_domain_map = MotifDomainMap(protein_len, height, width)

            novel_results = protein_results.get("novel")
            for novel_num_motif, motif_vals_dict in novel_results.items():
                start = int(motif_vals_dict.get("start"))
                masked_binding_score = motif_vals_dict.get("masked_binding_score")
                motif_domain_map.add_motif(start, masked_binding_score, motif_len, min_thickness_ratio)

            if display:
                motif_domain_map.show()
            if save:
                map_path = os.path.join(cwd, f"{gene_name}_{protein_id}_motif_map.tif")
                motif_domain_map.save(map_path)

    return motif_domain_map

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

def prompt_for_results(motif_len, protein_lengths_dict = None):
    # Create the main window to get the ID
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    popup = GeneQueryPopup(root)
    root.wait_window(popup)  # Wait until the popup window is closed

    query_data = popup.query_data.copy()
    root.destroy()

    # Query the dictionary
    gene_name = query_data.get("gene_name")
    gene_id = query_data.get("gene_id")
    protein_id = query_data.get("protein_id")
    query_taxid = query_data.get("query_taxid")
    homology_taxids = query_data.get("homology_taxids")

    # Define database to use
    if correlated_dict is None:
        query_dict = data_dict.get(int(query_taxid))
        if query_dict is None:
            query_dict = data_dict[str(query_taxid)]
        search_homologs = False
    else:
        query_dict = correlated_dict
        search_homologs = True

    # Search the database
    if protein_id:
        gene_id = protein_gene_dict.get(protein_id)
        print_entry(gene_id, protein_id, query_dict)
        if protein_lengths_dict is not None:
            protein_len = protein_lengths_dict.get(protein_id)
            generate_novel_motif_map(gene_id, protein_id, protein_len, motif_len, query_dict)
    elif gene_id:
        protein_ids = list(query_dict[gene_id].keys()) if query_dict.get(gene_id) is not None else []
        protein_ids.remove("gene_name")
        for protein_id in protein_ids:
            print_entry(gene_id, protein_id, query_dict)
            if protein_lengths_dict is not None:
                protein_len = protein_lengths_dict.get(protein_id)
                generate_novel_motif_map(gene_id, protein_id, protein_len, motif_len, query_dict)
    else:
        gene_ids = gene_name_id_dict.get(gene_name)
        if gene_ids is None:
            print(f"No results found.")
        elif len(gene_ids) > 1:
            print(f"Caution: \"{gene_name}\" matches multiple Ensembl gene IDs; showing results for each of them.")
        for gene_id in gene_ids:
            print(f"Current gene ID: {gene_id}")
            protein_ids = list(query_dict[gene_id].keys()) if query_dict.get(gene_id) is not None else []
            protein_ids.remove("gene_name")
            for protein_id in protein_ids:
                print_entry(gene_id, protein_id, query_dict)
                if protein_lengths_dict is not None:
                    protein_len = protein_lengths_dict.get(protein_id)
                    generate_novel_motif_map(gene_id, protein_id, protein_len, motif_len, query_dict)

if __name__ == "__main__":
    # Load the database
    data_dict, correlated_dict = load_db()
    protein_gene_dict, gene_name_id_dict = extract_protein_gene_dict(data_dict)
    protein_lengths, motif_len = get_protein_lengths()

    prompt_for_results(motif_len, protein_lengths)