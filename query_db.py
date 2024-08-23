import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
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
    def __init__(self, total_residues, scaling_factor = 1.0, protein_id = None):
        self.total_residues = total_residues
        self.height = round(150 * scaling_factor)
        self.width = round(2000 * scaling_factor)
        self.scaling_factor = scaling_factor

        self.arr = np.ones(shape=(self.height, self.width, 3), dtype=float)
        midline_thickness = round(10 * scaling_factor)
        h1 = round(130 * scaling_factor)
        h2 = h1 + midline_thickness
        self.arr[h1:h2, :, :] = 0  # set horizontal midline to black

        self.protein_id = protein_id
        if protein_id:
            self.label_protein_id(protein_id)

    def label_protein_id(self, protein_id):
        # Place label text using Matplotlib
        protein_label = f"{protein_id}:"
        label_center_position = (round(20 * self.scaling_factor), round(10 * self.scaling_factor))

        # Use 'Agg' backend for off-screen rendering
        plt.switch_backend('Agg')
        plt.figure(figsize=(self.arr.shape[1] / 100, self.arr.shape[0] / 100), dpi=100)
        plt.imshow(self.arr)
        plt.text(label_center_position[1], label_center_position[0], protein_label, fontsize=20, ha='left',
                 va='center', color='black')
        plt.axis('off')

        # Convert plot to image array
        plt.gca().set_position([0, 0, 1, 1])  # Remove padding
        plt.gca().set_axis_off()  # Hide axes
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        # Render the canvas and convert to numpy array
        plt.gcf().canvas.draw()  # Force the canvas to render
        self.arr = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
        self.arr = self.arr.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
        self.arr = self.arr.astype(float) / 255
        plt.close()

    def interpolate_color(self, color1, color2, t):
        t = np.clip(t, 0, 1)
        interpolated_color = color1 + t * (color2 - color1)
        return interpolated_color

    def add_motif(self, motif_start, motif_seq, motif_score, motif_len, min_thickness_ratio=0.005,
                  bottom_color=None, top_color=None):
        if motif_score > 0:
            # Place the actual motif tick
            distance_from_left = round((motif_start / self.total_residues) * self.arr.shape[1])
            motif_width = round(motif_len / self.total_residues)
            min_thickness = round(min_thickness_ratio * self.arr.shape[1])
            if motif_width >= min_thickness:
                delta_width = 0
                w1 = distance_from_left
                w2 = distance_from_left + motif_width
            else:
                delta_width = min_thickness - motif_width
                w1 = distance_from_left - round(delta_width / 2)
                w2 = distance_from_left + motif_width + round(delta_width / 2)

            motif_height = round(30 * self.scaling_factor)
            h1 = self.arr.shape[0] - motif_height
            h2 = self.arr.shape[0]

            if bottom_color is None:
                bottom_color = np.array([0.75, 0.75, 0.75])
            if top_color is None:
                top_color = np.array([0.0, 0.5, 1.0])
            interpolated_color = self.interpolate_color(bottom_color, top_color, t=motif_score)

            self.arr[h1:h2, w1:w2, :] = interpolated_color

            # Place label text using Matplotlib
            position_label = f"range={motif_start}:{motif_start + motif_len - 1}"
            score_label = f"score={motif_score:.2f}"

            seq_center_position = (round(45 * self.scaling_factor), distance_from_left + round(motif_len / 2) - round(delta_width / 2))
            start_center_position = (round(75 * self.scaling_factor), distance_from_left + round(motif_len / 2) - round(delta_width / 2))
            score_center_position = (round(105 * self.scaling_factor), distance_from_left + round(motif_len / 2) - round(delta_width / 2))

            # Use 'Agg' backend for off-screen rendering
            plt.switch_backend('Agg')
            plt.figure(figsize=(self.arr.shape[1] / 100, self.arr.shape[0] / 100), dpi=100)
            plt.imshow(self.arr)
            plt.text(seq_center_position[1], seq_center_position[0], motif_seq, fontsize=18, ha='center', va='center', color='black')
            plt.text(start_center_position[1], start_center_position[0], position_label, fontsize=18, ha='center', va='center', color='black')
            plt.text(score_center_position[1], score_center_position[0], score_label, fontsize=18, ha='center', va='center', color='black')
            plt.axis('off')

            # Convert plot to image array
            plt.gca().set_position([0, 0, 1, 1])  # Remove padding
            plt.gca().set_axis_off()  # Hide axes
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            # Render the canvas and convert to numpy array
            plt.gcf().canvas.draw()  # Force the canvas to render
            self.arr = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
            self.arr = self.arr.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
            self.arr = self.arr.astype(float) / 255
            plt.close()

    def to_image(self):
        im = Image.fromarray(self.arr.astype('uint8'))
        return im

    def show(self):
        imshow(self.arr)
        plt.show()

    def save(self, path):
        imwrite(path, self.arr)

def generate_novel_motif_map(gene_id, protein_id, protein_len, motif_len, query_dict, scaling_factor = 1.0,
                             min_thickness_ratio = 0.005, display = False, save = True):
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
                masked_binding_score = motif_vals_dict.get("masked_binding_score")
                motif_seq = motif_vals_dict.get("sequence")
                if masked_binding_score > 0:
                    motif_domain_map.add_motif(start, motif_seq, masked_binding_score, motif_len, min_thickness_ratio)

            if display:
                motif_domain_map.show()
            if save:
                map_path = os.path.join(cwd, f"{gene_name}_{protein_id}_motif_map.tif")
                motif_domain_map.save(map_path)

    return motif_domain_map

def merge_motif_maps(motif_domain_maps, gene_name, sep_px = 1):

    top_padding = 80
    combined_height = sum([motif_domain_map.arr.shape[0] for motif_domain_map in motif_domain_maps.values()])
    combined_height += top_padding
    max_width = max([motif_domain_map.arr.shape[1] for motif_domain_map in motif_domain_maps.values()])
    
    # Generate the stacked images
    scaling_factor = 1.0
    merged_img = np.ones(shape=(combined_height, max_width, 3), dtype=float)
    top = top_padding
    for protein_id, motif_domain_map in motif_domain_maps.items():
        scaling_factor = motif_domain_map.scaling_factor
        bottom = top + motif_domain_map.arr.shape[0]
        left = 0
        right = motif_domain_map.arr.shape[1]
        arr = motif_domain_map.arr.copy()
        if sep_px > 0:
            arr[sep_px:2*sep_px,:,:] = 0
            arr[3*sep_px:4*sep_px,:,:] = 0
        merged_img[top:bottom, left:right, :] = arr
        top = bottom
    
    # Add the title
    title = f"{gene_name} Motifs by Protein Isoform"
    title_center_position = (round(10 * scaling_factor), round(merged_img.shape[1] / 2))

    # Use 'Agg' backend for off-screen rendering
    plt.switch_backend('Agg')
    plt.figure(figsize=(merged_img.shape[1] / 100, merged_img.shape[0] / 100), dpi=100)
    plt.imshow(merged_img)
    plt.text(title_center_position[1], title_center_position[0], title, fontsize=28, ha='center', va='center', color='black')
    plt.axis('off')

    # Convert plot to image array
    plt.gca().set_position([0, 0, 1, 1])  # Remove padding
    plt.gca().set_axis_off()  # Hide axes
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    # Render the canvas and convert to numpy array
    plt.gcf().canvas.draw()  # Force the canvas to render
    merged_img = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
    merged_img = merged_img.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
    merged_img = merged_img.astype(float) / 255
    plt.close()

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

def prompt_for_results(motif_len, protein_lengths_dict = None, gene_name_id_dict = {}, gene_id_name_dict = {}):
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
            motif_domain_map = generate_novel_motif_map(gene_id, protein_id, protein_len, motif_len, query_dict)
    elif gene_id:
        gene_name = gene_id_name_dict.get(gene_id)
        protein_ids = list(query_dict[gene_id].keys()) if query_dict.get(gene_id) is not None else []
        protein_ids.remove("gene_name")

        motif_domain_maps = {}
        for protein_id in protein_ids:
            print_entry(gene_id, protein_id, query_dict)
            if protein_lengths_dict is not None:
                protein_len = protein_lengths_dict.get(protein_id)
                motif_domain_map = generate_novel_motif_map(gene_id, protein_id, protein_len, motif_len, query_dict)
                motif_domain_maps[protein_id] = motif_domain_map

        merged_map_path = os.path.join(cwd, f"{gene_name}_merged_motif_maps.tif")
        merged_maps = merge_motif_maps(motif_domain_maps, gene_name, sep_px=2)
        imwrite(merged_map_path, merged_maps)

    else:
        gene_ids = gene_name_id_dict.get(gene_name)
        if gene_ids is None:
            print(f"No results found.")
        elif len(gene_ids) > 1:
            print(f"Caution: \"{gene_name}\" matches multiple Ensembl gene IDs; showing results for each of them.")

        motif_domain_maps = {}
        for gene_id in gene_ids:
            print(f"Current gene ID: {gene_id}")
            protein_ids = list(query_dict[gene_id].keys()) if query_dict.get(gene_id) is not None else []
            protein_ids.remove("gene_name")
            for protein_id in protein_ids:
                print_entry(gene_id, protein_id, query_dict)
                if protein_lengths_dict is not None:
                    protein_len = protein_lengths_dict.get(protein_id)
                    motif_domain_map = generate_novel_motif_map(gene_id, protein_id, protein_len, motif_len, query_dict)
                    motif_domain_maps[protein_id] = motif_domain_map

        merged_map_path = os.path.join(cwd, f"{gene_name}_merged_motif_maps.tif")
        merged_maps = merge_motif_maps(motif_domain_maps, gene_name, sep_px = 2)
        imwrite(merged_map_path, merged_maps)

if __name__ == "__main__":
    # Load the database
    data_dict, correlated_dict = load_db()
    protein_gene_dict, gene_name_id_dict, gene_id_name_dict = extract_protein_gene_dict(data_dict)
    protein_lengths, motif_len = get_protein_lengths()

    prompt_for_results(motif_len, protein_lengths, gene_name_id_dict, gene_id_name_dict)