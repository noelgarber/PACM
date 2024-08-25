import tkinter as tk
from tkinter import messagebox

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