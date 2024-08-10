import os
import yaml
import numpy as np
import pandas as pd
import sqlite3
from Motif_Predictor.load_predictor_config import load_config

predictor_params = load_config(verbose=True)

# --------------------------------------------------- Schema Setup -----------------------------------------------------
def generate_schema_unpaired(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create table for top-level keys representing taxonomic identifiers (TaxID)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS TaxID (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL
    )
    """)

    # Create subordinate table for entries under each top-level key
    cur.execute("""
    CREATE TABLE IF NOT EXISTS GeneEntry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        top_level_key_id INTEGER NOT NULL,
        entry_name TEXT NOT NULL,
        FOREIGN KEY (top_level_key_id) REFERENCES TaxID(id)
    )
    """)

    # Create subordinate table for folders ("novel" and "classical") under each entry
    cur.execute("""
    CREATE TABLE IF NOT EXISTS Folder (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_id INTEGER NOT NULL,
        folder_name TEXT CHECK(folder_name IN ('novel', 'classical')) NOT NULL,
        FOREIGN KEY (entry_id) REFERENCES GeneEntry(id)
    )
    """)

    # Create subordinate table for novel model motifs
    cur.execute("""
    CREATE TABLE IF NOT EXISTS NovelMotif (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        folder_id INTEGER NOT NULL,
        motif_seq TEXT NOT NULL,
        topology_type TEXT NOT NULL,
        topology_desc TEXT,
        cytoplasmic_accessible BOOLEAN NOT NULL,
        classification_score REAL,
        binding_score REAL,
        masked_binding_score REAL,
        boolean_call BOOLEAN NOT NULL,
        specificity_score REAL,
        classical_score REAL,
        FOREIGN KEY (folder_id) REFERENCES Folder(id)
    )
    """)

    # Create subordinate table for classical model motifs (for when a classical algorithm is being compared)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ClassicalMotif (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        folder_id INTEGER NOT NULL,
        motif_seq TEXT NOT NULL,
        topology_type TEXT NOT NULL,
        topology_desc TEXT,
        cytoplasmic_accessible BOOLEAN NOT NULL,
        classical_score REAL,
        FOREIGN KEY (folder_id) REFERENCES Folder(id)
    )
    """)

    # Commit the changes
    conn.commit()
    conn.close()

def generate_schema_paired(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create table; the top-level key is the reference (host) taxonomic identifier (RefTaxID)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS RefTaxID (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL
    )
    """)

    # Create subordinate table; the key is the target (homologous) taxonomic identifier (TargetTaxID)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS TargetTaxID (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ref_taxid_key_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        FOREIGN KEY (ref_taxid_key_id) REFERENCES RefTaxID(id)
    )
    """)

    # Create subordinate table for gene entries
    cur.execute("""
    CREATE TABLE IF NOT EXISTS GeneEntry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        target_taxid_key_id INTEGER NOT NULL,
        entry_name TEXT NOT NULL,
        FOREIGN KEY (target_taxid_key_id) REFERENCES TargetTaxID(id)
    )
    """)

    # Create subordinate table for folders ("novel" and "classical") representing motif type
    cur.execute("""
    CREATE TABLE IF NOT EXISTS Folder (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_id INTEGER NOT NULL,
        folder_name TEXT CHECK(folder_name IN ('novel', 'classical')) NOT NULL,
        FOREIGN KEY (entry_id) REFERENCES GeneEntry(id)
    )
    """)

    # Create subordinate table for novel model motifs
    cur.execute("""
    CREATE TABLE IF NOT EXISTS NovelMotif (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        folder_id INTEGER NOT NULL,
        motif_seq TEXT NOT NULL,
        topology_type TEXT NOT NULL,
        topology_desc TEXT,
        cytoplasmic_accessible BOOLEAN NOT NULL,
        classification_score REAL,
        binding_score REAL,
        masked_binding_score REAL,
        boolean_call BOOLEAN NOT NULL,
        specificity_score REAL,
        classical_score REAL,
        homolog_gene_id TEXT NOT NULL,
        homolog_motif_seq TEXT NOT NULL,
        homolog_identity REAL,
        homolog_classification_score REAL,
        homolog_binding_score REAL,
        homolog_masked_binding_score REAL,
        homolog_boolean_call BOOLEAN NOT NULL,
        FOREIGN KEY (folder_id) REFERENCES Folder(id)
    )
    """)

    # Create subordinate table for classical model motifs (for when a classical algorithm is being compared)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ClassicalMotif (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        folder_id INTEGER NOT NULL,
        motif_seq TEXT NOT NULL,
        topology_type TEXT NOT NULL,
        topology_desc TEXT,
        cytoplasmic_accessible BOOLEAN NOT NULL,
        classical_score REAL,
        homolog_gene_id TEXT NOT NULL,
        homolog_motif_seq TEXT NOT NULL,
        homolog_identity REAL,
        homolog_classical_score REAL,
        FOREIGN KEY (folder_id) REFERENCES Folder(id)
    )
    """)

    # Commit the changes
    conn.commit()
    conn.close()

# ---------------------------------------- Functions to Populate the Database ------------------------------------------

# Function to insert into TaxID table
def insert_taxid(db_path, name):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("INSERT INTO TaxID (name) VALUES (?)", (name,))
    conn.commit()
    taxid_id = cur.lastrowid
    conn.close()
    return taxid_id

# Function to insert into RefTaxID table
def insert_ref_taxid(db_path, name):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("INSERT INTO RefTaxID (name) VALUES (?)", (name,))
    conn.commit()
    ref_taxid_id = cur.lastrowid
    conn.close()
    return ref_taxid_id

# Function to insert into TargetTaxID table
def insert_target_taxid(db_path, ref_taxid_id, name):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("INSERT INTO TargetTaxID (ref_taxid_key_id, name) VALUES (?, ?)", (ref_taxid_id, name))
    conn.commit()
    target_taxid_id = cur.lastrowid
    conn.close()
    return target_taxid_id

# Function to insert into GeneEntry table
def insert_gene_entry(db_path, target_taxid_id, entry_name):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("INSERT INTO GeneEntry (target_taxid_key_id, entry_name) VALUES (?, ?)", (target_taxid_id, entry_name))
    conn.commit()
    gene_entry_id = cur.lastrowid
    conn.close()
    return gene_entry_id

# Function to insert into Folder table
def insert_folder(db_path, gene_entry_id, folder_name):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("INSERT INTO Folder (entry_id, folder_name) VALUES (?, ?)", (gene_entry_id, folder_name))
    conn.commit()
    folder_id = cur.lastrowid
    conn.close()
    return folder_id

# Function to insert into NovelMotif table
def insert_novel_motif(db_path, folder_id, motif_seq, topology_type, topology_desc, cytoplasmic_accessible,
                       classification_score, binding_score, masked_binding_score, boolean_call,
                       specificity_score, classical_score = None, homolog_gene_id = None, homolog_motif_seq = None,
                       homolog_identity = None, homolog_classification_score = None, homolog_binding_score = None,
                       homolog_masked_binding_score = None, homolog_boolean_call = None):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    arg_names = ("folder_id, motif_seq, topology_type, topology_desc, cytoplasmic_accessible, classification_score, "
                 "binding_score, masked_binding_score, boolean_call, specificity_score, classical_score")
    args = [folder_id, motif_seq, topology_type, topology_desc, cytoplasmic_accessible, classification_score,
            binding_score, masked_binding_score, boolean_call, specificity_score, classical_score]
    if homolog_motif_seq is not None:
        homolog_arg_names = ("homolog_gene_id, homolog_motif_seq, homolog_identity, homolog_classification_score, "
                             "homolog_binding_score, homolog_masked_binding_score, homolog_boolean_call")
        arg_names = arg_names + homolog_arg_names
        args.extend([homolog_gene_id, homolog_motif_seq, homolog_identity, homolog_classification_score,
                     homolog_binding_score, homolog_masked_binding_score, homolog_boolean_call])
    question_marks = "? " * len(args)-1 + "?"
    cur.execute(f"INSERT INTO NovelMotif ({arg_names}) VALUES ({question_marks})",
                args)
    conn.commit()
    novel_motif_id = cur.lastrowid
    conn.close()
    return novel_motif_id

# Function to insert into ClassicalMotif table
def insert_classical_motif(db_path, folder_id, motif_seq, topology_type, topology_desc,
                           cytoplasmic_accessible, classical_score, homolog_gene_id = None, homolog_motif_seq = None,
                           homolog_identity = None, homolog_classical_score = None):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    arg_names = "folder_id, motif_seq, topology_type, topology_desc, cytoplasmic_accessible, classical_score"
    args = [folder_id, motif_seq, topology_type, topology_desc, cytoplasmic_accessible, classical_score]
    if homolog_motif_seq is not None:
        arg_names = arg_names + ", homolog_gene_id, homolog_motif_seq, homology_identity, homolog_classical_score"
        args.extend([homolog_gene_id, homolog_motif_seq, homolog_identity, homolog_classical_score])

    cur.execute(f"INSERT INTO ClassicalMotif ({arg_names}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", args)
    conn.commit()
    classical_motif_id = cur.lastrowid
    conn.close()
    return classical_motif_id

# ------------------------------------------ Parse DataFrame into Database ---------------------------------------------
def get_taxids(predictor_params = predictor_params):
    # Extracts reference and target taxids from params

    homolog_selection_mode = predictor_params["homology_params"]["homolog_selection_mode"]
    if homolog_selection_mode == "best":
        taxids = list(predictor_params["protein_seqs_paths"].keys())
        reference_taxid = taxids[0]
        target_taxids = taxids[1:]
    else:
        keys = list(predictor_params["protein_seqs_paths"].keys())
        reference_taxids = [key.split("_vs_")[0] for key in keys]
        if all([taxid == reference_taxids[0] for taxid in reference_taxids]):
            reference_taxid = int(reference_taxids[0])
        else:
            raise Exception(f"More than one reference taxid was found in the path keys: {reference_taxids}")
        target_taxids = [int(key.split("_vs_")[1]) for key in keys]

    return reference_taxid, target_taxids

def parse_dfs(csv_path, predictor_params = predictor_params):
    # Parse dataframe into taxid-specific dataframes

    df = pd.read_csv(csv_path)
    reference_taxid, target_taxids = get_taxids(predictor_params)

    # Get base cols for the reference taxid when homology is not being considered
    base_cols = []
    for col in df.columns:
        if all([not str(taxid) in col for taxid in target_taxids]):
            base_cols.append(col)

    # Get cols for each compared homologous taxid
    cols_by_taxid = {}
    for taxid in target_taxids:
        taxid_cols = [col for col in df.columns if str(taxid) in col]
        host_cols, homolog_cols = [], []
        if any(["Host" in col for col in taxid_cols]):
            if "Host" in col:
                host_cols.append(col)
            else:
                homolog_cols.append(col)
        cols_by_taxid[taxid] = (host_cols, homolog_cols)

    # Split dataframe into taxid-specific dataframes
    reference_df = df[base_cols].copy()
    target_taxid_dfs = {}
    for target_taxid in target_taxids:
        host_cols, homolog_cols = cols_by_taxid[target_taxid]
        host_taxid_df = df[host_cols].copy()
        homolog_taxid_df = df[homolog_cols].copy()
        target_taxid_dfs[target_taxid] = (host_taxid_df, homolog_taxid_df)

    return reference_taxid, reference_df, target_taxid_dfs

def apply_num_suffix(num):
    # Applies suffixes to numbers, i.e. 1st, 2nd, 3rd, 4th, etc.

    num = str(int(num))
    if num[-1] == "1":
        num_with_suffix = num + "st"
    elif num[-1] == "2":
        num_with_suffix = num + "nd"
    elif num[-1] == "3":
        num_with_suffix = num + "rd"
    else:
        num_with_suffix = num + "th"

    return num_with_suffix

def populate_unpaired_db(ref_gene_col, reference_taxid, reference_df, target_taxids, target_taxid_dfs, db_path,
                         novel_numbered, classical_numbered = None, compare_classical_method = False):
    # Populates unpaired database

    taxids = target_taxids.copy()
    taxids.insert(0, reference_taxid)
    gene_cols = {reference_taxid: ref_gene_col}
    for target_taxid in target_taxids:
        gene_cols[target_taxid] = f"{target_taxid}_best_homolog_id"

    dfs_by_taxid = target_taxid_dfs.copy()
    dfs_by_taxid[reference_taxid] = reference_df

    for taxid in taxids:
        df = dfs_by_taxid[taxid]
        taxid_id = insert_taxid(db_path, taxid)
        for i, row in df.iterrows():
            gene_col = gene_cols[taxid]
            ensembl_gene_id = row[gene_col]
            gene_key_id = insert_gene_entry(db_path, taxid_id, ensembl_gene_id)

            # Insert novel motifs into a table called novel
            novel_folder_id = insert_folder(db_path, gene_key_id, "novel")
            for novel_num in novel_numbered:
                val_col_names = [f"{novel_num}_motif", f"{novel_num}_motif_topology_type",
                                 f"{novel_num}_motif_topology_description", f"{novel_num}_total_motif_score",
                                 f"{novel_num}_binding_motif_score", f"{novel_num}_final_call",
                                 f"{novel_num}_motif_specificity_score"]
                if taxid != reference_taxid:
                    val_col_names = [f"{taxid}_{col_name}" for col_name in val_col_names]
                    classical_score_col = f"{taxid}_{novel_num}_classical_score"
                else:
                    classical_score_col = f"{novel_num}_classical_score"

                vals = row[val_col_names].to_list()
                masked_binding_score = vals[4] if vals[5] else 0.0
                vals.insert(5, masked_binding_score)
                classical_score = row[classical_score_col] if classical_score_col in df.columns else None
                vals.append(classical_score)

                insert_novel_motif(db_path, novel_folder_id, *vals)

            # Insert classical motifs into a table called classical
            if compare_classical_method:
                classical_folder_id = insert_folder(db_path, gene_key_id, "classical")
                for classical_num in classical_numbered:
                    val_col_names = [f"{classical_num}_motif", f"{classical_num}_motif_topology_type",
                                     f"{classical_num}_motif_topology_description",
                                     f"{classical_num}_total_motif_score"]
                    if taxid != reference_taxid:
                        val_col_names = [f"{taxid}_{col_name}" for col_name in val_col_names]
                    vals = row[val_col_names].to_list()
                    insert_classical_motif(db_path, classical_folder_id, *vals)


def populate_paired_db(ref_gene_col, reference_taxid, reference_df, target_taxids, target_taxid_dfs, db_path,
                       novel_numbered, classical_numbered = None, compare_classical_method = False):
    # Populates unpaired database

    taxids = target_taxids.copy()
    taxids.insert(0, reference_taxid)
    gene_cols = {reference_taxid: ref_gene_col}
    for target_taxid in target_taxids:
        gene_cols[target_taxid] = f"{target_taxid}_best_homolog_id"

    dfs_by_taxid = target_taxid_dfs.copy()
    dfs_by_taxid[reference_taxid] = reference_df

    ref_taxid_id = insert_ref_taxid(db_path, reference_taxid)
    for target_taxid in target_taxids:
        target_taxid_id = insert_target_taxid(db_path, ref_taxid_id, target_taxid)
        for i, row in reference_df.iterrows():
            ensembl_gene_id = row[ref_gene_col]
            gene_key_id = insert_gene_entry(db_path, target_taxid_id, ensembl_gene_id)

            # Insert novel motifs into a table called novel
            novel_folder_id = insert_folder(db_path, gene_key_id, "novel")
            novel_best_homolog_id = row[f"{target_taxid}_best_homolog_id"]
            if compare_classical_method:
                best_classical_homolog_id = row[f"{target_taxid}_classical_best_homolog_id"]
            else:
                best_classical_homolog_id = None

            for novel_num in novel_numbered:
                val_col_names = [f"{target_taxid}_Host_Match_{novel_num}_motif",
                                 f"{target_taxid}_Host_Match_{novel_num}_motif_topology_type",
                                 f"{target_taxid}_Host_Match_{novel_num}_motif_topology_description",
                                 f"{target_taxid}_Host_Match_{novel_num}_total_motif_score",
                                 f"{target_taxid}_Host_Match_{novel_num}_binding_motif_score",
                                 f"{target_taxid}_Host_Match_{novel_num}_final_call",
                                 f"{target_taxid}_Host_Match_{novel_num}_motif_specificity_score"]

                vals = row[val_col_names].to_list()
                masked_binding_score = vals[4] if vals[5] else 0.0
                vals.insert(5, masked_binding_score)

                classical_score_col = f"{target_taxid}_Host_Match_{novel_num}_classical_score"
                classical_score = row[classical_score_col] if classical_score_col in df.columns else None
                vals.append(classical_score)

                vals.append(novel_best_homolog_id)
                homolog_val_col_names = [f"{target_taxid}_{novel_num}_motif_homolog",
                                         f"{target_taxid}_{novel_num}_motif_homolog_identity",
                                         f"{target_taxid}_{novel_num}_motif_homolog_classification_score",
                                         f"{target_taxid}_{novel_num}_motif_homolog_binding_score",
                                         f"{target_taxid}_{novel_num}_motif_homolog_masked_binding_score",
                                         f"{target_taxid}_{novel_num}_motif_homolog_call"]
                homolog_vals = row[homolog_val_col_names].to_list()
                vals.extend(homolog_vals)

                insert_novel_motif(db_path, novel_folder_id, *vals)

            # Insert classical motifs into a table called classical
            if compare_classical_method:
                classical_folder_id = insert_folder(db_path, gene_key_id, "classical")
                for classical_num in classical_numbered:
                    val_col_names = [f"{classical_num}_motif", f"{classical_num}_motif_topology_type",
                                     f"{classical_num}_motif_topology_description",
                                     f"{classical_num}_total_motif_score"]
                    if taxid != reference_taxid:
                        val_col_names = [f"{taxid}_{col_name}" for col_name in val_col_names]
                    vals = row[val_col_names].to_list()

                    vals.append(best_classical_homolog_id)
                    homolog_val_col_names = [f"{target_taxid}_{classical_num}_motif_homolog",
                                             f"{target_taxid}_{classical_num}_motif_homolog_identity",
                                             f"{target_taxid}_{classical_num}_motif_homolog_score"]
                    homolog_vals = row[homolog_val_col_names].to_list()
                    vals.extend(homolog_vals)

                    insert_classical_motif(db_path, classical_folder_id, *vals)

def convert_to_sql(csv_path, predictor_params = predictor_params):
    # Main function for converting dataset to SQL database

    reference_taxid, reference_df, target_taxid_dfs = parse_dfs(csv_path, predictor_params = predictor_params)

    compare_classical_method = predictor_params["compare_classical_method"]
    return_count = predictor_params["return_count"]
    homolog_selection_mode = predictor_params["homology_params"]["homolog_selection_mode"]
    db_path = predictor_params["sql_params"]["db_path"]

    # Get substrings marking numbered novel and classical motifs
    nums_with_suffixes = [apply_num_suffix(num) for num in np.arange(1, return_count+1)]
    if compare_classical_method:
        novel_numbered = [f"Novel_{num_with_suffix}" for num_with_suffix in nums_with_suffixes]
        classical_numbered = [f"Classical_{num_with_suffix}" for num_with_suffix in nums_with_suffixes]
    else:
        novel_numbered = nums_with_suffixes
        classical_numbered = []

    # Generate SQL schema and insert data
    ref_gene_col = predictor_params["homology_params"]["ref_gene_col"]
    target_taxids = set(target_taxid_dfs.keys())
    if homolog_selection_mode == "best":
        '''When best motifs are picked for each comparator species irrehspective of alignment with best host motifs, 
        the SQL database is organized without trying to match host motifs to homolog motifs.'''
        generate_schema_unpaired(db_path)
        populate_unpaired_db(ref_gene_col, reference_taxid, reference_df, target_taxids, target_taxid_dfs, db_path,
                             novel_numbered, classical_numbered, compare_classical_method)
    else:
        generate_schema_paired(db_path)
        populate_paired_db(ref_gene_col, reference_taxid, reference_df, target_taxids, target_taxid_dfs, db_path,
                           novel_numbered, classical_numbered, compare_classical_method)

if __name__ == "__main__":
    csv_path = input(f"Enter the path to the gene-level processed dataset:  ")
    convert_to_sql(csv_path)