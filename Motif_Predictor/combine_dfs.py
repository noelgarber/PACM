import numpy as np
import pandas as pd
import os

def make_gene_df(input_df, verbose = True):
    '''
    Generates dataframe containing only one gene id per row

    Args:
        input_df (pd.DataFrame):  input dataframe

    Returns:
        unique_df (pd.DataFrame): dataframe with only one row for each gene ID
    '''

    print("Initializing dataframe with best protein hit per gene...") if verbose else None
    main_df = input_df.copy()

    novel_cols = [col for col in main_df.columns if "Novel" in col]
    # Sort for preferred proteins with better novel scores
    id_cols = ["ensembl_gene_id", "external_gene_name", "ensembl_transcript_id", "ensembl_peptide_id"]

    print("Generating dataframe with novel motifs by gene...") if verbose else None
    main_df.sort_values("Novel_1st_binding_motif_score", ascending=False, kind="stable", inplace=True)
    main_df.sort_values("Novel_1st_final_call", ascending=False, kind="stable", inplace=True)
    main_df.sort_values("Novel_1st_motif_topology_accessible", ascending=False, kind="stable", inplace=True)

    novel_df = main_df.loc[:, id_cols + novel_cols].copy()
    novel_base_cols = [col for col in novel_df.columns if "homolog" not in col]
    novel_base_cols.remove("ensembl_transcript_id")
    novel_base_cols.remove("external_gene_name")
    novel_homolog_cols = [col for col in novel_df.columns if "homolog" in col]

    # For each homologous TaxID, preferentially pick entries for which a homolog exists
    taxids = []
    for col in novel_homolog_cols:
        new_taxid = col.split("_homolog_")[0]
        if new_taxid not in taxids:
            taxids.append(new_taxid)
    print(f"Picking optimal entries for each TaxID: {taxids}") if verbose else None

    novel_taxid_dfs = []
    for taxid in taxids:
        taxid_cols = [col for col in novel_homolog_cols if taxid in col]
        novel_taxid_df = novel_df.loc[:,novel_base_cols+taxid_cols].copy()
        homolog_id_col = None
        for taxid_col in taxid_cols:
            if "_homolog_vs" in taxid_col and "id_best" in taxid_col:
                homolog_id_col = taxid_col
                break
        homolog_exists = np.logical_and(novel_taxid_df[homolog_id_col].notna(),
                                        novel_taxid_df[homolog_id_col].ne(""))
        novel_taxid_df[f"{taxid}_homolog_exists"] = homolog_exists
        novel_taxid_df.sort_values(f"{taxid}_homolog_exists", ascending=False, kind="stable", inplace=True)
        novel_taxid_df.drop_duplicates(subset="ensembl_gene_id", keep="first", inplace=True, ignore_index=True)
        novel_taxid_df.drop(f"{taxid}_homolog_exists", axis=1, inplace=True)

        host_cols = novel_base_cols.copy()
        host_cols.remove("ensembl_gene_id")
        matching_host_cols = {col: f"{taxid}_Host_Match_{col}" for col in host_cols}
        novel_taxid_df.rename(matching_host_cols, axis=1, inplace=True)

        novel_taxid_dfs.append(novel_taxid_df)

    # Merge dataframes for each TaxID for motifs found by the current model
    novel_df.drop(novel_homolog_cols, axis=1, inplace=True)
    novel_df.drop_duplicates(subset="ensembl_gene_id", keep="first", inplace=True, ignore_index=True)
    for novel_taxid_df in novel_taxid_dfs:
        novel_df = pd.merge(novel_df, novel_taxid_df, how="outer", on="ensembl_gene_id", validate="one_to_one")

    novel_df.rename({"ensembl_transcript_id": "ensembl_transcript_id_best_novel",
                     "ensembl_peptide_id": "ensembl_peptide_id_best_novel"}, axis=1, inplace=True)

    # Sort for preferred proteins with better classical scores
    classical_cols = [col for col in main_df.columns if "Classical" in col]
    if len(classical_cols) > 0:
        print("Generating dataframe with classical motifs by gene...") if verbose else None

        main_df.sort_values("Classical_1st_total_motif_score", ascending=True, kind="stable", inplace=True)
        main_df.sort_values("Classical_1st_motif_topology_accessible", ascending=False, kind="stable", inplace=True)

        classical_df = main_df.loc[:, id_cols + classical_cols].copy()
        classical_base_cols = [col for col in classical_df.columns if "homolog" not in col]
        classical_base_cols.remove("ensembl_transcript_id")
        classical_base_cols.remove("external_gene_name")
        classical_homolog_cols = [col for col in classical_df.columns if "homolog" in col]
        classical_taxid_dfs = []
        for taxid in taxids:
            taxid_cols = [col for col in classical_homolog_cols if taxid in col]
            classical_taxid_df = classical_df.loc[:,classical_base_cols+taxid_cols].copy()
            homolog_id_col = None
            for taxid_col in taxid_cols:
                if "_homolog_vs" in taxid_col and "id_best" in taxid_col:
                    homolog_id_col = taxid_col
                    break
            homolog_exists = np.logical_and(classical_taxid_df[homolog_id_col].notna(),
                                            classical_taxid_df[homolog_id_col].ne(""))
            classical_taxid_df[f"{taxid}_homolog_exists"] = homolog_exists
            classical_taxid_df.sort_values(f"{taxid}_homolog_exists", ascending=False, kind="stable", inplace=True)
            classical_taxid_df.drop_duplicates(subset="ensembl_gene_id", keep="first", inplace=True, ignore_index=True)
            classical_taxid_df.drop(f"{taxid}_homolog_exists", axis=1, inplace=True)

            host_cols = classical_base_cols.copy()
            host_cols.remove("ensembl_gene_id")
            matching_host_cols = {col: f"{taxid}_Host_Match_{col}" for col in host_cols}
            classical_taxid_df.rename(matching_host_cols, axis=1, inplace=True)

            classical_taxid_dfs.append(classical_taxid_df)

        classical_df.drop(classical_homolog_cols, axis=1, inplace=True)
        classical_df.drop_duplicates(subset="ensembl_gene_id", keep="first", inplace=True, ignore_index=True)
        for classical_taxid_df in classical_taxid_dfs:
            classical_df = pd.merge(classical_df, classical_taxid_df, how="outer", on="ensembl_gene_id",
                                    validate="one_to_one")

        classical_df.rename({"ensembl_transcript_id": "ensembl_transcript_id_best_classical",
                             "ensembl_peptide_id": "ensembl_peptide_id_best_classical"}, axis=1, inplace=True)

        print("Concatenating dataframes of gene-level motif hits...") if verbose else None
        unique_df = pd.merge(novel_df, classical_df, how="outer", on="ensembl_gene_id",
                             suffixes=("_novel", "_classical"), validate="one_to_one")

    else:
        unique_df = novel_df

    return unique_df

def fuse_dfs(csv_paths, verbose = True):
    '''
    This function constructs a combined dataframe from a list of CSVs, where each CSV has only one homology species

    Args:
        csv_paths (list|tuple): CSV paths to processed results for individual homology species

    Returns:
        combined_df (pd.DataFrame): combined dataframe
    '''

    # Generate beginning dataframe
    unique_protein_dfs = []
    for csv_path in csv_paths:
        print(f"Parsing: {csv_path}")
        homolog_df = pd.read_csv(csv_path)

        motif_prefixes = [col.split("_motif_topology_type")[0] for col in homolog_df.columns if "topology_type" in col]
        unique_ensembl_peptides = pd.unique(homolog_df["ensembl_peptide_id"]).tolist()

        # Apply column for topological accessibility
        print(f"\tAdding motif topological accessibility columns...")
        for motif_prefix in motif_prefixes:
            motif_col = motif_prefix + "_motif"
            topology_type_col = motif_prefix + "_motif_topology_type"
            topology_desc_col = motif_prefix + "_motif_topology_description"
            
            motif_isna = homolog_df[motif_col].isna()
            motif_isblank = homolog_df[motif_col].eq("")
            motif_exists = ~np.logical_or(motif_isna, motif_isblank)

            topo_type_isna = homolog_df[topology_type_col].isna()
            topo_type_isblank = homolog_df[topology_type_col].eq("")
            topo_type_exists = ~np.logical_or(topo_type_isna, topo_type_isblank)
            soluble = np.logical_and(~topo_type_exists, motif_exists)

            topo_accessible = np.logical_and(homolog_df[topology_type_col].eq("topological domain"),
                                             homolog_df[topology_desc_col].eq("Cytoplasmic"))

            accessible = np.logical_or(soluble, topo_accessible)
            accessible_col = topology_type_col.split("_type")[0] + "_accessible"
            insert_col_idx = homolog_df.columns.get_loc(topology_desc_col) + 1
            homolog_df.insert(insert_col_idx, accessible_col, accessible)

        # Initialize unique protein df
        current_ensembl_peptides = homolog_df["ensembl_peptide_id"].to_list()
        base_row_index_dict = {}
        for idx, peptide_id in enumerate(current_ensembl_peptides):
            if base_row_index_dict.get(peptide_id) is None:
                base_row_index_dict[peptide_id] = idx
        base_row_indices = [base_row_index_dict.get(protein_id) for protein_id in unique_ensembl_peptides]

        host_cols = [col for col in homolog_df.columns if "homolog" not in col and "Unnamed" not in col]
        host_cols.remove("refseq_peptide")
        host_cols.remove("refseq_peptide_predicted")

        unique_protein_df = homolog_df.loc[base_row_indices, host_cols].copy()
        unique_protein_df.reset_index(drop=True, inplace=True)

        # Collapse to one row per protein
        print(f"\tCollapsing to one row per protein...")
        all_homolog_cols = [col for col in homolog_df.columns if "homolog" in col]
        for motif_prefix in motif_prefixes:
            motif_homolog_cols = [col for col in all_homolog_cols if motif_prefix in col]

            # Sort by similarity and preferentially pick passing scores if they exist
            similarity_col = [col for col in motif_homolog_cols if "similarity" in col][0]
            homolog_df.sort_values(similarity_col, ascending=False, kind="stable", inplace=True)
            if "Novel" in motif_prefix or "Classical" not in motif_prefix:
                model_call_col = [col for col in motif_homolog_cols if "model_call" in col][0]
                homolog_df.sort_values(model_call_col, ascending=False, kind="stable", inplace=True)
            homolog_df.reset_index(drop=True, inplace=True)

            # Pick most similar match for each protein
            current_protein_ids = homolog_df["ensembl_peptide_id"].to_list()
            ref_row_index_dict = {}
            for idx, peptide_id in enumerate(current_protein_ids):
                if ref_row_index_dict.get(peptide_id) is None:
                    ref_row_index_dict[peptide_id] = idx
            ref_row_indices = [ref_row_index_dict[protein_id] for protein_id in unique_ensembl_peptides]

            new_data_df = homolog_df.loc[ref_row_indices,motif_homolog_cols].copy()
            new_data_df.reset_index(drop=True, inplace=True)
            if len(new_data_df) == len(unique_protein_df):
                unique_protein_df = pd.concat([unique_protein_df, new_data_df], axis=1)
            else:
                raise Exception(f"unique_protein_df length ({len(unique_protein_df)}) is different from new_data_df length ({len(new_data_df)})")

            # Refseq cols vary based on individual homolog matches, so a separate column per match is needed
            refseq_col = motif_homolog_cols[0].split("_vs")[0] + "_vs_" + motif_prefix + "_match_refseq"
            refseq_pred_col = motif_homolog_cols[0].split("_vs")[0] + "_vs_" + motif_prefix + "_match_refseq_predicted"

            refseq_cols_from = ["refseq_peptide","refseq_peptide_predicted"]
            new_refseq_df = homolog_df.loc[ref_row_indices,refseq_cols_from].copy()
            new_refseq_df.reset_index(drop=True, inplace=True)
            new_refseq_df.rename({"refseq_peptide": refseq_col, "refseq_peptide_predicted": refseq_pred_col},
                                 axis=1, inplace=True)
            if len(new_data_df) == len(unique_protein_df):
                unique_protein_df = pd.concat([unique_protein_df, new_refseq_df], axis=1)
            else:
                raise Exception(f"unique_protein_df length ({len(unique_protein_df)}) is different from new_refseq_df length ({len(new_refseq_df)})")

        unique_protein_dfs.append(unique_protein_df)

    # Concatenate dataframes
    print("Combining homolog dataframes...") if verbose else None
    combined_df = unique_protein_dfs[0]
    for unique_protein_df in unique_protein_dfs[1:]:
        homolog_cols = [col for col in unique_protein_df.columns if "homolog" in col]
        homolog_cols = ["ensembl_peptide_id"] + homolog_cols
        combined_df = pd.merge(combined_df, unique_protein_df[homolog_cols],
                               how="outer", on="ensembl_peptide_id", validate="one_to_one")

    return combined_df

if __name__ == "__main__":
    csv_paths = []
    while True:
        path = input("Enter another path or leave blank if done:  ")
        if path != "":
            csv_paths.append(path)
        else:
            break

    print("Fusing dataframes together...")
    combined_df = fuse_dfs(csv_paths)

    parent_folder = csv_paths[0].rsplit("/",1)[0]
    save_path = os.path.join(parent_folder, "combined_datasets.csv")
    combined_df.to_csv(save_path)

    print("Making version with only one row per gene (best protein used)...")
    gene_df = make_gene_df(combined_df)
    gene_save_path = os.path.join(parent_folder, "combined_datasets_by_gene.csv")
    gene_df.to_csv(gene_save_path)