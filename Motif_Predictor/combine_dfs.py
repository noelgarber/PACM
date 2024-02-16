import numpy as np
import pandas as pd
import os

def make_unique_df(csv_path):
    '''
    This function constructs a combined dataframe from a list of CSVs, where each CSV has only one homology species

    Args:
        csv_path (str): CSV path to processed results

    Returns:
        unique_df (pd.DataFrame): dataframe with only one row per ensembl peptide id
        motif_cols (list):        list of motif seq col names
        motif_col_groups (list):  list of lists where each sub-list contains cols relating to a specific motif seq col
    '''

    base_df = pd.read_csv(csv_path)
    unnamed_cols = [col for col in base_df.columns if "Unnamed" in col]
    homolog_cols = [col for col in base_df.columns if "homolog" in col]
    drop_cols = unnamed_cols + homolog_cols
    if len(unnamed_cols) != 0:
        base_df.drop(drop_cols, axis=1, inplace=True)

    # Construct dataframe from unique Ensembl IDs
    unique_ensembl_peptides = base_df["ensembl_peptide_id"].unique().tolist()
    base_df.sort_values("external_gene_name", axis=0, ascending=False, inplace=True)

    gene_dict, transcript_dict, gene_name_dict = {}, {}, {}
    for i in np.arange(len(base_df)):
        ensembl_peptide_id = base_df.at[i, "ensembl_peptide_id"]
        gene_dict[ensembl_peptide_id] = base_df.at[i, "ensembl_gene_id"]
        transcript_dict[ensembl_peptide_id] = base_df.at[i, "ensembl_transcript_id"]
        gene_name_dict[ensembl_peptide_id] = base_df.at[i, "external_gene_name"]

    unique_ensembl_genes = [gene_dict[unique_peptide_id] for unique_peptide_id in unique_ensembl_peptides]
    unique_ensembl_transcripts = [transcript_dict[unique_peptide_id] for unique_peptide_id in unique_ensembl_peptides]
    unique_ensembl_names = [gene_name_dict[unique_peptide_id] for unique_peptide_id in unique_ensembl_peptides]

    unique_col_data = [unique_ensembl_genes, unique_ensembl_transcripts, unique_ensembl_peptides, unique_ensembl_names]
    unique_col_names = ["ensembl_gene_id", "ensembl_transcript_id", "ensembl_peptide_id", "external_gene_name"]
    unique_data_dict = {col_name: col_data for col_name, col_data in zip(unique_col_names, unique_col_data)}
    unique_df = pd.DataFrame(unique_data_dict)

    # Sort by topology of first motif
    topology_type_cols = [col for col in base_df.columns if "topology_type" in col]
    topology_desc_cols = [col for col in base_df.columns if "topology_description" in col]
    motif_cols = [col.split("_topology_type")[0] for col in topology_type_cols]

    other_cols = []
    motif_col_groups = [[] for i in np.arange(len(motif_cols))]
    for col in base_df.columns:
        for i, motif_col in enumerate(motif_cols):
            motif_col_prefix = motif_col.split("_motif")[0]
            if motif_col_prefix in col:
                motif_col_groups[i].append(col)
                break
        other_cols.append(col)

    zipped_cols = zip(motif_cols, motif_col_groups, topology_type_cols, topology_desc_cols)
    for i, (motif_col, motif_col_group, topology_type_col, topology_desc_col) in enumerate(zipped_cols):
        # Transfer all the motif col info
        for col in motif_col_group:
            if "topology" not in col:
                col_dict = {base_df.at[idx,"ensembl_peptide_id"]:base_df.at[idx,col] for idx in np.arange(len(base_df))}
                unique_col_vals = [col_dict.get(unique_peptide_id) for unique_peptide_id in unique_ensembl_peptides]
                unique_df[col] = unique_col_vals

        # Sort values by topology, favouring existent topology info
        base_df.sort_values(topology_type_col, axis=0, ascending=False, inplace=True)
        base_df.sort_values(topology_desc_col, axis=0, ascending=False, inplace=True)

        topology_type_dict, topology_desc_dict = {}, {}
        uniprot_dict, trembl_dict = {}, {}

        for j in np.arange(len(base_df)):
            ensembl_peptide_id = base_df.at[j, "ensembl_peptide_id"]

            if topology_type_dict.get(ensembl_peptide_id) is None:
                uniprot_dict[ensembl_peptide_id] = base_df.at[j, "uniprot"]
                trembl_dict[ensembl_peptide_id] = base_df.at[j, "trembl"]
                topology_type_dict[ensembl_peptide_id] = base_df.at[j, topology_type_col]
                topology_desc_dict[ensembl_peptide_id] = base_df.at[j, topology_desc_col]

        # Assign topology info
        topology_types = [topology_type_dict.get(unique_peptide_id) for unique_peptide_id in unique_ensembl_peptides]
        topology_descs = [topology_desc_dict.get(unique_peptide_id) for unique_peptide_id in unique_ensembl_peptides]
        source_uniprot_ids = [uniprot_dict.get(unique_peptide_id) for unique_peptide_id in unique_ensembl_peptides]
        source_trembl_ids = [trembl_dict.get(unique_peptide_id) for unique_peptide_id in unique_ensembl_peptides]

        # Determine whether topologically accessible to cytoplasm
        unique_df[topology_type_col] = topology_types
        unique_df[topology_desc_col] = topology_descs

        motif_exists = np.logical_and(np.logical_not(unique_df[motif_col].isna()),
                                      np.logical_not(unique_df[motif_col].eq("")))
        soluble = np.logical_and(np.logical_not(unique_df[topology_type_col].eq("")),
                                 np.logical_and(np.logical_not(unique_df[topology_type_col].isna()), motif_exists))
        topo_accessible = np.logical_and(unique_df[topology_type_col].eq("topological domain"),
                                         unique_df[topology_desc_col].eq("Cytoplasmic"))
        accessible = np.logical_or(soluble, topo_accessible)
        accessible_col = topology_type_col.split("_type")[0] + "_accessible"
        unique_df[accessible_col] = accessible

        # Assign source uniprot accessions to dataframe
        unique_df[motif_col + "_topology_source_uniprot_id"] = source_uniprot_ids
        unique_df[motif_col + "_topology_source_trembl_id"] = source_trembl_ids

    return unique_df, motif_cols, motif_col_groups

def make_gene_df(input_df):
    '''
    Generates dataframe containing only one gene id per row

    Args:
        input_df (pd.DataFrame):  input dataframe

    Returns:
        unique_df (pd.DataFrame): dataframe with only one row for each gene ID
    '''

    main_df = input_df.copy()

    unique_genes = pd.unique(main_df["ensembl_gene_id"])

    unique_df = pd.DataFrame(index=np.arange(len(unique_genes)), columns = main_df.columns)
    unique_df["ensembl_gene_id"] = unique_genes
    gene_name_row_indices = pd.Index(main_df["ensembl_gene_id"].to_list()).get_indexer_for(unique_genes)
    unique_df.loc[:, "external_gene_name"] = main_df.loc[gene_name_row_indices, "external_gene_name"]

    novel_cols = [col for col in main_df.columns if "Novel" in col]
    if len(novel_cols) > 0:
        # Sort for preferred proteins with better novel scores
        main_set1 = ["ensembl_transcript_id", "ensembl_peptide_id"] + novel_cols
        unique_set1 = ["ensembl_transcript_id_best_novel", "ensembl_peptide_id_best_novel"] + novel_cols

        main_df.sort_values("Novel_1st_binding_motif_score", ascending=False, inplace=True)
        main_df.sort_values("Novel_1st_binding_motif_score", ascending=False, inplace=True)
        main_df.sort_values("Novel_1st_motif_topology_accessible", ascending=False, inplace=True)
        main_df.reset_index(drop=True)

        novel_ref_row_indices = pd.Index(main_df["ensembl_gene_id"].to_list()).get_indexer_for(unique_genes)
        unique_df.loc[:, unique_set1] = main_df.loc[novel_ref_row_indices, main_set1]

        # Sort for preferred proteins with better classical scores
        classical_cols = [col for col in main_df.columns if "Classical" in col]
        main_set2 = ["ensembl_transcript_id", "ensembl_peptide_id"] + classical_cols
        unique_set2 = ["ensembl_transcript_id_best_classical", "ensembl_peptide_id_best_classical"] + classical_cols

        main_df.sort_values("Classical_1st_total_motif_score", ascending=True, inplace=True)
        main_df.sort_values("Classical_1st_motif_topology_accessible", ascending=False, inplace=True)
        main_df.reset_index(drop=True)

        classical_ref_row_indices = pd.Index(main_df["ensembl_gene_id"].to_list()).get_indexer_for(unique_genes)
        unique_df.loc[:, unique_set2] = main_df.loc[classical_ref_row_indices, main_set2]

        # Drop blank cols
        unique_df.drop(["ensembl_transcript_id", "ensembl_peptide_id"], axis=1, inplace=True)
        
    else:
        # Sort for preferred proteins with better novel scores
        assign_cols = [col for col in unique_df.columns if col != "ensembl_gene_id"]

        main_df.sort_values("1st_binding_motif_score", ascending=False, inplace=True)
        main_df.sort_values("1st_binding_motif_score", ascending=False, inplace=True)
        main_df.sort_values("1st_motif_topology_accessible", ascending=False, inplace=True)
        main_df.reset_index(drop=True)

        ref_row_indices = pd.Index(main_df["ensembl_gene_id"].to_list()).get_indexer_for(unique_genes)
        unique_df.loc[:, assign_cols] = main_df.loc[ref_row_indices, assign_cols]
    
    return unique_df

def fuse_dfs(csv_paths):
    '''
    This function constructs a combined dataframe from a list of CSVs, where each CSV has only one homology species

    Args:
        csv_paths (list|tuple): CSV paths to processed results for individual homology species

    Returns:
        combined_df (pd.DataFrame): combined dataframe
    '''

    # Generate beginning dataframe
    combined_df, motif_cols, motif_col_groups = make_unique_df(csv_paths[0])

    # Process and organize homologs
    for csv_path in csv_paths:
        current_df = pd.read_csv(csv_path)
        for col in current_df.columns:
            if "homolog" in col and "id_best" in col:
                homolog_col_prefix = col.split("_id_best")[0]
                current_df.sort_values(col, axis=0, ascending=False, inplace=True)

                refseq_dict, refseq_predicted_dict = {}, {}
                homolog_ids_dict, homolog_seqs_dict, homolog_similarities_dict = {}, {}, {}
                homolog_identities_dict, homolog_binding_scores_dict, homolog_positive_scores_dict = {}, {}, {}
                homolog_suboptimal_scores_dict, homolog_forbidden_scores_dict = {}, {}
                homolog_total_scores_dict, homolog_model_calls_dict, homolog_specificity_scores_dict = {}, {}, {}
                for i in np.arange(len(current_df)):
                    ensembl_peptide_id = current_df.at[i, "ensembl_peptide_id"]
                    if homolog_ids_dict.get(ensembl_peptide_id) is None:
                        refseq_dict[ensembl_peptide_id] = current_df.at[i, "refseq_peptide"]
                        refseq_predicted_dict[ensembl_peptide_id] = current_df.at[i, "refseq_peptide_predicted"]
                        homolog_ids_dict[ensembl_peptide_id] = current_df.at[i, homolog_col_prefix + "_id_best"]
                        homolog_seqs_dict[ensembl_peptide_id] = current_df.at[i, homolog_col_prefix + "_best"]
                        homolog_similarities_dict[ensembl_peptide_id] = current_df.at[i, homolog_col_prefix + "_similarity_best"]
                        homolog_identities_dict[ensembl_peptide_id] = current_df.at[i, homolog_col_prefix + "_identity_best"]
                        homolog_binding_scores_dict[ensembl_peptide_id] = current_df.at[i, homolog_col_prefix + "_best_binding_model_score"]
                        homolog_positive_scores_dict[ensembl_peptide_id] = current_df.at[i, homolog_col_prefix + "_best_positive_model_score"]
                        homolog_suboptimal_scores_dict[ensembl_peptide_id] = current_df.at[i, homolog_col_prefix + "_best_suboptimal_model_score"]
                        homolog_forbidden_scores_dict[ensembl_peptide_id] = current_df.at[i, homolog_col_prefix + "_best_forbidden_model_score"]
                        homolog_total_scores_dict[ensembl_peptide_id] = current_df.at[i, homolog_col_prefix + "_best_total_model_score"]
                        homolog_model_calls_dict[ensembl_peptide_id] = current_df.at[i, homolog_col_prefix + "_best_model_call"]
                        homolog_specificity_scores_dict[ensembl_peptide_id] = current_df.at[i, homolog_col_prefix + "_best_specificity_score"]

                ensembl_peptide_ids = combined_df["ensembl_peptide_id"].to_list()

                combined_df[homolog_col_prefix + "_id_best"] = [homolog_ids_dict.get(unique_peptide_id) for unique_peptide_id in ensembl_peptide_ids]
                combined_df[homolog_col_prefix + "_best"] = [homolog_seqs_dict.get(unique_peptide_id) for unique_peptide_id in ensembl_peptide_ids]
                combined_df[homolog_col_prefix + "_similarity_best"] = [homolog_similarities_dict.get(unique_peptide_id) for unique_peptide_id in ensembl_peptide_ids]
                combined_df[homolog_col_prefix + "_identity_best"] = [homolog_identities_dict.get(unique_peptide_id) for unique_peptide_id in ensembl_peptide_ids]

                combined_df[homolog_col_prefix + "_best_binding_model_score"] = [homolog_binding_scores_dict.get(unique_peptide_id) for unique_peptide_id in ensembl_peptide_ids]
                combined_df[homolog_col_prefix + "_best_positive_model_score"] = [homolog_positive_scores_dict.get(unique_peptide_id) for unique_peptide_id in ensembl_peptide_ids]
                combined_df[homolog_col_prefix + "_best_suboptimal_model_score"] = [homolog_suboptimal_scores_dict.get(unique_peptide_id) for unique_peptide_id in ensembl_peptide_ids]
                combined_df[homolog_col_prefix + "_best_forbidden_model_score"] = [homolog_forbidden_scores_dict.get(unique_peptide_id) for unique_peptide_id in ensembl_peptide_ids]
                combined_df[homolog_col_prefix + "_best_total_model_score"] = [homolog_total_scores_dict.get(unique_peptide_id) for unique_peptide_id in ensembl_peptide_ids]
                combined_df[homolog_col_prefix + "_best_model_call"] = [homolog_model_calls_dict.get(unique_peptide_id) for unique_peptide_id in ensembl_peptide_ids]
                combined_df[homolog_col_prefix + "_best_specificity_score"] = [homolog_specificity_scores_dict.get(unique_peptide_id) for unique_peptide_id in ensembl_peptide_ids]

                combined_df[homolog_col_prefix + "_homologene_refseq_peptide"] = [refseq_dict.get(unique_peptide_id) for unique_peptide_id in ensembl_peptide_ids]
                combined_df[homolog_col_prefix + "_homologene_refseq_peptide_predicted"] = [refseq_predicted_dict.get(unique_peptide_id) for unique_peptide_id in ensembl_peptide_ids]

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