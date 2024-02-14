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

        # Assign topology info and source uniprot accessions to dataframe
        topology_types = [topology_type_dict.get(unique_peptide_id) for unique_peptide_id in unique_ensembl_peptides]
        topology_descs = [topology_desc_dict.get(unique_peptide_id) for unique_peptide_id in unique_ensembl_peptides]
        source_uniprot_ids = [uniprot_dict.get(unique_peptide_id) for unique_peptide_id in unique_ensembl_peptides]
        source_trembl_ids = [trembl_dict.get(unique_peptide_id) for unique_peptide_id in unique_ensembl_peptides]

        unique_df[topology_type_col] = topology_types
        unique_df[topology_desc_col] = topology_descs
        unique_df[motif_col + "_topology_source_uniprot_id"] = source_uniprot_ids
        unique_df[motif_col + "_topology_source_trembl_id"] = source_trembl_ids

    return unique_df, motif_cols, motif_col_groups

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

    combined_df = fuse_dfs(csv_paths)

    parent_folder = csv_paths[0].rsplit("/",1)[0]
    save_path = os.path.join(parent_folder, "combined_datasets.csv")
    combined_df.to_csv(save_path)