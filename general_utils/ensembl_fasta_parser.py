# User-initiated helper script for parsing Ensembl FASTA Files to dataframes and saving

import pandas as pd
from Bio import SeqIO

def main(fasta_path):
    data = []
    records = [record for record in SeqIO.parse(fasta_path, "fasta")]
    for record in records:
        ensembl_protein_id = record.id
        protein_seq = record.seq
        description_elements = record.description.split(" ")
        gene_name = ""
        ensembl_transcript_id = ""
        ensembl_gene_id = ""
        for element in description_elements:
                if "gene_symbol" in element:
                        gene_name = element.split(":",1)
                elif "gene:" in element:
                        ensembl_gene_id = element.split(":",1)
                elif "transcript:" in element:
                        ensembl_transcript_id = element.split(":",1)
        data.append([ensembl_protein_id, ensembl_transcript_id, ensembl_gene_id, gene_name, protein_seq])

    cols = ["Ensembl_Protein_ID","Ensembl_Transcript_ID","Ensembl_Gene_ID","Gene_Name","Sequence"]
    df = pd.DataFrame(data, columns=cols)

    csv_path = fasta_path.split(".fa")[0] + ".csv"
    df.to_csv(csv_path)

    return df

if __name__ == "__main__":
    fasta_path = input("Enter the Ensembl FASTA file path: ")
    main(fasta_path)