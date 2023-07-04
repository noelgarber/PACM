# Declare the amino acid chemical characteristics dictionary
aa_charac_dict = {
    "Acidic": ["D", "E"],
    "Basic": ["K", "R", "H"],
    "ST": ["S", "T", "B", "J"],
    "Aromatic": ["F", "Y", "O"],
    "Aliphatic": ["A", "V", "I", "L"],
    "Other_Hydrophobic": ["W", "M"],
    "Polar_Uncharged": ["N", "Q", "C"],
    "Special_Cases": ["P", "G"]
}

# Declare the sorted list of amino acids
amino_acids = ("D", "E", "R", "H", "K", "S", "T", "N", "Q", "C", "G", "P", "A", "V", "I", "L", "M", "F", "Y", "W")
amino_acids_phos = ("D", "E", "R", "H", "K", "S", "T", "N", "Q", "C", "G", "P", "A", "V", "I", "L", "M", "F", "Y", "W", "B", "J", "O") # B=pSer, J=pThr, Y=pTyr

