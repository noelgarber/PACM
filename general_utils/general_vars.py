'''
Amino acid chemical characteristics dictionary
    --> Proline is treated separately due to being able to induce kinking
    --> Glycine is treated separately due to being able to induce high conformational flexibility
    --> Histidine is treated as a polar uncharged amino acid because it is rarely protonated at physiological pH
'''
aa_charac_dict = {
    "Acidic": ["D", "E"],
    "Basic": ["K", "R"],
    "ST": ["S", "T", "B", "J"],
    "Aromatic": ["F", "Y", "W", "O"],
    "Aliphatic": ["A", "V", "I", "L", "M"],
    "Polar": ["N", "Q", "H", "C"],
    "Proline": ["P"],
    "Glycine": ["G"]
}

# Declare the sorted list of amino acids
amino_acids = ("D", "E", "R", "H", "K", "S", "T", "N", "Q", "C", "G", "P", "A", "V", "I", "L", "M", "F", "Y", "W")
amino_acids_phos = ("D", "E", "R", "H", "K", "S", "T", "N", "Q", "C", "G", "P", "A", "V", "I", "L", "M", "F", "Y", "W", "B", "J", "O") # B=pSer, J=pThr, Y=pTyr

