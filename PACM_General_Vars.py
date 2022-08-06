#General reference objects

aa_charac_dict = {
	"Acidic": ["D", "E"],
	"Basic": ["K", "R", "H"],
	"ST": ["S", "T"],
	"Aromatic": ["F", "Y"],
	"Aliphatic": ["A", "V", "I", "L"],
	"Other_Hydrophobic": ["W", "M"],
	"Polar_Uncharged": ["N", "Q", "C"],
	"Special_Cases": ["P", "G"]
}

list_aa_no_phos = ["D", "E", "R", "H", "K", "S", "T", "N", "Q", "C", "G", "P", "A", "V", "I", "L", "M", "F", "Y", "W"]

list_aa = ["D", "E", "R", "H", "K", "S", "T", "N", "Q", "C", "G", "P", "A", "V", "I", "L", "M", "F", "Y", "W", "B", "J", "O"]
#We had to assign single letter codes to phosphorylated residues, so we used B = pSer, J = pThr, and O = pTyr..