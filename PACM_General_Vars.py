#General reference objects

#Define chemical characteristics of amino acids. This could conceivably be adjusted, but we have not tested it. 
#We group amino acids by characteristic because processing them individually in Step2_Pairwise_SLiM_Matrices_Generator.py would require a much larger input dataset (on the order of thousands of peptides).
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

#General list of proteinogenic amino acids
list_aa_no_phos = ["D", "E", "R", "H", "K", "S", "T", "N", "Q", "C", "G", "P", "A", "V", "I", "L", "M", "F", "Y", "W"]

#List of amino acids that includes pSer (assigned as "B"), pThr (assigned as "J"), and pTyr (assigned as "O"). The single letter codes for phospho-residues are arbitrary. 
list_aa = ["D", "E", "R", "H", "K", "S", "T", "N", "Q", "C", "G", "P", "A", "V", "I", "L", "M", "F", "Y", "W", "B", "J", "O"]

#Dict of amino acids with their row indices for lookup in position-weighted matrices. Arbitrary, but must be universal between scripts. 
index_aa_dict = {
	"D": 0,
	"E": 1,
	"R": 2,
	"H": 3,
	"K": 4,
	"S": 5,
	"T": 6,
	"N": 7,
	"Q": 8,
	"C": 9,
	"G": 10,
	"P": 11,
	"A": 12,
	"V": 13,
	"I": 14,
	"L": 15,
	"M": 16,
	"F": 17,
	"Y": 18,
	"W": 19
}