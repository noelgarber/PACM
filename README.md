# PACM
By Noel Garber, 
PhD Candidate, University of Toronto
noel.garber@sickkids.ca

Peptide Array-based Conditional position-weighted Matrices

This is a pipeline for analyzing SPOT peptide array in vitro binding data for defining short linear motifs (SLiMs). 
As no other tools (to my knowledge) exist for analyzing this kind of data, this one is now public and free to use (with attribution)! 

SLiMs are often defined by a position-weighted matrix that assigns values to different positions in the peptide sequence. 
Historically, this approach is unable to account for intra-peptide residue-residue interactions. For example, a particular residue existing 
at one position may affect what its neighbours can be, but a typical matrix cannot study this. 

Here, I have used a dynamical approach that generates a dictionary of conditional position-weighted matrices. 
It looks up values in matrices corresponding to neighbouring residue type. 
For example, if a SLiM has the sequence Leu1-Asp2-Ala3-Met4-Lys5, and we are looking up the matrix value for Ala in Position 3, the algorithm 
will look up Ala3 in a SLiM matrix for when Position 2 is acidic (Asp2) and in another SLiM matrix for when Position 4 is hydrophobic (Met), 
take the mean of those values, and assign that as the matrix value for Ala3. 

# How To Use

Ensure that Python 3 is installed (version 3.6 or higher). 

Python scripts are run from Command Prompt (Windows) or Terminal (Mac and Linux) by typing "python3" or "python" and then the script name (.py file), followed by hitting enter. 

1. [Required] Run Step1_Array_Analyzer.py and ensure your data is in the same folder, then follow the prompts to input your data file and apply standardization etc. 
2. [Required] Run Step2_Pairwise_SLiM_Matrices_Generator.py to generate conditional position-weighted matrices for SLiM identification. 
3. [Optional] Run Step3_Bait_Specificity_Matrix_Generator.py to generate a matrix predicting which bait will preferentially bind a particular motif. 
4. [Optional] Run Step4_Pairwise_FFAT_Predictor.py to apply the SLiM prediction algorithm made in Step 2 to a list of protein sequences. 
5. [Optional] Run Step5_Bait_Specificity_Predictor.py to apply the bait specificity prediction algorithm made in Step 3 to the list of protein sequences from Step 4. 
6. [Optional] Run Step6_Topo_SLiM_Predictor.py to use UniProt for predicting the topological orientation/localization of motifs found in Steps 4 and 5. 