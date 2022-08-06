# PACM
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

The workflow is divided into six (6) parts as follows. 

Step1_Array_Analyzer.py - Accepts SPOT peptide array densitometric input values and assumes 2 replicates. 
[To be continued.]
