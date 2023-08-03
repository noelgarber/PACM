# PACM
By Noel Garber, PhD Candidate, University of Toronto (noel.garber@sickkids.ca)

Peptide Array-based Conditional position-weighted Matrices

This is a pipeline for analyzing in vitro peptide array binding data (e.g. using the SPOT method) for dynamically defining, analyzing, and predicting short linear motifs (SLiMs). 

SLiMs are often defined by a position-weighted matrix that assigns values to different positions in the peptide sequence. 
Historically, this approach is unable to account for intra-peptide residue-residue interactions. For example, a particular residue existing 
at one position may affect what its neighbours can be, but a typical matrix cannot represent this. 

Here, I have used a dynamical approach that generates a set of conditional position-weighted matrices. 
It looks up values in matrices corresponding to neighbouring residue type. 
For example, if a SLiM has the sequence Leu1-Asp2-Ala3-Met4-Lys5, and we are looking up the matrix value for Ala in Position 3, the algorithm 
will look up Ala3 in a SLiM matrix for when Position 2 is acidic (Asp2) and in another SLiM matrix for when Position 4 is hydrophobic (Met), 
take the mean of those values, and assign that as the matrix value for Ala3. This is then repeated for all the residues.

This tool is written in Python 3. To use, ensure that you have the latest version of Python 3 installed, along with the requisite modules. You will be prompted to install dependencies if required. 

You may run these scripts by typing ```python``` or ```python3```, followed by the script file path, into the Command Prompt (Windows) or Terminal in Mac or Linux. 

# How To Use

This repository is divided into two parts. ```matrix_generator.py```, which uses scripts in the Matrix_Generator folder, is the first part that quantifies spot peptide arrays and constructs context-aware position-weighted matrices based on them. Once the matrices have been generated, ```matrix_predictor.py``` (not yet available, stay tuned for an impending update) can be used to apply them to novel protein sequences to find the best motifs.

# Building Dynamic Position-Weighted Matrices

To quantify the data and build matrices, first fill in all the arguments in ```/Matrix_Generator/config.py``` (or a copy as ```config_local.py``` in the same directory if desired). Then, run ```matrix_generator.py```, follow any given prompts, and allow the algorithm to run. Please note that if position weight optimization is used, the script can take several hours to run, depending on the size of your dataset and the capabilities of your machine.

# Predicting SLiMs in Protein Sequences

Once the dynamic position-weigihted matrices have been built, you may apply them as a predictive algorithm onto new protein sequences to predict whether they contain your identified SLiM of interest. This may be done for an unlimited number of protein sequences listed in the prescribed format. 

This section is being updated to work with the newer code from matrix_generator.py and will become functional in an update soon; please stay tuned for this! 