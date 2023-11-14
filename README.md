# PACM
By Noel Garber, PhD Candidate, University of Toronto (noel.garber@sickkids.ca)

Peptide Array-based Conditional position-weighted Matrices

This is a pipeline for analyzing in vitro peptide array binding data (e.g. using the SPOT method) for dynamically defining, analyzing, and predicting short linear motifs (SLiMs). 

SLiMs are often defined by a position-weighted matrix that assigns values to different positions in the peptide sequence. 
Historically, this approach is unable to account for intra-peptide residue-residue interactions. For example, a particular residue existing 
at one position may affect what its neighbours can be, but a typical matrix cannot represent this. 

Here, I have used a dynamical approach that generates a set of conditional position-weighted matrices for positive elements (improves binding), suboptimal elements (inhibits binding), and forbidden elements (kills binding). 
It looks up residue score values in matrices corresponding to neighbouring residue type on either side of a given residue being scored. Residue scores are then weighted and summed to produce a final motif score. 

This tool is written in Python 3. To use, ensure that you have the latest version of Python 3 installed, along with the requisite modules. 

You may run these scripts by typing ```python``` or ```python3```, followed by the script file path, into the Command Prompt (Windows) or Terminal in Mac or Linux. 

# How To Use

This repository is divided into several parts. 
1. ```matrix_generator.py```, which uses scripts in the Matrix_Generator folder, is the first part that quantifies spot peptide arrays and constructs context-aware position-weighted matrices (the model)
2. ```matrix_predictor.py``` can be used to apply the pre-generated matrices to novel protein sequences to find the best motifs
3. ```assemble_data/generate_dataset.py``` can be used to produce a proteome dataset if you want to run the model against whole proteomes, including homology analysis

# Building Dynamic Position-Weighted Matrices

To quantify the data and build matrices, first fill in all the arguments in ```/Matrix_Generator/config.py``` (or a copy as ```config_local.py``` in the same directory if desired). Then, run ```matrix_generator.py```, follow any given prompts, and allow the algorithm to run. Please note that if position weight optimization is used, the script can take several hours to run, depending on the size of your dataset and the capabilities of your machine.

# Predicting SLiMs in Protein Sequences

Once the dynamic position-weigihted matrices have been built, you may apply them as a predictive algorithm onto new protein sequences to predict whether they contain your identified SLiM of interest. This may be done for an unlimited number of protein sequences listed in the prescribed format. To run it, please first fill in all the arguments in ```/Motif_Predictor/predictor_config.py``` (or a copy as ```predictor_config_local.py``` in the same directory). 
