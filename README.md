# PACM
By Noel Garber, PhD Candidate, University of Toronto (noel.garber@sickkids.ca)

Peptide Array-based Conditional position-weighted Matrices

This is a pipeline for analyzing in vitro peptide array binding data (e.g. using the SPOT method) for dynamically defining, analyzing, and predicting short linear motifs (SLiMs). 

SLiMs are often defined by a position-weighted matrix that assigns values to different positions in the peptide sequence. 
Historically, this approach is unable to account for intra-peptide residue-residue interactions. For example, a particular residue existing 
at one position may affect what its neighbours can be, but a typical matrix cannot represent this. 

Here, I have used a dynamical approach that generates a set of conditional position-weighted matrices for positive elements (sequence elements that improve binding strength), suboptimal elements (sequence elements that reduce the likelihood of binding), and forbidden elements (sequence elements that kill binding). 

It looks up residue score values in matrices corresponding to neighbouring residue type on either side of a given residue being scored. Residue scores are then weighted and summed to produce a final motif score. 

This tool is written in Python 3.11.6, but might still work for earlier versions as early as 3.7.0. To use, ensure that you have the latest version of Python 3 installed, along with the requisite modules. Running the programs will notify you if modules are missing. 

You may run these scripts by typing ```python``` or ```python3```, followed by the script file path, into the Command Prompt (Windows) or Terminal in Mac or Linux. Note that paths are handled based on Linux/Mac conventions (forward slash, /), so these may fail on Windows, and this has not been tested. 

# How To Use

This repository is divided into several parts. 
1. ```matrix_generator.py```, which uses scripts in the Matrix_Generator folder, is the first part that quantifies spot peptide arrays and constructs context-aware position-weighted matrices (the model)
2. ```matrix_predictor.py``` can be used to apply the pre-generated matrices to novel protein sequences to find the best motifs
3. ```assemble_data/generate_dataset.py``` can be used to produce a proteome dataset if you want to run the model against whole proteomes, including homology analysis

# 1. Training a model

To quantify the data and build matrices, first fill in all the arguments in ```/Matrix_Generator/config.py``` (or a copy as ```config_local.py``` in the same directory if desired). Then, run ```matrix_generator.py```, follow any given prompts, and allow the algorithm to run. This program may take minutes to hours to run, depending on the power of your machine. 

Briefly, model training involves first quantifying your SPOT peptide array images, then generating the conditional matrices for motif prediction, and then generating the specificity matrix for specificity prediction. It is possible to train a model from another kind of data (e.g. ELISA, other microarrays, biolayer interferometry (BLI), surface plasmon resonance (SPR), etc.), but this feature has not yet been added (although it would be relatively simple to do). 

# 2. Applying a trained model

Once the dynamic position-weigihted matrices have been built, you may apply them as a predictive algorithm onto new protein sequences to predict whether they contain your identified SLiM of interest. This may be done for an unlimited number of protein sequences listed in the prescribed format. To run it, please first fill in all the arguments in ```/Motif_Predictor/predictor_config.py``` (or a copy as ```predictor_config_local.py``` in the same directory). Note that running this on entire genomes can take hours if running on a personal computer. 

# 3. Generate a proteome dataset to apply the model to (optional)

If you have a trained model and you want to apply it to an entire genome's protein products, or even multiple genomes, use ```assemble_data/generate_dataset.py```. You will be prompted for two files: 1) an Ensembl FASTA file (available from Ensembl's FTP site) and 2) the ```homologene.data``` file from NCBI FTP containing HomoloGene's homology database. We use HomoloGene instead of Ensembl for this because it can handle very distant species (e.g. humans vs. plants). 