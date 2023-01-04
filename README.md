# PACM
By Noel Garber, PhD Candidate, University of Toronto
noel.garber@sickkids.ca

Peptide Array-based Conditional position-weighted Matrices

This is a pipeline for analyzing peptide array in vitro binding data (e.g. using the SPOT method) for dynamically defining, analyzing, and predicting short linear motifs (SLiMs). 

SLiMs are often defined by a position-weighted matrix that assigns values to different positions in the peptide sequence. 
Historically, this approach is unable to account for intra-peptide residue-residue interactions. For example, a particular residue existing 
at one position may affect what its neighbours can be, but a typical matrix cannot study this. 

Here, I have used a dynamical approach that generates a dictionary of conditional position-weighted matrices. 
It looks up values in matrices corresponding to neighbouring residue type. 
For example, if a SLiM has the sequence Leu1-Asp2-Ala3-Met4-Lys5, and we are looking up the matrix value for Ala in Position 3, the algorithm 
will look up Ala3 in a SLiM matrix for when Position 2 is acidic (Asp2) and in another SLiM matrix for when Position 4 is hydrophobic (Met), 
take the mean of those values, and assign that as the matrix value for Ala3. This is then repeated for all the residues.

This tool is written in Python 3. To use, ensure that you have the latest version of Python 3 installed, along with the requisite modules. You will be prompted to install dependencies if required. 

You may run these scripts by typing ```python``` or ```python3```, followed by the script file path, into the Command Prompt (Windows) or Terminal in Mac or Linux. 

# How To Use

Optionally, you may preprocess and quantify your SPOT peptide array image data with the included tool, ```image_prep.py```. This tool is built for quantifying grids of spots in TIFF images. Alternatively, you can use your own quantification method and proceed directly to analysis. 

# Building Dynamic Position-Weighted Matrices

Perform the following steps in order: 

1. Run ```/Matrix_Generator/process_arrays.py```, which pre-processes and standardizes your quantified array data that conforms to the standard format (see ```/Matrix_Generator/array_data_example.csv```). This will also perform bait-bait comparisons and calculate log2fc to compare peptide binding when more than one bait protein was tested. Additionally, it can stitch together multiple datasets from multiple arrays and standardize appropriately to controls. 
2. Run ```/Matrix_Generator/make_pairwise_matrices.py```, which generates conditional position-weighted matrices that take neighbouring residues into account when defining a short linear motif (SLiM). 
3. Optionally, run ```/Matrix_Generator/make_specificity_matrices.py```, which generates a specificity position-weighted matrix for predicting which bait will be preferentially bound by a particular SLiM based on the input data peptide sequences. Results should be used in conjunction with those from ```make_pairiwise_matrices.py```, as specificity scores are only interpretable for peptide sequences that are predicted to qualify as the SLiM of interest. 

# Predicting SLiMs in Protein Sequences

Once the dynamic position-weigihted matrices have been built, you may apply them as a predictive algorithm onto new protein sequences to predict whether they contain your identified SLiM of interest. This may be done for an unlimited number of protein sequences listed in the prescribed format. 

To do so, perform these steps in order: 

1. Run ```/Motif_Predictor/slim_predictor.py```, which will prompt you to input the protein sequences to analyze. 
2. Optionally, if a specificity matrix was generated previously, run ```/Motif_Predictor/slim_specificity_predictor.py```, which will prompt you to input data from the previous step. 
3. Optionally, you may run ```/Motif_Predictor/slim_topology_finder.py``` to find the predicted topology of identified SLiMs in the output data from previous steps. Topology information can help you to identify which SLiMs will face the compartment where your SLiM receptor of interest is localized. 
4. Optionally, to perform evolutionary conservation analysis, you may run ```/Motif_Predictor/slim_conservation.py``` and enter the homolog species you would like to use for conservation analysis.