import numpy as np
import pandas as pd
import pickle
import os
import xmltodict

def load_uniprot(path = None):
    '''
    Simple function to load Uniprot database, available over FTP.
    Warning: loads whole file in memory, so this will only work for subsets, not the whole database.

    Args:
        path (str):  path to either the XML file or a pickled version of it as a dictionary of dictionaries (pickled)

    Returns:
        data (dict): dictionary of results
    '''

    # Look for path in repo directory if not given
    if path is None:
        parent_dir = os.getcwd().rsplit("/", 1)[0]
        paths = os.listdir(parent_dir)
        for existing_path in paths:
            if existing_path.split("_",1)[0] == "uniprot":
                extension = existing_path.rsplit(".",1)[1]
                if extension == "pkl":
                    path = existing_path
                    break
                elif extension == "xml":
                    path = existing_path

    # If path not found, prompt the user
    if path is None:
        valid_path = False
        while not valid_path:
            path = input("Please enter the path to the Uniprot pkl (pickled) or XML data:  ")
            if os.path.exists(path):
                valid_path = True

    # Parse the path into a dictionary of dictionaries
    with open(path, "rb") as file:
        if path.rsplit(".",1)[1] == "xml":
            data = xmltodict.parse(file)
        elif path.rsplit(".",1)[1] == "pkl":
            data = pickle.load(file)
        else:
            raise Exception(f"Incorrect filetype; must be xml or pkl (pickled): {path}")

    return data

def get_topological_domains(data):
    '''
    Function that parses the dictionary-of-dictionaries into topological domain and sequence dictionaries

    Args:
        data (dict): dictionary of results

    Returns:
        topological_domains (dict): dictionary of accession --> topological features list
        sequences (dict):           dictionary of accession --> sequence
    '''

    topological_domains = {}
    sequences = {}

    entries = data["uniprot"]["entry"]
    for entry in entries:
        # Parse topological features into a list of tuples
        features = entry["feature"]
        topological_features = []
        for feature in features:
            feature_type = feature["@type"]
            if "topo" in feature_type or "transmem" in feature_type or "intramem" in feature_type:
                begin = feature["location"]["begin"]["@position"]
                end = feature["location"]["end"]["@position"]
                topological_feature_tuple = (feature_type, begin, end)
                topological_features.append(topological_feature_tuple)

        # Parse sequence and assign data to accessions
        sequence = entry["sequence"]["#text"]
        accessions = entry["accession"]
        for accession in accessions:
            sequences[accession] = sequence
            topological_domains[accession] = topological_features

    return topological_domains, sequences