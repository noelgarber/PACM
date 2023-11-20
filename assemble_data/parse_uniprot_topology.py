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

def get_topological_domains(path = None):
    '''
    Function that parses the dictionary-of-dictionaries into topological domain and sequence dictionaries

    Args:
        path (str):  path to either the XML file or a pickled version of it as a dictionary of dictionaries (pickled)

    Returns:
        topological_domains (dict): dictionary of accession --> topological features list
        sequences (dict):           dictionary of accession --> sequence
    '''

    data = load_uniprot(path)

    topological_domains = {}
    sequences = {}

    entries = data["uniprot"]["entry"]
    for i, entry in enumerate(entries):
        # Parse topological features into a list of tuples
        features = entry["feature"]
        topological_features = []
        for feature in features:
            if isinstance(feature, dict):
                feature_type = feature.get("@type")
                if feature_type is not None:
                    if "topo" in feature_type or "transmem" in feature_type or "intramem" in feature_type:
                        description = feature.get("@description")
                        begin = feature["location"].get("begin")
                        end = feature["location"].get("end")
                        if begin is not None and end is not None:
                            begin_position = begin.get("@position")
                            end_position = end.get("@position")
                            if begin_position is not None and end_position is not None:
                                print(f"Entry {i}: found topological domain at ({begin_position},{end_position})")
                                begin_position = int(begin_position)
                                end_position = int(end_position)
                                topological_feature_tuple = (feature_type, description, begin_position, end_position)
                                topological_features.append(topological_feature_tuple)
                            else:
                                print(f"Entry {i}: couldn't find @position tag for begin/end in topology: {description}")
                        else:
                            print(f"Entry {i}: couldn't find location tag for begin/end in topology: {description}")
                else:
                    print(f"Entry {i}: could not find feature type for a feature in features")
            else:
                print(f"Entry {i}: a feature ({feature}) was not a dictionary; it was skipped")

        # Parse sequence and assign data to accessions only if topological domains are listed (saves memory)
        if len(topological_features) > 0:
            sequence = entry["sequence"]["#text"]
            accessions = entry["accession"]
            for accession in accessions:
                sequences[accession] = sequence
                topological_domains[accession] = topological_features

    del data

    return topological_domains, sequences