# This workflow processes SPOT images and uses them to build a convolutional neural network to predict FFATs in novel sequences

# Import standard packages
import numpy as np
import pandas as pd
import os

# Import functions for SPOT image analysis and data standardization
from Matrix_Generator.standardize_and_concatenate import main_workflow as standardized_concatenate

# Import functions for training the convolutional neural network
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, MaxPooling1D, Flatten, Dense, Dropout

# Define dictionaries for amino acid side chain characteristics
charge_dict = {'A': 0.0, 'C': 0.0, 'D': -1.0, 'E': -1.0, 'F': 0.0, 'G': 0.0, 'H': 0.0, 'I': 0.0,
               'K': 1.0, 'L': 0.0, 'M': 0.0, 'N': 0.0, 'P': 0.0, 'Q': 0.0, 'R': 1.0, 'S': 0.0,
               'T': 0.0, 'V': 0.0, 'W': 0.0, 'Y': 0.0, 'B': -2.0, 'J': -2.0, 'O': -2.0}

mol_weight_dict = {'A': 71.08, 'C': 103.14, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                   'I': 113.16, 'K': 128.17, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                   'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18, 'B': 181.04,
                   'J': 195.07, 'O': 257.14}

hydrophobicity_dict = {'A': 0.62, 'C': 0.29, 'D': -0.9, 'E': -0.74, 'F': 1.19, 'G': 0.48, 'H': -0.4,
                       'I': 1.38, 'K': -1.5, 'L': 1.06, 'M': 0.64, 'N': -0.78, 'P': 0.12, 'Q': -0.85,
                       'R': -2.53, 'S': -0.18, 'T': -0.05, 'V': 1.08, 'W': 0.81, 'Y': 0.26, 'B': -0.8,
                       'J': -0.7, 'O': -1.3}

aromaticity_dict = {'A': 0.0, 'C': 0.0, 'D': 0.0, 'E': 0.0, 'F': 1.0, 'G': 0.0, 'H': 0.25, 'I': 0.0,
                    'K': 0.0, 'L': 0.0, 'M': 0.0, 'N': 0.0, 'P': 0.0, 'Q': 0.0, 'R': 0.0, 'S': 0.0,
                    'T': 0.0, 'V': 0.0, 'W': 0.5, 'Y': 0.75, 'B': 0.0, 'J': 0.0, 'O': 0.0}

def encode_data(data_df, seq_col = "BJO_Sequence", aa_alphabet = "ACDEFGHIKLMNPQRSTVWYBJO", test_size = 0.2,
                only_use_characteristics = False):
    '''
    Function to encode data for convolutional neural network training and testing

    Args:
        data_df (pd.DataFrame):          a dataframe that must contain peptide sequences and Yes/No categorical calls
        seq_col (str):                   the column name for where sequences are stored
        call_col (str):                  the column name for where calls are stored (Yes/No categories)
        aa_alphabet (str):               a string with the alphabet of amino acids to use, which may also include extra letters
        only_use_characteristics (bool): whether to only use amino acid characteristics and exclude the amino acid letters themselves

    Returns:
        X_train, X_test, y_train, y_test: train/test split data
    '''

    # Check length of sequences
    if data_df[seq_col].str.len().min() != data_df[seq_col].str.len().max():
        raise ValueError(f"encode_data found variable sequence lengths in data_df[{seq_col}] ranging from {data_df[seq_col].str.len().min()} to {data_df[seq_col].str.len().max()}, but equal lengths were expected.")

    # Convert the peptide sequences to arrays of letters, charges, mol_weights, and hydrophobicities
    peptide_sequences = data_df[seq_col].values
    sequence_length = len(peptide_sequences[0])

    # Create arrays to store the encoded data
    charges = np.zeros((len(peptide_sequences), sequence_length), dtype=int)
    mol_weights = np.zeros((len(peptide_sequences), sequence_length), dtype=int)
    hydrophobicities = np.zeros((len(peptide_sequences), sequence_length), dtype=int)
    aromaticities = np.zeros((len(peptide_sequences), sequence_length), dtype=int)
    if not only_use_characteristics:
        encoded_sequences = np.zeros((len(peptide_sequences), sequence_length), dtype=int)

    # Fill the arrays with the encoded data
    for i, seq in enumerate(peptide_sequences):
        charges[i] = [charge_dict[aa] for aa in seq]
        mol_weights[i] = [mol_weight_dict[aa] for aa in seq]
        hydrophobicities[i] = [hydrophobicity_dict[aa]*100 for aa in seq]
        aromaticities[i] = [aromaticity_dict[aa]*100 for aa in seq]
        if not only_use_characteristics:
            encoded_sequences[i] = [aa_alphabet.index(aa) for aa in seq]

    # One-hot encode the letter sequences
    if not only_use_characteristics:
        one_hot_sequences = tf.keras.utils.to_categorical(encoded_sequences, num_classes=len(aa_alphabet))

    # Reshape the arrays to match the dimensions of one_hot_sequences
    charges = np.expand_dims(charges, axis=2)
    mol_weights = np.expand_dims(mol_weights, axis=2)
    hydrophobicities = np.expand_dims(hydrophobicities, axis=2)
    aromaticities = np.expand_dims(aromaticities, axis=2)

    # Combine the one-hot encoded sequences with the chemical information
    if only_use_characteristics:
        input_data = np.concatenate((charges, mol_weights, hydrophobicities, aromaticities), axis=2)
    else:
        input_data = np.concatenate((one_hot_sequences, charges, mol_weights, hydrophobicities, aromaticities), axis=2)

    # Get the call categories ("Yes" is 1)
    mapping = {"Yes": 1, "": 0}
    mapped_calls = data_df['One_Passes'].map(mapping).fillna(0)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(input_data, mapped_calls, test_size=test_size, random_state=42)
    X_train, X_test, y_train, y_test = X_train.astype(int), X_test.astype(int), y_train.astype(int), y_test.astype(int)
    train_test_data = (X_train, X_test, y_train, y_test)

    print(X_train)
    print(y_train)

    if only_use_characteristics:
        num_features = 4
    else:
        num_features = len(aa_alphabet) + 4

    feature_info = (sequence_length, num_features)

    return train_test_data, feature_info

def train_cnn(data_df, seq_col = "BJO_Sequence", aa_alphabet = "ACDEFGHIKLMNPQRSTVWYBJO", test_size = 0.2,
              only_use_characteristics = False, save_path = None):
    '''
    Function to train the convolutional neural network

    Args:
        data_df (pd.DataFrame):          a dataframe that must contain peptide sequences and Yes/No categorical calls
        seq_col (str):                   the column name for where sequences are stored
        call_col (str):                  the column name for where calls are stored (Yes/No categories)
        aa_alphabet (str):               a string with the alphabet of amino acids to use, which may also include extra letters
        only_use_characteristics (bool): whether to only use amino acid characteristics and exclude the amino acid letters themselves

    Returns:
        model (tensorflow.keras.models.Sequential): the trained neural network model
    '''

    # Encode and split the data
    train_test_data, feature_info = encode_data(data_df = data_df, seq_col = seq_col, aa_alphabet = aa_alphabet,
                                                test_size = test_size, only_use_characteristics = only_use_characteristics)
    X_train, X_test, y_train, y_test = train_test_data
    print(f"X_train shape: {X_train.shape} | X_train dtype: {X_train.dtype}")
    sequence_length, num_features = feature_info

    # Define the CNN model
    model = Sequential()
    model.add(Conv1D(32, 5, activation='relu', input_shape=(sequence_length, num_features)))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    # Compile and train the model
    print("Training the convolutional neural network...")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    print("Evaluating the model based on testing data: ")
    model.evaluate(X_test, y_test)

    if save_path is not None:
        model.save(os.path.join(save_path, "model.h5"))

    return model

def get_data(output_folder = None, add_peptide_seqs = True, 
             peptide_seq_cols = ["Phos_Sequence", "No_Phos_Sequence", "BJO_Sequence"], verbose = False):

    # Get output folder if not provided
    if output_folder is None:
        user_output_folder = input(
            "Enter the folder for saving data, or leave blank to use the working directory:  ")
        if user_output_folder != "":
            output_folder = user_output_folder
        else:
            output_folder = os.getcwd()

    # Get the standardized concatenated dataframe containing all of the quantified peptide spot data
    print("Processing and standardizing the SPOT image data...") if verbose else None
    data_df, percentiles_dict = standardized_concatenate(predefined_batch = True, add_peptide_seqs = add_peptide_seqs,
                                                         peptide_seq_cols = peptide_seq_cols)
    reindexed_data_df = data_df.reset_index(drop = False)
    reindexed_data_df.to_csv(os.path.join(output_folder, "standardized_and_concatenated_data.csv"))
    
    return reindexed_data_df
    
def main(output_folder = None, add_peptide_seqs = True, seq_col = "BJO_Sequence", aa_alphabet = "ACDEFGHIKLMNPQRSTVWYBJO",
         peptide_seq_cols = ["Phos_Sequence", "No_Phos_Sequence", "BJO_Sequence"], use_cached_data = False, verbose = False):

    if use_cached_data:
        cached_data_path = input("Enter path to cached dataframe:  ")
        reindexed_data_df = pd.read_csv(cached_data_path)
    else:
        reindexed_data_df = get_data(output_folder = output_folder, add_peptide_seqs = add_peptide_seqs,
                                     peptide_seq_cols = peptide_seq_cols, verbose = verbose)

    # Train the convolutional neural network
    model = train_cnn(data_df = reindexed_data_df, seq_col = seq_col, aa_alphabet = aa_alphabet,
                      test_size = 0.3, only_use_characteristics = True)

# If the script is executed directly, invoke the main workflow
if __name__ == "__main__":
    use_cached_data = input("Use cached data? (Y/n)  ")
    if use_cached_data == "Y":
        use_cached_data = True
    else:
        use_cached_data = False

    main(use_cached_data = use_cached_data)