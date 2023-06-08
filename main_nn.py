# This workflow processes SPOT images and uses them to build a neural network to predict FFATs in novel sequences

# Import standard packages
import numpy as np
import pandas as pd
import os

# Import functions for SPOT image analysis and data standardization
from Matrix_Generator.standardize_and_concatenate import main_workflow as standardized_concatenate

# Import the sequence encoder
from general_utils.aa_encoder import encode_seq

# Import functions for training the convolutional neural network
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def infer_data(peptide_sequences):
    '''
    Function to infer chemical characteristics of residues in peptide sequences and construct a dataframe of them

    Args:
        peptide_sequences (list): the list of peptide sequences to infer data about

    Returns:
        peptide_data (pd.DataFrame): a dataframe containing characteristics as tuples of length equal to sequence length
    '''

    encoded_peptide_dict = {}
    for sequence in peptide_sequences:
        encoded_sequence = encode_seq(sequence)
        encoded_peptide_dict[sequence] = encoded_sequence

    for key, value in encoded_peptide_dict:
        print(key, ":", value)

    return encoded_peptide_dict

def encode_data(data_df, seq_col = "BJO_Sequence", pass_col = "One_Passes", pass_str = "Yes", test_size = 0.2, empirical_scaling = True, return_feature_info = False):
    '''
    Function to encode data for neural network training and testing

    Args:
        data_df (pd.DataFrame):          a dataframe that must contain peptide sequences and Yes/No categorical calls
        seq_col (str):                   the column name for where sequences are stored
        pass_col (str):                  the column name for where pass/fail calls are stored
        pass_str (str):                  the string representing a positive call; e.g. "Yes" or "Pass"
        test_size (float):               the proportion of the data to include in the test split
        empirical_scaling (bool):        whether to apply StandardScaler based on empirical training data ranges,
                                         rather than theoretical allowed ranges

    Returns:
        X_train, X_test, y_train, y_test: train/test split data
        feature_info (tuple):             information about the encoded features (sequence length, number of features)
    '''

    # Check input data for required elements
    if data_df[seq_col].str.len().min() != data_df[seq_col].str.len().max():
        # Raise an error if unequal sequence lengths are found
        raise ValueError(f"encode_data found variable sequence lengths in data_df[{seq_col}] ranging from {data_df[seq_col].str.len().min()} to {data_df[seq_col].str.len().max()}, but equal lengths were expected.")

    # Encode the sequences
    sequences = data_df[seq_col].values.tolist()
    encoded_sequences = []
    for seq in sequences:
        theoretical_scaling = not empirical_scaling
        encoded_seq = encode_seq(seq, scaling = theoretical_scaling)
        encoded_sequences.append(encoded_seq)
    encoded_sequences = np.array(encoded_sequences)

    if empirical_scaling:
        scaler = StandardScaler()

        '''Reshape the array of encoded sequence from (num_samples, sequence_length, chemical_features) 
        to (num_samples * sequence_length, chemical_features), to preserve chemical features regardless of position'''
        reshaped_encoded_sequences = encoded_sequences.reshape(-1, encoded_sequences.shape[-1])
        reshaped_scaled_array = scaler.fit_transform(reshaped_encoded_sequences)

        # Restore the original shape
        encoded_sequences = reshaped_scaled_array.reshape(encoded_sequences.shape)

    # Get the positive/negative call mappings for encoding calls in binary integers
    mapping = {}
    unique_call_values = data_df[pass_col].unique().tolist()
    for unique_call in unique_call_values:
        if unique_call == pass_str:
            mapping[unique_call] = 1
        else:
            mapping[unique_call] = 0

    # Map the positive/negative calls
    mapped_calls = data_df[pass_col].map(mapping).fillna(0)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(encoded_sequences, mapped_calls, test_size = test_size, random_state = 42)
    train_test_data = (X_train, X_test, y_train, y_test)

    # Get info about number of features and length of sequences
    if return_feature_info:
        sequence_length = encoded_sequences[0].shape[0]
        chemical_features = encoded_sequences[0].shape[1]
        input_shape = (sequence_length, chemical_features)
        total_features = sequence_length * chemical_features
        feature_info = (input_shape, total_features)
        return train_test_data, feature_info
    else:
        return train_test_data

def train_bidirectional_rnn(train_test_data, verbose = True):
    '''
    Function to train a bidirectional recurrent neural network (RNN) based on encoded peptides

    Args:
        train_test_data (tuple): a tuple of (X_train, X_test, y_train, y_test), where:
                                    X_train, X_test: arrays of shape (num_samples, sequence_length, chemical_features)
                                    y_train, y_test: 1D arrays of shape (num_samples,)
        verbose (bool):          whether to display debugging information
    '''

    # Unpack the train_test_data tuple
    X_train, X_test, y_train, y_test = train_test_data
    datapoint_shape = X_train.shape[1:]
    print(f"Training the bidirectional RNN using {X_train.shape[0]} datapoints of shape {datapoint_shape}...") if verbose else None

    # Create the bidirectional RNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences = True), input_shape = datapoint_shape),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20)

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
    
def main(output_folder = None, add_peptide_seqs = True, seq_col = "BJO_Sequence", pass_col = "One_Passes",
         pass_str = "Yes", peptide_seq_cols = ["Phos_Sequence", "No_Phos_Sequence", "BJO_Sequence"],
         test_size = 0.3, empirical_scaling = True, use_cached_data = False, verbose = False):

    if use_cached_data:
        cached_data_path = input("Enter path to cached dataframe:  ")
        reindexed_data_df = pd.read_csv(cached_data_path)
    else:
        reindexed_data_df = get_data(output_folder = output_folder, add_peptide_seqs = add_peptide_seqs,
                                     peptide_seq_cols = peptide_seq_cols, verbose = verbose)

    train_test_data = encode_data(data_df = reindexed_data_df, seq_col = seq_col, pass_col = pass_col,
                                  pass_str = pass_str, test_size = test_size, empirical_scaling = empirical_scaling)

    # Train the neural network
    model = train_bidirectional_rnn(train_test_data = train_test_data)
    save_model = input("Would you like to save the model? (Y/N)  ")
    if save_model == "Y":
        if output_folder is None:
            output_folder = input("Enter the folder to save the model to: ")
        model_path = os.path.join(output_folder, "RNN_model_output.h5")
        model.save(model_path)
        print(f"Saved model to {model_path}")

# If the script is executed directly, invoke the main workflow
if __name__ == "__main__":
    use_cached_data = input("Use cached data? (Y/n)  ")
    if use_cached_data == "Y":
        use_cached_data = True
    else:
        use_cached_data = False

    main(use_cached_data = use_cached_data)