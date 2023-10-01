# This script trains a bidirectional recurrent neural network based on in vitro peptide binding data

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Sequential
from sklearn.metrics import matthews_corrcoef

class EncodedPeptides:
    '''
    Represents encoded peptide sequences carrying positional information, along with positive/negative binary calls,
    and relative binding intensities as floats from 0.0 to 1.0.
    '''
    def __init__(self, peptide_seqs, peptide_seqs_2d, signal_values, passes_bools, frame_length = 3, test_size = 0.2):
        '''
        Initialization function that encodes a 2D array of peptides of equal lengths, where each row is a peptide and
        each cell is an amino acid as a single letter code.

        Args:
            peptides_2d (np.ndarray):    2D array of peptides of shape (peptides_count, peptide_length)
            signal_values (np.ndarray):  array of binding signal values
            passes_bools (np.ndarray):   array of True/False bools for whether each peptide passes as a positive hit
        '''

        # Check input data
        pep_count = len(peptide_seqs)
        sig_count = len(signal_values)
        bool_count = len(passes_bools)
        if pep_count != sig_count:
            raise ValueError(f"Non-matching lengths of peptide_seqs ({pep_count}) and signal_values ({sig_count})")
        elif pep_count != bool_count:
            raise ValueError(f"Non-matching lengths of peptide_seqs ({pep_count}) and passes_bools ({bool_count})")

        # Encode peptide sequences
        unique_residues = np.unique(peptide_seqs_2d)
        unique_residues.sort()
        encoded_seqs_2d = np.empty(shape=peptide_seqs_2d.shape, dtype=int)
        self.aa_encoding = {}

        for i, unique_residue in enumerate(unique_residues):
            encoded_seqs_2d[peptide_seqs_2d == unique_residue] = i
            self.aa_encoding[unique_residue] = i

        # Encode overlapping frame vectors into a dataframe such that each row is a datapoint
        frame_count = peptide_seqs_2d.shape[1] - frame_length
        encoded_data = []

        for frame_start in np.arange(frame_count):
            frame_end = frame_start + frame_length
            frame_seqs = peptide_seqs_2d[:,frame_start:frame_end]
            repeated_position = np.repeat(frame_start, len(frame_seqs)).reshape(-1,1)
            encoded_vectors = np.concatenate([repeated_position, frame_seqs], axis=1)
            encoded_data.append(encoded_vectors)

        encoded_data = np.array(encoded_data)
        self.encoded_data = encoded_data

        # Assign relative binding values and pass/fail bools
        self.intensities = signal_values / np.max(signal_values)
        self.binary_calls = passes_bools

        # Do train/test splitting
        self.train_test_split(test_size)

    def train_test_split(self, test_size = 0.2):
        '''
        Simple function that splits the data into training and testing sets

        Args:
            test_size (float): value from 0.0 to 1.0 representing the proportion of data to separate for testing

        Returns:
            None; assigns to self
        '''

        random_seed = 42  # Set a random seed for reproducibility
        x_train, x_test, strength_train, strength_test, y_train, y_test = train_test_split(self.encoded_data,
                                                                                           self.intensities,
                                                                                           self.binary_calls,
                                                                                           test_size = test_size,
                                                                                           random_state = random_seed)

        self.training_data = x_train
        self.training_intensities = strength_train
        self.training_calls = y_train

        self.testing_data = x_test
        self.testing_intensities = strength_test
        self.testing_calls = y_test

def train_rnn(encoded_peptides):
    '''
    Main function to train a bidirectional recurrent neural network using data from an EncodedPeptides object

    Args:
        encoded_peptides (EncodedPeptides): the encoded peptides object containing peptides, calls, and intensities

    Returns:
        model (tf.keras.layers.Sequential): the trained sequential deep learning model
        train_accuracy (float):             accuracy of the model evaluated on training data
        test_accuracy (float):              accuracy of the model evaluated on testing data
        train_mcc (float):                  Matthews correlation coefficient of the model evaluated on training data
        test_mcc (float):                   Matthews correlation coefficient of the model evaluated on testing data
    '''

    # Get the training data from the EncodedPeptides object
    input_shape = (encoded_peptides.training_data.shape[1], encoded_peptides.training_data.shape[2])

    # Define the model architecture
    model = Sequential([
        Dense(64, activation="relu", input_shape=input_shape),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(64)),
        Dense(1, activation="sigmoid"),  # Output for boolean calls
        Dense(1, activation="linear")    # Output for intensity values with linear activation
    ])

    # Compile the model
    model.compile(optimizer="adam", loss=["binary_crossentropy", "mean_squared_error"], loss_weights=[1.0, 1.0])

    # Train the model
    model.fit(encoded_peptides.training_data,
              [encoded_peptides.training_calls, encoded_peptides.training_intensities],
              epochs=100, batch_size=8)

    # Calculate predictions
    train_predictions = model.predict(encoded_peptides.training_data)[0] > 0.5  # Assuming threshold is 0.5
    test_predictions = model.predict(encoded_peptides.testing_data)[0] > 0.5

    # Compute metrics
    train_accuracy = np.mean(train_predictions == calls)
    test_accuracy = np.mean(test_predictions == encoded_peptides.testing_calls)
    train_mcc = matthews_corrcoef(calls, train_predictions)
    test_mcc = matthews_corrcoef(encoded_peptides.testing_calls, test_predictions)

    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Training MCC: {train_mcc:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    print(f"Testing MCC: {test_mcc:.4f}")

    return model, train_accuracy, test_accuracy, train_mcc, test_mcc