# This script trains a locally connected neural network to predict whether a peptide will be positive or not

import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, accuracy_score
from general_utils.aa_encoder import encode_seqs_2d

def collapse_regions(feature_matrix, collapse_indices):
    # Helper function that selectively sums parts of the feature matrix

    positions = feature_matrix.shape[1]

    for indices in collapse_indices:
        start_index, end_index = indices
        collapse_region = feature_matrix[:, start_index:end_index + 1, :]
        summed_collapsed_region = collapse_region.mean(axis=1, keepdims=True)

        regions_list = []
        regions_list.append(feature_matrix[:, 0:start_index, :]) if start_index > 0 else None
        regions_list.append(summed_collapsed_region)
        regions_list.append(feature_matrix[:, end_index + 1:, :]) if end_index < positions - 1 else None

        feature_matrix = np.concatenate(regions_list, axis=1)

    return feature_matrix

def train_model(sequences_2d, actual_truths, graph_loss = True, save_path = None,
                collapse_indices = None, remove_indices = None, verbose = True):
    '''
    Function that trains a simple dense neural network based on scoring information and actual truth values

    Args:
        sequences_2d (np.ndarray):          peptide sequences as a 2D array, where each row is an array of amino acids
        actual_truths (np.ndarray):         array of bools representing experimentally observed classifications
        graph_loss (bool):                  whether to graph the loss function decay
        save_path (str):                    if given, the model will be saved to this folder
        collapse_indices (None|list):       list of tuples of (start_index, end_index) for regions of the motif to merge
        remove_indices (None|list):         list of indices to drop (manually marked unimportant)

    Returns:
        model (tf.keras.models.Sequential): the trained model
        stats (dict):                       dictionary of training and testing statistics
        complete_predictions (np.ndarray):  array of back-calculated prediction probabilities for the input data
    '''

    # Seqs are encoded in terms of chemical characteristics described numerically, e.g. charge, weight, rings, etc.
    feature_matrix, characteristic_count = encode_seqs_2d(sequences_2d, scaling=True, output_dims=3)
    feature_matrix = np.transpose(feature_matrix, axes=(0,2,1)) # final shape = (samples, positions, channels)
    print(f"feature_matrix shape = {feature_matrix.shape}") if verbose else None
    print("----------------------------")

    # If specified, collapse and/or remove defined positions
    if collapse_indices is not None:
        if verbose:
            print(f"Collapsing defined regions: ") if verbose else None
            for start, end in collapse_indices:
                print(f"{start}-{end}")
        feature_matrix = collapse_regions(feature_matrix, collapse_indices)
        print(f"New feature_matrix shape = {feature_matrix.shape}",
              f"\n----------------------------") if verbose else None

    if remove_indices is not None:
        print(f"Removing defined post-collapse indices ({remove_indices})...") if verbose else None
        feature_matrix = np.delete(feature_matrix, remove_indices, axis=1)
        print(f"New feature_matrix shape = {feature_matrix.shape}",
              f"\n----------------------------") if verbose else None

    # Split the data and enforce consistent proportions of actual_truths with 'stratify'
    positions = feature_matrix.shape[1]
    channels = characteristic_count
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, actual_truths, test_size=0.3,
                                                        random_state=42, stratify=actual_truths)

    print("X_train: ")
    print(X_train)
    print("----------------------------")

    '''
    Model architecture setup; note that: 
        - The input data must be of shape (sample_count, characteristic_count, position_count)
        - For a single datapoint, the input tensor is of shape (characteristic_count, position_count)
        - The LocallyConnected1D layer slides over positions in the sequence and uses separate weights per region/frame
        - The output tensor is of shape (lc_filters, position_count - kernel_size + 1)
        - This tensor is flattened and passed to a Dense layer, which is then passed to another Dense layer
        - The final layer is a Dense layer with 1 unit and a sigmoid activation for binary classification
    '''

    # Initialize the model
    model = tf.keras.models.Sequential()

    # The 1st layer is a LocallyConnected1D layer that treats sequence positions as the temporal dimension to slide over
    frame_width = 2
    model.add(tf.keras.layers.LocallyConnected1D(filters=16, kernel_size=frame_width, strides=1, activation="tanh",
                                                 input_shape=(positions, channels)))
    model.add(tf.keras.layers.Dropout(0.4))

    # The 2nd layer is a fully connected layer receiving flattened input from the previous layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(12, activation="tanh"))
    model.add(tf.keras.layers.Dropout(0.4))

    # The 3rd layer is a smaller fully connected layer
    model.add(tf.keras.layers.Dense(6, activation="tanh"))
    model.add(tf.keras.layers.Dropout(0.4))

    # Use a final single-unit dense layer to output a binary-interpretable categorization value
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    # Define the optimizer and compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    print("------------------------------------------------------")
    print("Compiled LCNN architecture: ")
    model.summary()

    # Train the model
    print(f"Fitting the model to chemical characteristic feature matrix")
    history = model.fit(X_train, y_train, epochs=300, validation_data=(X_test,y_test))

    # Display the loss and accuracy curves
    if graph_loss:
        import matplotlib.pyplot as plt
        sample_len = len(history.history["loss"])

        # Plot loss decay
        plt.plot(history.history["loss"], label="training loss", color="blue")
        plt.plot(history.history["val_loss"], label="validation loss", color="magenta")

        # Plot accuracy growth
        plt.plot(history.history["accuracy"], label="training accuracy", color="green")
        plt.plot(history.history["val_accuracy"], label="validation accuracy", color="red")

        # Also plot lines representing naive expected accuracy given all-false predictions
        plt.plot(np.repeat(np.mean(~y_train), sample_len),
                 label="all-false train accuracy", linestyle="dashed", color = "green")
        plt.plot(np.repeat(np.mean(~y_test), sample_len),
                 label="all-false validation accuracy", linestyle="dashed", color = "red")

        plt.title("Training & Validation Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss / Accuracy / Precision / Recall")
        plt.legend()
        plt.show()

    # Save the model
    if save_path is not None:
        file_path = os.path.join(save_path, "lcnn.h5")
        model.save(file_path)

    # Get binary predictions from the trained model
    y_train_pred = model.predict(X_train)
    y_train_pred_binary = (y_train_pred > 0.5).astype(int).flatten()
    y_test_pred = model.predict(X_test)
    y_test_pred_binary = (y_test_pred > 0.5).astype(int).flatten()

    # Calculate a dict of output statistics for training and testing data
    accuracy_train = accuracy_score(y_train, y_train_pred_binary)
    precision_train = precision_score(y_train, y_train_pred_binary)
    recall_train = recall_score(y_train, y_train_pred_binary)
    mcc_train = matthews_corrcoef(y_train, y_train_pred_binary)
    accuracy_test = accuracy_score(y_test, y_test_pred_binary)
    precision_test = precision_score(y_test, y_test_pred_binary)
    recall_test = recall_score(y_test, y_test_pred_binary)
    mcc_test = matthews_corrcoef(y_test, y_test_pred_binary)

    stats = {"accuracy_train": accuracy_train,
             "precision_train": precision_train, "recall_train": recall_train, "mcc_train": mcc_train,
             "accuracy_test": accuracy_test,
             "precision_test": precision_test, "recall_test": recall_test, "mcc_test": mcc_test}

    complete_predictions = model.predict(feature_matrix)

    return (model, stats, complete_predictions)