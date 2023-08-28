# This script trains a locally connected neural network to predict whether a peptide will be positive or not

import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, accuracy_score
from general_utils.aa_encoder import encode_seqs_2d

def train_model(sequences_2d, actual_truths, graph_loss = True, save_path = None):
    '''
    Function that trains a simple dense neural network based on scoring information and actual truth values

    Args:
        sequences_2d (np.ndarray):          peptide sequences as a 2D array, where each row is an array of amino acids
        actual_truths (np.ndarray):         array of bools representing experimentally observed classifications
        graph_loss (bool):                  whether to graph the loss function decay
        save_path (str):                    if given, the model will be saved to this folder

    Returns:
        model (tf.keras.models.Sequential): the trained model
        stats (dict):                       dictionary of training and testing statistics
        complete_predictions (np.ndarray):  array of back-calculated prediction probabilities for the input data
    '''

    # Encode the sequences in terms of numerical descriptors, such as charge, hydrophobicity, etc.
    feature_matrix, characteristic_count = encode_seqs_2d(sequences_2d, scaling=True)

    # Reshape the feature matrix to work with LocallyConnected1D
    sample_count = feature_matrix.shape[0]
    feature_count = feature_matrix.shape[1]
    position_count = int(feature_count / characteristic_count)
    feature_matrix = feature_matrix.reshape(sample_count, characteristic_count, position_count)

    # Split the data and enforce consistent proportions of actual_truths with 'stratify'
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, actual_truths, test_size=0.3, random_state=42,
                                                        stratify=actual_truths)

    '''
    Model architecture setup; note that: 
        - The input data must be of shape (sample_count, characteristic_count, position_count)
        - For a single datapoint, the input tensor is of shape (characteristic_count, position_count)
        - The LocallyConnected1D layer slides over positions in the sequence and uses separate weights per region/frame
        - The output tensor is of shape (lc_filters, position_count - kernel_size + 1)
        - This tensor is flattened and passed to a Dense layer, which is then passed to another Dense layer
        - The final layer is a Dense layer with 1 unit and a sigmoid activation for binary classification
    '''
    lc_kernel_size = 2
    lc_strides = 1
    dropout_rate = 0.3

    model = tf.keras.models.Sequential([
        tf.keras.layers.LocallyConnected1D(filters=8, kernel_size=lc_kernel_size, strides=lc_strides,
                                           input_shape=(characteristic_count, position_count)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.LocallyConnected1D(filters=4, kernel_size=lc_kernel_size, strides=lc_strides),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Flatten(),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # Define the optimizer and compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    print("------------------------------------------------------")
    print("Compiled locally connected neural network architecture: ")
    model.summary()

    # Train the model
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')
    print(f"Fitting the model to chemical characteristic feature matrix")
    history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_test,y_test), callbacks=[tensorboard_callback])

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
        file_path = os.path.join(save_path, "interpret_scores_nn.h5")
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