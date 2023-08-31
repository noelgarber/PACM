# This script trains a dense neural network to use score values to predict whether a peptide will be positive or not

import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, accuracy_score
from Matrix_Generator.ScoredPeptideResult import ScoredPeptideResult

def train_score_model(scored_result, actual_truths, graph_loss = True, save_path = None):
    '''
    Function that trains a simple dense neural network based on scoring information and actual truth values

    Args:
        scored_result (ScoredPeptideResult):  scoring results for the training data
        actual_truths (np.ndarray):           array of bools representing experimentally observed classifications
        graph_loss (bool):                    whether to graph the loss function decay
        save_path (str):                      if given, the model will be saved to this folder

    Returns:
        model (Sequential):                   the trained model
        mcc_train (float):                    Matthews correlation coefficient of predictions on training data
        mcc_test (float):                     Matthews correlation coefficient of predictions on testing data
    '''

    # Assemble the matrix of features, which includes total scores, binned scores, and scores for each residue
    feature_matrix = np.hstack([scored_result.positive_scores_2d,
                                scored_result.suboptimal_scores_2d,
                                scored_result.forbidden_scores_2d])
    max_feature_vals = feature_matrix.max(axis=0)
    max_feature_vals[max_feature_vals == 0] = 1 # avoid divide-by-zero errors
    feature_matrix = feature_matrix / max_feature_vals

    # Add integer-encoded residue groups by position as additional features
    feature_matrix = np.hstack([feature_matrix, scored_result.stacked_encoded_characs])
    feature_count = feature_matrix.shape[1]

    # Split the data and enforce consistent proportions of actual_truths with 'stratify'
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, actual_truths, test_size=0.3, random_state=42,
                                                        stratify=actual_truths)

    # Assemble the model as a dense neural network with dropout regularization to prevent overfitting
    neuron_counts = np.array([int(feature_count * 1/4), int(feature_count * 1/4), int(feature_count * 1/4)])
    neuron_counts[neuron_counts < 4] = 4 # minimum of 4 neurons per layer
    print(f"Neuron counts: {neuron_counts}")
    dropout_rate = 0.3
    reg_strength = 0.01
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(feature_count,)),
        tf.keras.layers.Dense(neuron_counts[0], kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(neuron_counts[1], kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(neuron_counts[2], kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # Define the optimizer and compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    print("------------------------------------------------------")
    print("Compiled model architecture for score interpretation: ")
    model.summary()

    # Train the model
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')
    batch_size = int(len(y_train))
    print(f"Fitting the model to the score data; input shape = {X_train.shape[0], X_train.shape[1]}")
    history = model.fit(X_train, y_train, epochs=200, batch_size=batch_size, validation_data=(X_test, y_test),
                        callbacks=[tensorboard_callback])

    # Display the loss and accuracy curves
    if graph_loss:
        import matplotlib.pyplot as plt
        sample_len = len(history.history["loss"])

        # Plot loss decay
        plt.plot(history.history["loss"], label="training loss", color="blue")
        plt.plot(history.history["val_loss"], label="validation loss", color="magenta")

        # Plot accuracy growth
        plt.plot(history.history['accuracy'], label='training accuracy', color="green")
        plt.plot(history.history['val_accuracy'], label='validation accuracy', color="red")

        # Also plot lines representing naive expected accuracy given all-false predictions
        plt.plot(np.repeat(np.mean(~y_train), sample_len),
                 label="all-false train accuracy", linestyle="dashed", color = "green")
        plt.plot(np.repeat(np.mean(~y_test), sample_len),
                 label="all-false validation accuracy", linestyle="dashed", color = "red")

        plt.title("Loss & Accuracy Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss / Accuracy")
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