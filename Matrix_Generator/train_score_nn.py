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

    # Standardize the input data
    input_scores_matrices = [scored_result.positive_scores_2d,
                             scored_result.suboptimal_scores_2d,
                             scored_result.forbidden_scores_2d]
    stacking_matrices = []
    for input_score_matrix in input_scores_matrices:
        max_vals = input_score_matrix.max(axis=0)
        max_vals[max_vals == 0] = 1 # avoids divide-by-zero
        standardized_matrix = input_score_matrix / max_vals
        stacking_matrices.append(standardized_matrix)

    # Construct the feature matrix of shape (sample_count, position_count, channels_count)
    feature_matrix = np.stack(stacking_matrices, axis=2)

    # Concatenate with chemical characteristic classification matrix
    feature_matrix = np.concatenate([feature_matrix, scored_result.encoded_characs_3d], axis=2)
    position_count = feature_matrix.shape[1]
    channels_count = feature_matrix.shape[2]

    # Split the data and enforce consistent proportions of actual_truths with 'stratify'
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, actual_truths, test_size=0.3, random_state=42,
                                                        stratify=actual_truths)

    # Initialize the model
    model = tf.keras.models.Sequential()

    # In the 1st LocallyConnected1D layer, we slide over 4-aa frames, i.e. each aa has a ±3 aa region of influence
    model.add(tf.keras.layers.LocallyConnected1D(filters=8, kernel_size=4, strides=1, activation="tanh",
                                                 input_shape=(position_count, channels_count)))
    model.add(tf.keras.layers.Dropout(0.4))

    # In the 2nd LocallyConnected1D layer, we slide over 4-position frames again, using the filters as channels
    model.add(tf.keras.layers.LocallyConnected1D(filters=4, kernel_size=4, strides=1, activation="tanh"))
    model.add(tf.keras.layers.Dropout(0.4))

    # In the 3rd LocallyConnected1D layer, we slide over 3-position frames, with only 1 filter
    model.add(tf.keras.layers.LocallyConnected1D(filters=1, kernel_size=3, strides=1, activation="tanh"))
    model.add(tf.keras.layers.Dropout(0.4))

    # In the final layer, we flatten the inputs and feed to a single neuron for binary classification
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    # Define the optimizer and compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    print("------------------------------------------------------")
    print("Compiled LCNN architecture: ")
    model.summary()

    # Train the model
    print(f"Fitting the model to score/characteristic feature matrix")
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