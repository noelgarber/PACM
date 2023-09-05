# This script trains a dense neural network to use score values to predict whether a peptide will be positive or not

import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, accuracy_score
from sklearn.metrics import precision_recall_curve, f1_score
from Matrix_Generator.ScoredPeptideResult import ScoredPeptideResult

def make_feature_matrix(scored_result):
    '''
    Function that trains a simple dense neural network based on scoring information and actual truth values

    Args:
        scored_result (ScoredPeptideResult):  scoring results for the training data

    Returns:
        feature_matrix (np.ndarray): 3D matrix of features of shape (samples, positions, channels)
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

    return feature_matrix

def plot_loss(history, precision_recall_callback, y_train, y_test, save_path = None, save_filename = None):
    '''
    Helper function to graph the loss and accuracy curves from training

    Args:
        history (tf.keras.callbacks.History):                history object during training with model.fit()
        precision_recall_callback (PrecisionRecallCallback): the callback object holding precision and recall values
        y_train (np.ndarray):                                actual truth labels of training data
        y_test (np.ndarray):                                 actual truth labels of testing data
        save_path (str):                                     the path to save the figure to, if given
        save_filename (str):                                 filename to save the figure as, within the save_path folder

    Returns:
        None
    '''
    import matplotlib.pyplot as plt
    sample_len = len(history.history["loss"])

    # Plot loss decay
    plt.plot(history.history["loss"], label="training loss")
    plt.plot(history.history["val_loss"], label="validation loss")

    # Plot accuracy growth
    plt.plot(history.history["accuracy"], label="training accuracy")
    plt.plot(history.history["val_accuracy"], label="validation accuracy")

    # Plot evolving precision/recall
    plt.plot(precision_recall_callback.precision_train, label="training_precision")
    plt.plot(precision_recall_callback.recall_train, label="training_recall")
    plt.plot(precision_recall_callback.precision_test, label="testing_precision")
    plt.plot(precision_recall_callback.recall_test, label="testing_recall")

    # Also plot lines representing naive expected accuracy given all-false predictions
    plt.plot(np.repeat(np.mean(~y_train), sample_len),
             label="all-false train accuracy", linestyle="dashed")
    plt.plot(np.repeat(np.mean(~y_test), sample_len),
             label="all-false validation accuracy", linestyle="dashed")

    plt.title("Training & Validation Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy / Precision / Recall")
    plt.legend()
    if save_path is not None and save_filename is not None:
        plot_path = os.path.join(save_path, save_filename)
        plt.savefig(plot_path, format="pdf")
    plt.show()

def optimize_threshold(y_true, y_probs):
    '''
    Helper function that finds the ideal threshold for getting binary classifications from y_pred float values

    Args:
        y_true (np.ndarray):  array of binary actual truth values represented as 0 or 1
        y_probs (np.ndarray): array of output values to be thresholded for making binary calls

    Returns:
        y_pred (np.ndarray):       array of binary predicted truth values obtained by thresholding y_probs
        optimal_threshold (float): optimal threshold for converting y_pred to binary classifications
    '''

    # Calculate precision and recall for different threshold values
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)

    # Calculate F1-score for each threshold and find the threshold with the highest F1-score
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold = thresholds[np.argmax(f1_scores)]

    y_pred = np.greater_equal(y_probs, best_threshold).astype(int)

    return y_pred, best_threshold


class PrecisionRecallCallback(tf.keras.callbacks.Callback):
    '''
    Callback that allows for keeping track of how precision and recall evolve during training
    '''
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__()

        self.X_train = X_train
        self.y_train = y_train
        self.precision_train = []
        self.recall_train = []

        self.X_test = X_test
        self.y_test = y_test
        self.precision_test = []
        self.recall_test = []

    def on_epoch_end(self, epoch, logs=None):
        y_train_probs = self.model.predict(self.X_train)
        y_train_pred, optimal_threshold = optimize_threshold(self.y_train, y_train_probs)
        y_test_probs = self.model.predict(self.X_test)
        y_test_pred = np.greater_equal(y_test_probs, optimal_threshold)

        precision_train_val = precision_score(self.y_train, y_train_pred)
        precision_test_val = precision_score(self.y_test, y_test_pred)
        recall_train_val = recall_score(self.y_train, y_train_pred)
        recall_test_val = recall_score(self.y_test, y_test_pred)

        self.precision_train.append(precision_train_val)
        self.precision_test.append(precision_test_val)
        self.recall_train.append(recall_train_val)
        self.recall_test.append(recall_test_val)

        print(f'Epoch {epoch + 1} - Training: precision={precision_train_val}, recall={recall_train_val}',
              f'| Testing: precision={precision_test_val}, recall={recall_test_val}')


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

    # Generate the feature matrix and split into training and testing sets
    feature_matrix = make_feature_matrix(scored_result)
    sample_count, position_count, channels_count = feature_matrix.shape
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, actual_truths, test_size=0.4, random_state=42,
                                                        stratify=actual_truths)

    # Initialize the model
    model = tf.keras.models.Sequential()

    # In the 1st LocallyConnected1D layer, we slide over 3-aa frames, i.e. each aa has a ±2 aa region of influence
    model.add(tf.keras.layers.LocallyConnected1D(filters=12, kernel_size=3, strides=1, activation="tanh",
                                                 input_shape=(position_count, channels_count)))
    model.add(tf.keras.layers.Dropout(0.3))

    # In the 2nd LocallyConnected1D layer, we slide over 3-position frames, with only 1 filter
    model.add(tf.keras.layers.LocallyConnected1D(filters=1, kernel_size=3, strides=1, activation="tanh"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.3))

    # In the final layer, binary classification is achieved by thresholding a sigmoid activation output
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    # Define the optimizer and compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    print("------------------------------------------------------")
    print("Compiled LCNN architecture: ")
    model.summary()

    # Train the model
    print(f"Fitting the model to score/characteristic feature matrix")
    precision_recall_callback = PrecisionRecallCallback(X_train, y_train, X_test, y_test)
    history = model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test),
                        callbacks=[precision_recall_callback])

    # Display the loss and accuracy curves
    if graph_loss:
        plot_loss(history, precision_recall_callback, y_train, y_test, save_path, "interpret_scores_nn_training.pdf")

    # Save the model
    if save_path is not None:
        file_path = os.path.join(save_path, "interpret_scores_nn.h5")
        model.save(file_path)

    # Get binary predictions from the trained model
    y_train_probs = model.predict(X_train)
    y_test_probs = model.predict(X_test)
    y_train_pred, optimal_threshold = optimize_threshold(y_train, y_train_probs)
    y_test_pred = np.greater_equal(y_test_probs, optimal_threshold).astype(int)
    print(f"Optimal threshold for binary classification output: {optimal_threshold}")

    # Calculate a dict of output statistics for training and testing data
    accuracy_train = accuracy_score(y_train, y_train_pred)
    precision_train = precision_score(y_train, y_train_pred)
    recall_train = recall_score(y_train, y_train_pred)
    mcc_train = matthews_corrcoef(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred)
    recall_test = recall_score(y_test, y_test_pred)
    mcc_test = matthews_corrcoef(y_test, y_test_pred)

    stats = {"accuracy_train": accuracy_train,
             "precision_train": precision_train, "recall_train": recall_train, "mcc_train": mcc_train,
             "accuracy_test": accuracy_test,
             "precision_test": precision_test, "recall_test": recall_test, "mcc_test": mcc_test}

    complete_predictions = model.predict(feature_matrix)

    return (model, stats, complete_predictions)