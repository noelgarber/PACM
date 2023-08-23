# This script trains a dense neural network to use score values to predict whether a peptide will be positive or not

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, accuracy_score
from Matrix_Generator.ScoredPeptideResult import ScoredPeptideResult

def train_score_model(scored_result, actual_truths):
    '''
    Function that trains a simple dense neural network based on scoring information and actual truth values

    Args:
        scored_result (ScoredPeptideResult):  scoring results for the training data
        actual_truths (np.ndarray):           array of bools representing experimentally observed classifications

    Returns:
        model (Sequential):                   the trained model
        mcc_train (float):                    Matthews correlation coefficient of predictions on training data
        mcc_test (float):                     Matthews correlation coefficient of predictions on testing data
    '''

    # Assemble the matrix of features, which includes total scores, binned scores, and scores for each residue
    feature_matrix = np.hstack([scored_result.stacked_scores_original,
                               scored_result.predicted_signals_2d,
                               scored_result.suboptimal_scores_2d,
                               scored_result.forbidden_scores_2d])
    feature_count = feature_matrix.shape[1]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, actual_truths, test_size=0.2, random_state=42)

    # Assemble and train the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(feature_count,)),
        tf.keras.layers.Dense(128, activation="tanh"),
        tf.keras.layers.Dense(64, activation="tanh"),
        tf.keras.layers.Dense(32, activation="tanh"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Get predictions for training and testing data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    y_train_pred_binary = (y_train_pred > 0.5).astype(int).flatten()
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