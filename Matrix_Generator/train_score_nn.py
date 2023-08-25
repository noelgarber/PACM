# This script trains a dense neural network to use score values to predict whether a peptide will be positive or not

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, accuracy_score
from Matrix_Generator.ScoredPeptideResult import ScoredPeptideResult

def train_score_model(scored_result, actual_truths, graph_loss = True):
    '''
    Function that trains a simple dense neural network based on scoring information and actual truth values

    Args:
        scored_result (ScoredPeptideResult):  scoring results for the training data
        actual_truths (np.ndarray):           array of bools representing experimentally observed classifications
        graph_loss (bool):                    whether to graph the loss function decay

    Returns:
        model (Sequential):                   the trained model
        mcc_train (float):                    Matthews correlation coefficient of predictions on training data
        mcc_test (float):                     Matthews correlation coefficient of predictions on testing data
    '''

    # Assemble the matrix of features, which includes total scores, binned scores, and scores for each residue
    feature_matrix = np.hstack([scored_result.positive_scores_2d,
                                scored_result.suboptimal_scores_2d,
                                scored_result.forbidden_scores_2d,
                                scored_result.stacked_scores])
    max_feature_vals = feature_matrix.max(axis=0)
    max_feature_vals[max_feature_vals == 0] = 1 # avoid divide-by-zero errors
    feature_matrix = feature_matrix / max_feature_vals
    feature_count = feature_matrix.shape[1]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, actual_truths, test_size=0.3, random_state=10)

    # Assemble the model as a dense neural network with dropout regularization to prevent overfitting
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(feature_count,)),
        tf.keras.layers.Dense(int(feature_count / 2), activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(int(feature_count / 4), activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(int(feature_count / 8), activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # Define the optimizer and compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    print("Compiled model architecture: ")
    model.summary()

    # Train the model and display the loss curve
    print("Fitting the model to the score data...")
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

    if graph_loss:
        import matplotlib.pyplot as plt
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title("Loss & Accuracy Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss / Accuracy")
        plt.legend()
        plt.show()

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