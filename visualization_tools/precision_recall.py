import os
import numpy as np
import matplotlib.pyplot as plt

def plot_precision_recall(precisions, recalls, accuracies, thresholds, save_path = None):
    '''
    Plots precision, recall, and accuracy against various possible thresholds

    Args:
        precisions (np.ndarray): precision (PPV) values at each threshold in thresholds
        recalls (np.ndarray):    recall (NPV) values at each threshold in thresholds
        accuracies (np.ndarray): accuracy values at each threshold in thresholds
        thresholds (np.ndarray): possible thresholds for making binary classification calls from continuous scores
        save_path (str):         path to save the plot into

    Returns:
        None
    '''

    top_accuracy_idx = np.nanargmax(accuracies)
    top_threshold = thresholds[top_accuracy_idx]
    top_accuracy = accuracies[top_accuracy_idx]
    top_precision = precisions[top_accuracy_idx]
    top_recall = recalls[top_accuracy_idx]

    if save_path is not None:
        save_folder = save_path.rsplit("/",1)[0]
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    plt.plot(thresholds, precisions, color="b", linestyle="-", label="Precision")
    plt.plot(thresholds, recalls, color="r", linestyle="-", label="Recall")
    plt.plot(thresholds, accuracies, color="g", linestyle="-", label="Accuracy")

    plt.axhline(y=top_precision, color="b", linestyle='--', alpha=0.5)
    plt.axhline(y=top_recall, color="r", linestyle='--', alpha=0.5)
    plt.axhline(y=top_accuracy, color="g", linestyle='--', alpha=0.5)
    plt.axvline(x=top_threshold, color="black", linestyle='--', alpha=0.5)

    plt.xlabel("Score Threshold")
    plt.legend()

    plt.savefig(save_path, format="pdf") if save_path is not None else None
    plt.show()

    return (top_threshold, top_accuracy)