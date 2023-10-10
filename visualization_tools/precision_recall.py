import os
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

    if save_path is not None:
        save_folder = save_path.rsplit("/",1)[0]
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    plt.plot(thresholds, precisions, linestyle="-", label="Precision")
    plt.plot(thresholds, recalls, linestyle="-", label="Recall")
    plt.plot(thresholds, accuracies, linestyle="-", label="Accuracy")
    plt.xlabel("Score Threshold")
    plt.legend()
    plt.savefig(save_path, format="pdf") if save_path is not None else None
    plt.show()