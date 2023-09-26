import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return y

def fit_sigmoid(x_values, y_values, save_as = None, x_label = None, y_label = None, title = None):
    '''
    Function to perform logistic regression on a set of x- and y-values

    Args:
        x_values (np.ndarray): data x-values
        y_values (np.ndarray): data y-values

    Returns:
        None
    '''

    if save_as is not None:
        parent_directory = save_as.rsplit("/",1)[0]
        if not os.path.exists(parent_directory):
            os.makedirs(parent_directory)

    finite_mask = np.logical_and(np.isfinite(x_values), np.isfinite(y_values))
    x_values = x_values[finite_mask]
    y_values = y_values[finite_mask]
    p0 = [y_values.max(), np.median(x_values), 1, y_values.min()]
    try:
        params, covariance = curve_fit(sigmoid, x_values, y_values, p0, method="dogbox")
        success = True
    except Exception as e:
        warnings.warn(f"Caught exception in fit_sigmoid(): {e}")
        success = False

    # Create a scatter plot of the data and the fitted sigmoid function
    if success:
        # Get the R2 value
        y_pred = np.array([sigmoid(x, *params) for x in x_values])
        r_squared = r2_score(y_values, y_pred)

        # Generate points for defined sigmoid function
        x_range = x_values.max() - x_values.min()
        extrapolation = 0.2 * x_range
        x_sigmoid = np.linspace(x_values.min() - extrapolation, x_values.max() + extrapolation, 1000)
        y_sigmoid = np.array([sigmoid(x, *params) for x in x_sigmoid])

        plt.scatter(x_values, y_values, label="Data", color="blue", marker="o", s=7)
        plt.plot(x_sigmoid, y_sigmoid, label="Logistic Regression", color="red")

        # Add the formula and R-squared value as text on the graph
        L, k, x0, b = params
        k_str = f"{-k:.2f}"
        x0_str = "+" + f"{x0:.2f}" if x0 >= 0 else f"{x0:.2f}"
        b_str = "+" + f"{b:.2f}" if b >= 0 else f"{b:.2f}"
        formula_latex = r"y = \frac{" + f"{L:.2f}" + r"}{1+e^{" + k_str + "(x" + x0_str + ")}}" + b_str
        r2_latex = f"R^2={r_squared:.2f}"
        full_latex = f"${formula_latex}$\n${r2_latex}$"
        plt.text(2, 0.7, full_latex, fontsize=12)

    else:
        plt.scatter(x_values, y_values, label="Data", color="blue", marker="o")

    x_label = "X-values" if x_label is None else x_label
    y_label = "Y-values" if y_label is None else y_label
    title = "Scatter Plot with Logistic Regression" if title is None else title

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_as is not None:
        plt.savefig(save_as, format="pdf")
        plt.show()
    else:
        plt.show()

    plt.clf()