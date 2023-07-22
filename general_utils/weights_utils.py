import numpy as np
from general_utils.user_helper_functions import get_possible_weights

def permute_weights(slim_length, possible_weights = None, dtype = float):
    '''
    Simple function to generate permutations of weights from 0-3 for scoring a peptide motif of defined length

    Args:
        slim_length (int): 	     the length of the motif for which weights are being generated
        possible_weights (list): a list of numpy arrays with possible weights at each position for permutation

    Returns:
        result (np.ndarray):     an array of shape (permutations_number, slim_length)
    '''

    if possible_weights is None:
        possible_weights = get_possible_weights(slim_length)

    permutations_count = np.prod([len(arr) for arr in possible_weights])

    result = np.empty((permutations_count, slim_length), dtype = dtype)
    meshgrid_arrays = np.meshgrid(*possible_weights)
    for i, arr in enumerate(meshgrid_arrays):
        result[:, i] = arr.ravel()

    print(f"Shape of permuted weights array: {result.shape}")

    return result
