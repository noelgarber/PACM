import numpy as np
from general_utils.user_helper_functions import get_position_copies

def permute_weights(slim_length, position_copies = None):
    '''
    Simple function to generate permutations of weights from 0-3 for scoring a peptide motif of defined length

    Args:
        slim_length (int): 	    the length of the motif for which weights are being generated
        position_copies (dict): a dictionary of position --> copy_num, where the sum of dict values equals slim_length

    Returns:
        expanded_weights_array (np.ndarray): an array of shape (permutations_number, slim_length)
    '''

    if position_copies is None:
        position_copies = get_position_copies(slim_length)

    # Check that the dictionary of position copies is the correct length
    if slim_length != sum(position_copies.values()):
        raise ValueError(f"permute_weights error: slim_length ({slim_length}) is not equal to position_copies dict values sum ({sum(position_copies.values())})")

    # Get permutations of possible weights at each position
    permutations_length = slim_length
    for position, total_copies in position_copies.items():
        permutations_length = permutations_length - (total_copies - 1)
    permuted_weights = np.array(np.meshgrid(*([[3, 2, 1, 0]] * permutations_length))).T.reshape(-1, permutations_length)

    # Expand permutations to copied columns
    expanded_weights_list = []
    for position, total_copies in position_copies.items():
        current_column = permuted_weights[:, position:position + 1]
        if total_copies == 1:
            expanded_weights_list.append(current_column)
        else:
            repeated_column = np.repeat(current_column, total_copies, axis=1)
            expanded_weights_list.append(repeated_column)
    expanded_weights_array = np.hstack(expanded_weights_list)

    print(f"Shape of expanded_weights_array: {expanded_weights_array.shape}")

    return expanded_weights_array
