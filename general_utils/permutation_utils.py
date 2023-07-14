import numpy as np

def permutation_split_memory(position_elements, positions, memory_limit, element_bits = 16, meshgrid_bits = 64):
    '''
    For when permuting an array across a defined number of positions, splitting can save memory; this function
    obtains the ideal split ratio that stays under an arbitrary memory limit.

    The maximum optimization occurs at a 50/50 split of positions, but has the greatest speed cost due to for-loops.

    Args:
        position_elements (int): number of possible values at each position
        positions (int):         number of positions for permutation
        memory_limit (int):      the memory limit in bytes
        element_bits (int):      the bit depth for permuted elements
        meshgrid_bits (int):     the bit depth for meshgrid elements

    Returns:
        iteration_positions (int): first part of split positions
        partial_positions (int):   second part of split positions
    '''

    permutations_count = position_elements ** positions
    permuted_elements_count = permutations_count * positions
    permuted_elements_size = permuted_elements_count * element_bits / 8

    meshgrid_elements_size = permuted_elements_count * meshgrid_bits / 8

    total_size = permuted_elements_size + meshgrid_elements_size
    iteration_positions = 0
    partial_positions = positions
    while total_size > memory_limit:
        iteration_positions += 1
        partial_positions -= 1
        if partial_positions < 1:
            memory_error = f"Failed to stay under the thread memory limit ({memory_limit / (10**9)} GB)"
            raise MemoryError(memory_error)

        # Get size of partial permuted thresholds and meshgrid
        partial_permutations_count = position_elements ** partial_positions
        partial_permuted_elements = partial_permutations_count * partial_positions
        partial_permuted_size = partial_permuted_elements * element_bits / 8
        partial_meshgrid_size = partial_permuted_elements * meshgrid_bits / 8

        # Get size of other part of thresholds for iteration
        iteration_permutations_count = position_elements ** iteration_positions
        iteration_elements_count = iteration_permutations_count * iteration_positions
        iteration_permuted_size = iteration_elements_count * element_bits / 8
        iteration_meshgrid_size = iteration_elements_count * meshgrid_bits / 8

        # Get size of array resulting from concatenating one row of iterable threshold part to partial thresholds array
        concatenated_iteration_elements = partial_permutations_count * positions
        concatenated_iteration_size = concatenated_iteration_elements * element_bits / 8

        total_size = np.sum([partial_permuted_size, partial_meshgrid_size,
                             iteration_permuted_size, iteration_meshgrid_size,
                             concatenated_iteration_size])

    return iteration_positions, partial_positions