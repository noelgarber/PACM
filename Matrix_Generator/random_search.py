# This is a simple script for a random search algorithm to minimize an objective function

import numpy as np
import os
import multiprocessing
from tqdm import trange

class RandomSearchOptimizer():
    '''
    This contains a simple random search optimizer that generates random arrays and tests against an objective function
    '''

    def __init__(self, objective_function, array_len, value_range, mode):
        '''
        Initialization function that assigns input objects to self

        Args:
            objective_function (function): objective function to be minimized/maximized
            array_len (int):               number of elements in an array to be fed to the objective function
            value_range (iterable):        list/tuple/array of (min_val, max_val), the bounds applied to array values
            mode (str):                    either 'maximize' or 'minimize'
        '''

        # Set optimization mode
        self.mode = mode

        # Assign inputs to self
        self.objective_function = objective_function
        self.array_len = array_len
        self.value_range = value_range

        # Assign best_array and evaluated objective function as None; will be reset later in self.search()
        self.best_array = None
        self.x = np.inf if mode == "minimize" else -np.inf

        # Calculate and display baseline accuracy of all-ones weights
        baseline_weights = np.ones(shape=self.array_len, dtype=float)
        self.baseline_x = self.objective_function(baseline_weights)
        print(f"Initialized RandomSearchOptimizer; baseline unweighted accuracy objective x={self.baseline_x}")

    def search(self, sample_size):
        '''
        Main search function that can be run any number of times for any sample size; updates best_array continuously

        Args:
            sample_size (int): the number of arrays to test against the objective function
        '''

        trial_arrays = np.random.uniform(self.value_range[0], self.value_range[1], size=(sample_size, self.array_len))

        chunk_size = int(np.ceil(trial_arrays.shape[0] / (100 * os.cpu_count())))
        trial_arrays_chunks = [trial_arrays[i:i + chunk_size] for i in range(0, len(trial_arrays), chunk_size)]

        pool = multiprocessing.Pool()

        with trange(len(trial_arrays_chunks), desc="Random search optimization in progress...") as pbar:
            for chunk_results in pool.imap_unordered(self.search_chunk, trial_arrays_chunks):
                if chunk_results[1] > self.x and self.mode == "maximize":
                    self.best_array, self.x = chunk_results
                    print(f"\tRandomSearchOptimizer new record: x={self.x}")
                elif chunk_results[1] < self.x and self.mode == "minimize":
                    self.best_array, self.x = chunk_results
                    print(f"\tRandomSearchOptimizer new record: x={self.x}")

                pbar.update()

    def search_chunk(self, chunk):
        '''
        Helper function to parallelize the search task

        Args:
            chunk (np.ndarray): chunk of trial_arrays

        Returns:
            best_array (np.ndarray): best array
            x (float):               the value of the objective function for best_array
        '''

        objective_vals = np.apply_along_axis(self.objective_function, axis=1, arr=chunk)

        if self.mode == "maximize":
            best_idx = np.nanargmax(objective_vals)
        elif self.mode == "minimize":
            best_idx = np.nanargmin(objective_vals)
        else:
            raise ValueError(f"RandomSearchOptimizer mode is set to {self.mode}, but needs `maximize` or `minimize`")

        best_array = chunk[best_idx]
        x = objective_vals[best_idx]

        return (best_array, x)