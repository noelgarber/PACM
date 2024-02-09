# This is a simple script for a random search algorithm to minimize an objective function

import numpy as np
import os
import multiprocessing
from tqdm import trange

class RandomSearchOptimizer():
    '''
    This contains a simple random search optimizer that generates random arrays and tests against an objective function
    '''

    def __init__(self, objective_function, array_len, value_range, mode, forced_values_dict=None, test_function=None,
                 optimize_train_only = True):
        '''
        Initialization function that assigns input objects to self

        Args:
            objective_function (function|partial): objective function to be minimized/maximized
            array_len (int):                       number of elements in an array to be fed to the objective function
            value_range (iterable):                list/tuple/array of (min_val, max_val), the bounds applied to values
            mode (str):                            either 'maximize' or 'minimize'
            forced_values_dict (dict):             dict of indices and forced values at those indices
            test_function (function|partial|None): objective function for the test set if given
            optimize_train_only (bool):            if False, best mean of train and test x-values is found
        '''

        # Set optimization mode
        self.mode = mode
        self.optimize_train_only = optimize_train_only

        # Assign inputs to self
        self.objective_function = objective_function
        self.test_function = test_function
        self.array_len = array_len
        self.value_range = value_range

        # Assign best_array and evaluated objective function as None; will be reset later in self.search()
        self.best_array = None
        self.x = np.inf if mode == "minimize" else -np.inf
        self.mean_x = np.inf if mode == "minimize" else -np.inf

        # Calculate and display baseline accuracy of all-ones weights
        self.forced_values_dict = forced_values_dict
        baseline_weights = np.ones(shape=self.array_len, dtype=float)
        if isinstance(self.forced_values_dict, dict):
            for idx, value in self.forced_values_dict.items():
                baseline_weights[idx] = value

        self.baseline_x = self.objective_function(baseline_weights)
        self.test_baseline_x = self.test_function(baseline_weights) if test_function is not None else None

        print(f"\tInitialized RandomSearchOptimizer; baseline objective x={self.baseline_x}")

    def search(self, sample_size):
        '''
        Main search function that can be run any number of times for any sample size; updates best_array continuously

        Args:
            sample_size (int): the number of arrays to test against the objective function
        '''

        trial_arrays = np.random.uniform(self.value_range[0], self.value_range[1], size=(sample_size, self.array_len))
        if isinstance(self.forced_values_dict, dict):
            for idx, value in self.forced_values_dict.items():
                trial_arrays[:,idx] = value

        chunk_size = int(np.ceil(trial_arrays.shape[0] / (100 * os.cpu_count())))
        trial_arrays_chunks = [trial_arrays[i:i + chunk_size] for i in range(0, len(trial_arrays), chunk_size)]

        pool = multiprocessing.Pool()

        with trange(len(trial_arrays_chunks), desc="\tRandom search optimization in progress...") as pbar:
            for chunk_results in pool.imap_unordered(self.search_chunk, trial_arrays_chunks):
                better = chunk_results[1] > self.mean_x if self.mode == "maximize" else chunk_results[1] < self.mean_x

                if better:
                    self.best_array, self.mean_x, self.x = chunk_results[0:3]
                    best_array_str = ",".join(self.best_array.round(2).astype(str))
                    if self.test_function is not None:
                        self.test_x = chunk_results[3]
                        print(f"\tRandomSearchOptimizer new record: train x={self.x}, test x={self.test_x}, arr=[{best_array_str}]")
                    else:
                        print(f"\tRandomSearchOptimizer new record: x={self.x}, arr=[{best_array_str}]")

                pbar.update()

        pool.close()
        pool.join()

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

        if self.mode == "maximize" and not self.optimize_train_only and self.test_function is not None:
            test_objective_vals = np.apply_along_axis(self.test_function, axis=1, arr=chunk)
            mean_objective_vals = np.mean([objective_vals, test_objective_vals])
            best_idx = np.nanargmax(mean_objective_vals)
        elif self.mode == "minimize" and not self.optimize_train_only and self.test_function is not None:
            test_objective_vals = np.apply_along_axis(self.test_function, axis=1, arr=chunk)
            mean_objective_vals = np.mean([objective_vals, test_objective_vals])
            best_idx = np.nanargmin(mean_objective_vals)
        elif self.mode == "maximize":
            best_idx = np.nanargmax(objective_vals)
        elif self.mode == "minimize":
            best_idx = np.nanargmin(objective_vals)
        else:
            raise ValueError(f"RandomSearchOptimizer mode is set to {self.mode}, but needs `maximize` or `minimize`")

        best_array = chunk[best_idx]
        x = objective_vals[best_idx]

        if self.test_function is not None:
            test_x = self.test_function(best_array)
            mean_x = np.mean([x, test_x])
            return (best_array, mean_x, x, test_x)
        else:
            mean_x = x
            return (best_array, mean_x, x)