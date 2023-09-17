# This is a simple script for a random search algorithm to minimize an objective function

import numpy as np

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

    def search(self, sample_size):
        '''
        Main search function that can be run any number of times for any sample size; updates best_array continuously

        Args:
            sample_size (int): the number of arrays to test against the objective function
        '''

        trial_arrays = np.random.uniform(self.value_range[0], self.value_range[1], size=(sample_size, self.array_len))
        objective_vals = np.apply_along_axis(self.objective_function, axis=1, arr=trial_arrays)

        if self.mode == "maximize":
            best_idx = np.nanargmax(objective_vals)
        elif self.mode == "minimize":
            best_idx = np.nanargmin(objective_vals)
        else:
            raise ValueError(f"RandomSearchOptimizer mode is set to {self.mode}, but needs `maximize` or `minimize`")

        best_array = trial_arrays[best_idx]
        x = objective_vals[best_idx]
        greater_better = x > self.x and self.mode == "maximize"
        less_better = x < self.x and self.mode == "minimize"
        if greater_better or less_better:
            self.x = x
            self.best_array = best_array
            print(f"RandomSearchOptimizer new record: x={self.x} for array: {best_array}")