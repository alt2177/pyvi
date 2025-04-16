"""
Module for core functionality, including variables, optimization parameters,
and objective functions.
"""

import numpy as np

class Variable:
    """
    Class to hold an optimization variable
    """
    def __init__(self, shape) -> None:
        self.value = np.random.randn(*shape)
        self.shape = shape


class Parameter:
    """
    Class to hold parameters for optimization
    """
    def __init__(self, shape) -> None:
        self.value = np.zeros(shape)
        self.shape = shape


class Problem:
    """
    Class for the objective function/problem to be optimized
    """
    def __init__(self, objective, constraints=None) -> None:
        self.objective = objective
        self.constraints = constraints if constraints is not None else []

    def solve(self, method='gradient_ascent', max_iters=1000, learning_rate=0.01):
        if method == 'gradient_ascent':
            self._gradient_ascent(max_iters, learning_rate)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _gradient_ascent(self, max_iters, learning_rate):
        # for _ in range(max_iters):
        #     grad = self._compute_gradient()
        #     self.objective.variable.value += learning_rate * grad
        raise NotImplementedError

    def _compute_gradient(self):
        # Placeholder for gradient computation
        # return np.random.randn(*self.objective.variable.shape)
        raise NotImplementedError


class Objective:
    def __init__(self, expression, variable):
        self.expression = expression
        self.variable = variable
