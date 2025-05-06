"""
Module for core functionality, including variables, optimization parameters,
and objective functions.
"""

from typing import Callable, Dict, Any, Optional
from enum import Enum
import torch
from torch import Tensor
from pyvi.families import get_variational_family


class RandomVariable:
    """Represents a random variable in the model"""
    
    def __init__(self, name: str, shape: Optional[tuple] = None) -> None:
        self.name = name
        self.shape = shape
        self.prior = None
        self.approx_posterior = None
    
    def set_prior(self, prior: Callable):
        """Set the prior distribution for this variable"""
        self.prior = prior
    
    def set_posterior(self, family: str, **params):
        """Set the variational family for this variable's posterior"""
        self.approx_posterior = get_variational_family(family)(**params)


class VIProblem:
    """Main variational inference problem container"""
    
    def __init__(self):
        self.variables = {}
        self.model_log_prob = None
        self.inference_method = None
    
    def add_variable(self, name: str, shape: Optional[tuple] = None) -> RandomVariable:
        """Add a new random variable to the problem"""
        self.variables[name] = RandomVariable(name, shape)
        return self.variables[name]
    
    def set_model_log_prob(self, log_prob_fn: Callable):
        """Set the joint log probability function for the model"""
        self.model_log_prob = log_prob_fn
    
    def set_inference_method(self, method: str, **kwargs):
        """Set the inference method to use"""
        from pyvi.inference import get_inference_method
        self.inference_method = get_inference_method(method)(self, **kwargs)
    
    def solve(self, **kwargs):
        """Solve the VI problem using the configured method"""
        if self.inference_method is None:
            raise ValueError("No inference method set. Call set_inference_method() first.")
        return self.inference_method.run(**kwargs)
