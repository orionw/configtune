import pandas as pd
import numpy as np
import skopt
import copy
import logging
import math
import os
import random
import typing
import time
import warnings

from tuningdeap.base_tuner import TuningBase
from tuningdeap.config_mapping import set_by_path, get_paths

from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize

logger = logging.getLogger(__name__)

class TuningBayes(TuningBase):
    """
    A class to tune algorithms with bayesian optimization. Can handle a config file. Wraps the `sckit-optimize` package.
    """

    def __init__(
        self, evaluate_outside_function: typing.Callable, tuning_config: dict, original_config: dict = None,
        output_dir: str = None, verbose: bool = False, minimize: bool = True, timeout: float = float("inf"),
        n_calls: int = 1, n_random_starts: int = 1, random_seed: int = None
    ):
        """
        Sets up the class, creates the config mapper if needed, the evaluate function, and the genetic algorithm parameters
        :param evaluate_outside_function: the function to evaluate for the outside caller
        :param tuning_config: the config for the genetic algorithm
        :param original_config: the original_configuration file, if needed, so that we can evaluate
        :param output_dir: the location to write the output files
        :param verbose: whether to report progress to logging while running
        :param minimize: if True, minimize evaluate_outside_function; else maximize it
        :param timeout: the time in seconds that tuningDEAP should continue to generate populations
        :param n_calls: the number of calls to make
        :param n_random_starts: the number of times to randomly start
        :param random_seed: set the random seed
        """
        super().__init__(evaluate_outside_function, tuning_config, original_config, output_dir, verbose, minimize, timeout, random_seed)
        self.n_calls = n_calls
        self.n_random_starts = n_random_starts

        self.validate_config()
        self.space = self._instantiate_space()

        if self.random_seed is None:
            self.random_seed = random.randint(0, 9999999)
        if self.verbose:
            logger.info("Finished initializing.")

    def run(self):
        res_gp = gp_minimize(self.evaluate_function, self.space, n_calls=self.n_calls, random_state=self.random_seed, 
                             n_random_starts=self.n_random_starts, verbose=self.verbose)
        if self.verbose or self.output_dir is not None:
            self.create_dataset_from_results(res_gp.x_iters, res_gp.func_vals, self.output_dir is not None, self.verbose, prefix="bayes")

        return res_gp.x, res_gp.fun
 
    def validate_config(self):
        pass

    def _instantiate_space(self):
        space = []
        self.bool_values = []
        self.categorical_values = {}
        self.order_of_keys = []
        # for each attribute given to model, set up the required limits and parameters for the genetic algorithm
        for attribute_name, attribute_map in self.tuning_config["attributes"].items():
            # gather the needed info
            self.order_of_keys.append(attribute_name)
            param_type = attribute_map["type"]
            if param_type not in ["bool", "categorical"]:
                min_val = attribute_map["min"]
                max_val = attribute_map["max"]
                step_size =  attribute_map["step"] if "step" in attribute_map else 1
                if step_size != 1:
                    warnings.warn("TuningBayes can not take a step value: ignoring...")
            # create the values in the program
            if param_type == "int":
                space.append(Integer(min_val, max_val, name=attribute_name))
            if param_type == "bool":
                space.append(Integer(0, 1, name=attribute_name))
                self.bool_values.append(attribute_name)
            if param_type == "float":
                space.append(Real(min_val, max_val, name=attribute_name))
            if param_type == "categorical":
                space.append(Integer(0, len(attribute_map["values"]) - 1, name=attribute_name))
                self.categorical_values[attribute_name] = attribute_map["values"]

        assert len(space) != 0, "No values were added for the bayesian optimization"
        return space
