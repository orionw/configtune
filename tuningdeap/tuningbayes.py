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
        n_calls: int = None, n_random_starts: int = None, init_population_path: str = None, random_seed: int = None
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
        :param n_generations: the number of generations to tune
        :param population_size: the number of individuals to evaluate each generation
        :param init_population_path: the string path to a csv file containing previous tuned values and scores
        :param random_seed: set the random seed for deap
        """
        super().__init__(evaluate_outside_function, tuning_config, original_config, output_dir, verbose, minimize, timeout, random_seed)
        self.population_size = population_size
        self.n_generations = n_generations
        self.init_population_path = init_population_path

        self.validate_config()
        self.toolbox = base.Toolbox()
        self._instatiate_attributes()

        if self.verbose:
            logger.info("Finished initializing.")