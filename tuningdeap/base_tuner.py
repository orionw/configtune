import copy
import logging
import math
import os
import random
import typing
import time

from deap import algorithms, base, creator, tools
import numpy as np
import pandas as pd

from tuningdeap.config_mapping import set_by_path, get_paths

logger = logging.getLogger(__name__)


class TuningBase:
    """
    A base class for tuning algorithms. Can handle a config file.
    """

    def __init__(
        self, evaluate_outside_function: typing.Callable, tuning_config: dict, original_config: dict = None,
        output_dir: str = None, verbose: bool = False, minimize: bool = True, timeout: float = float("inf"), random_seed: int = None
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
        self.tuning_config: dict = tuning_config
        self.using_config = False

        self.output_dir = output_dir
        if self.output_dir is not None and not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        self.minimize = minimize
        self.timeout = timeout
       
        self.random_seed = random_seed
        if self.random_seed is not None:
            random.seed(self.random_seed)

        self.verbose = verbose
        if self.verbose:
            logger.setLevel(logging.INFO)
            logger.info("verbose is set to: {}".format(self.verbose))

        if original_config is not None:
            self.original_config: dict = original_config
            self.using_config = True
            self.map_config: dict = self.map_tuning_config_to_original()

        
        self.evaluate_outside_function = evaluate_outside_function
        self.evaluate_function: typing.Callable = self.create_evaluate_func()

    def create_evaluate_func(self) -> typing.Callable:
        """
        The function to create the function used by the genetic algorithm.
        We have to wrap the evaluation function with another function to change the chromosomes into a dict like the `original_config` file.
        """
        def evaluate_function(values: typing.List) -> typing.Tuple:
            """
            Simply map the chromosome values into a dict like the original config file and return that to the evaluate function
            :param values: the "chromosone" or parameter for this "individual"
            :return a tuple containing the score of fitness
            """
            config = self.map_tuning_config_back(values)
            try:
                return self.evaluate_outside_function(config)
            except Exception as e:
                if self.verbose:
                    logger.info("There was an exception evaluating: {}".format(e))
                return tuple((float('inf'), )) if self.minimize else tuple((float("-inf"), ))
        return evaluate_function

    def enforce_limits(self, init_population: typing.List[typing.List]) -> typing.List[typing.List]:
        """
        Enforces the given limits on the mutated/crossovered population since deap does not.
        :param init_population: a list of lists containing `inividual's chromosomes`
        :return a list of lists representing the valid members of the population
        """
        final_population = []
        for individual in init_population:
            invalid = False
            for index, parameter in enumerate(individual):
                parameter_info = self.tuning_config["attributes"]
                parameter_name = self.order_of_keys[index]
                parameter_map = parameter_info[parameter_name]
                attribute_type = parameter_map["type"]

                if attribute_type not in ["bool", "categorical"]:
                    min_val = parameter_map["min"]
                    max_val = parameter_map["max"]
                    step_size =  parameter_map["step"] if "step" in parameter_map else 1
                    if min_val <= parameter <= max_val:
                        try:
                            # see if strict bounds are enforced
                            if "strict" in parameter_info and self.enforce_steps(parameter, min_val, max_val, step_size):
                                continue
                        except IndexError:
                            continue
                    else:
                        if self.verbose:
                            logger.info("Rejecting float/int parameter for individual: {}={} with bounds: {}".format(parameter_name, parameter,
                                        " ".join([str(item) for item in [min_val, max_val, step_size]])))
                        invalid = True
                        break

                elif attribute_type == "bool":
                    # handle the boolean case
                    if parameter == 0 or parameter == 1:
                        continue
                    else: 
                        if self.verbose:
                            logger.info("Rejecting boolean parameter for individual: {} with bounds: {}".format(parameter_name, parameter))
                        invalid = True
                        break

                elif attribute_type == "categorical":
                    values = parameter_map["values"]
                    if 0 <= parameter <= len(values):
                        continue
                    else:
                        if self.verbose:
                            logger.info("Rejecting categorical parameter for individual: {}={} with bounds: 0-{}".format(parameter_name, parameter,
                                        " ".join(len(values))))
                        invalid = True
                        break
                else:
                    raise Exception("Wrong type of value for parameter: {}".format(attribute_type))

            if not invalid:
                final_population.append(copy.deepcopy(individual))

        if self.verbose:
            logger.info("Enforcing limits for the population. Have {} after rejecting the invalid: population_size is {}".format(len(final_population), self.population_size))
        return final_population

    def enforce_steps(self, parameter, min_val: int, max_val: int, step_size) -> bool:
        # TODO: add the enforcement for this step
        return True

    def rand_with_step(self, low: float, high: float, step: float, count: int = 1, bias: bool = True):
        """
        A function to return a random float with steps
        :param low: the `low` step to start out on
        :parm high: the `high` end of the range of options
        :param step: the step size to take between `low` and `high`
        :param count: how many floats to return
        :param bias: whether to start on the `low` step
        :return a float between `low` and `high`, with steps sizes of size `step`
        """
        n = 1 / step
        if count > 1:
            val = np.random.randint(low * n, high * n, count) * step
        else:
            val= np.random.randint(low * n,high * n) * step

        if bias:
            bias = math.ceil(low / step) * step - low
        else:
            bias = 0
        return val - bias

    def map_tuning_config_to_original(self):
        """
        The function that maps the tuning config and genetic algorithm parameters to the original config for evaluation
        """
        paths = get_paths(self.original_config)
        mapping = {}
        for key, value in self.tuning_config["attributes"].items():
            for index, path in enumerate(paths):
                if key == path[-1]:
                    mapping[key] = paths[index]
                    continue
            if key not in mapping:
                raise ValueError("Tuning config contained a attribute parameter not in the original config: {}".format(key))
        return mapping

    def map_tuning_config_back(self, chromosome: typing.List) -> typing.Container:
        """
        Maps the list of chromosomes back into a dict with the same structure as the original config file
        :param chromosome: a list of values for that individual in the population
        :return a dict containing the individuals values, but in the original config file form if using a config file, else a list of parameters
        """
        if self.using_config:
            new_config = copy.deepcopy(self.original_config)
            for index, name in enumerate(self.order_of_keys):
                current_value = chromosome[index]
                path_to_original = self.map_config[name]
                set_by_path(new_config, path_to_original, current_value, is_bool=name in self.bool_values,
                            categorical_value=self.categorical_values[name][current_value] if name in self.categorical_values else None)
            return new_config
        else:
            return chromosome

    def create_dataset_from_results(self, params: typing.List[typing.List], scores: typing.List[float], output_to_file: bool = False, 
                                    verbose: bool = False, prefix="bayes"):
        """
        This function is abstracted away to gather the info needed to write the results to file.  It will gather the correct names, scores and 
        deal with outputting or printing them depending on the config
        :param params: the parameters for that hyperparameter run
        :param scores: the scores for each parameter run
        :param output_to_file: a boolean indicating whether or not to write to file
        :param verbose: a boolean indicating whether or not to print the results
        :param prefix: the prefix for the file to be written
        """
        assert len(params) == len(scores), "scores and params were different lengths! params: {}, scores: {}".format(len(params), len(scores))
        full_named_items = []
        for ind_index, ind_params in enumerate(params):
            named_items = {}
            # get the named parameters
            for index, item in enumerate(ind_params):
                name = self.order_of_keys[index]
                named_items[name] =  item
            # flip the sign if we need to
            named_items["score"] = scores[ind_index]
            full_named_items.append(named_items)
        # write out to file
        results = pd.DataFrame(full_named_items)
        if output_to_file:
            output_path = os.path.join(self.output_dir, '{}.csv'.format(prefix))
            results.to_csv(output_path)
        if verbose:
            print(results)