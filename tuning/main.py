from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import argparse
import typing
import random
import collections
import json
import numpy as np
import math
from functools import reduce
import operator
import copy
import time
import pickle 
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class TuningDeap:
    """
    A class to genetically tune algorithms. Can handle a config file. Wraps the `deap` package.
    """

    def __init__(self, evaluate_outside_function: typing.Callable, tuning_config: dict, original_config: dict = None):
        """
        Sets up the class, creates the config mapper if needed, the evaluate function, and the genetic algorithm parameters
        :param evaluate_outside_function: the function to evaluate for the outside caller
        :param tuning_config: the config for the genetic algorithm
        :param original_config: the original_configuration file, if needed, so that we can evaluate
        """
        self.tuning_config: dict = tuning_config
        self.using_config = False
        # define output parameters if given
        self.output_path = None if "output_path" not in tuning_config else tuning_config["output_path"]
        self.output = False if self.output_path is None else True

        self.minimize = True if "minimize" not in tuning_config else tuning_config["minimize"]

        if self.output:
            logger.setLevel(logging.INFO)
        logger.info("Output is set to", self.output)

        if original_config is not None:
            self.original_config: dict = original_config
            self.using_config = True
            self.map_config: dict = self.map_tuning_config_to_original()

        self.validate_config()
        self.toolbox = base.Toolbox()
        self.evaluate_outside_function = evaluate_outside_function
        self.evaluate_function: typing.Callable = self.create_evaluate_func()
        self._instatiate_attributes()

        logger.info("Finished initializing.")

    def create_evaluate_func(self) -> typing.Callable:
        """
        The function to create the function used by the genetic algorithm. 
        We have to wrap the evaluation function with another function to change the chromosomes into a dict like the `original_config` file.
        """
        def evaluate_function(values: typing.List) -> typing.List[float]:
            """
            Simply map the chromosome values into a dict like the original config file and return that to the evaluate function
            :param values: the "chromosone" or parameter for this "individual"
            :return a list where each index corresponds to a parameter in the chromosome
            """
            config, n_values = self.map_tuning_config_back(values)
            return self.evaluate_outside_function(config, n_values)
        return evaluate_function


    def run_evolutionary(self) -> (typing.List, float):
        """
        The main function to run the evoluationary algorithm
        """
        population = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        if self.output:
            halloffame = tools.HallOfFame(maxsize=self.population_size)
        topone = tools.HallOfFame(maxsize=1)

        for gen in range(0, self.n_generations):
            print("On generation {}".format(gen))
            population = algorithms.varAnd(population, self.toolbox, cxpb=0.5, mutpb=0.2)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            if self.output:
                halloffame.update(population)
                # it's a list of lists
                individuals = halloffame.items
                # get the names of the parameters for ease of reading
                full_named_items = []
                for list_num, ind_params in enumerate(individuals):
                    named_items = {}
                    for index, item in enumerate(ind_params):
                        name = self.order_of_keys[index]
                        named_items[name] =  item
                    named_items["score"] = halloffame.keys[list_num].wvalues[0] * self.optimize
                    full_named_items.append(named_items)
                # write out to file 
                results = pd.DataFrame(full_named_items)
                results.to_csv("{}/generation-{}.csv".format(self.output_path, gen))
                halloffame.clear()

            topone.update(population)
            record = stats.compile(population)
            population = self.toolbox.select(population, k=len(population))

        return topone.items[0], topone.keys[0].wvalues[0]

    def validate_config(self):
        """
        Used to validate essential config properties
        """
        assert "population_size" in self.tuning_config, "Tuning config did not have population size!"
        assert "n_generations" in self.tuning_config, "Tuning config did not have n_generations!"

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

    def getConfigOr(self, key: str, alternate: typing.Callable) -> typing.Callable:
        """
        Used to override genetic algorithm parameters
        :param key: the key to use to lookup the function
        :param alternate: the default function to use if the key does not exist
        :return the function to be used
        """
        if key in self.tuning_config:
            # do something 
            # TODO: make this dynamicly change based on input
            pass
        return alternate

    def get_paths(self, source: dict) -> typing.List[typing.List[str]]:
        """
        Returns all paths from a dictionary object to all different keys
        :param source: the dict object to find paths in
        :return a list of paths, where each path is a list of keys
        """
        paths = []
        if isinstance(source, collections.MutableMapping):  # found a dict-like structure...
            for k, v in source.items():  # iterate over it; Python 2.x: source.iteritems()
                paths.append([k])  # add the current child path
                paths += [[k] + x for x in self.get_paths(v)]  # get sub-paths, extend with the current
        # else, check if a list-like structure, remove if you don't want list paths included
        elif isinstance(source, collections.Sequence) and not isinstance(source, str):
            #                          Python 2.x: use basestring instead of str ^
            for i, v in enumerate(source):
                paths.append([i])
                paths += [[i] + x for x in self.get_paths(v)]  # get sub-paths, extend with the current
        return paths


    def map_tuning_config_to_original(self):
        """
        The function that maps the tuning config and genetic algorithm parameters to the original config for evaluation
        """
        paths = self.get_paths(self.original_config)
        mapping = {}
        for key, value in self.tuning_config["attributes"].items():
            for index, path in enumerate(paths):
                if key == path[-1]:
                    mapping[key] = paths[index]
                    continue
            if key not in mapping:
                raise ValueError("Tuning config contained a attribute parameter not in the original config: {}".format(key))
        return mapping

    def map_tuning_config_back(self, chromosome: typing.List) -> (dict, int):
        """
        Maps the list of chromosomes back into a dict with the same structure as the original config file
        :param chromosome: a list of values for that individual in the population
        :return a dict containing the individuals values, but in the original config file form
        """
        if self.using_config:
            new_config = copy.deepcopy(self.original_config)
            for index, name in enumerate(self.order_of_keys):
                current_value = chromosome[index]
                path_to_original = self.map_config[name]
                self.set_by_path(new_config, path_to_original, current_value, is_bool=name in self.bool_values)
            return new_config, len(chromosome)
        else:
            return chromosome, len(chromosome)

    def get_by_path(self, root: dict, items: typing.List[str]):
        """
        Access a nested object in root by item sequence.
        :param root: the dictionary to access
        :param items: the list of strings containing the path
        :return the value of the path
        """
        return reduce(operator.getitem, items, root)

    def set_by_path(self, root: dict, items: typing.List[str], value, is_bool: bool = False):
        """
        Set a value in a nested object in root by item sequence.
        :param root: the dictionary to access
        :param items: the list of strings containing the path
        :param value: the value to set that path to
        """
        if is_bool:
            self.get_by_path(root, items[:-1])[items[-1]] = bool(value)
        else:
            self.get_by_path(root, items[:-1])[items[-1]] = value

    def _instatiate_attributes(self):
        """
        This function sets the necessary class values and creates the layout for the genetic algorithm
        """
        self.population_size: int = self.tuning_config["population_size"]
        self.n_generations: int = self.tuning_config["n_generations"]

        # set up the general genetic algorithm functions
        self.toolbox.register("evaluate", self.evaluate_function)
        self.toolbox.register("mate", self.getConfigOr("mate", tools.cxTwoPoint))
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select", self.getConfigOr("select", tools.selTournament), tournsize=3)
        # 1.0 means that we are maximizing the function
        self.optimize = -1 if self.minimize else 1
        creator.create("FitnessMax", base.Fitness, weights=(self.optimize,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        chromosome = []
        self.bool_values = []
        self.order_of_keys = []
        # for each attribute given to model, set up the required limits and parameters for the genetic algorithm
        for attribute_name, attribute in self.tuning_config["attributes"].items():
            self.order_of_keys.append(attribute_name)
            assert len(attribute.keys()) == 1
            param_type = str(list(attribute.keys())[0])
            attr_values = list(attribute.values())[0]
            if param_type == "int":
                self.toolbox.register("attr_int", random.randrange, attr_values[0], attr_values[1], attr_values[2] if len(attr_values) == 3 else 1)
                chromosome.append(self.toolbox.attr_int)
            if param_type == "bool":
                self.toolbox.register("attr_bool", random.randint, 0, 1)
                chromosome.append(self.toolbox.attr_bool)
                self.bool_values.append(attribute_name)
            if param_type == "float":
                self.toolbox.register("attr_float", self.rand_with_step, attr_values[0], attr_values[1], attr_values[2])
                chromosome.append(self.toolbox.attr_float)

        assert len(chromosome) != 0, "No values were added for the genetic algorithm"
        # finalize setting up the algorithm by specifying the chromosones, individual makeup, and the population
        self.toolbox.register("individual", tools.initCycle, creator.Individual, chromosome, n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

