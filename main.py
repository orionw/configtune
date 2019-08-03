from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import argparse
import typing
from dummy_model import DummyModel
import random
import collections
import json
import numpy as np
import math
from functools import reduce
import operator
import copy


class tuningDeap:

    def __init__(self, evaluate_outside_function: typing.Callable, tuning_config: dict, original_config: dict = None):
        self.tuning_config: dict = tuning_config
        self.original_config: dict = original_config
        self.validate_config()
        self.toolbox = base.Toolbox()
        self.evaluate_outside_function = evaluate_outside_function
        self.map_config: dict = self.map_tuning_config_to_original()
        self.evaluate_function: typing.Callable = self.create_evaluate_func()
        self._instatiate_attributes()

    def create_evaluate_func(self):
        def evaluate_function(values):
            config, n_values = self.map_tuning_config_back(values)
            return self.evaluate_outside_function(config, n_values)
        return evaluate_function


    def run_evolutionary(self):
        """
        The main function to run the evoluationary algorithm
        """
        pop = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=self.n_generations, 
                                       stats=stats, halloffame=hof, verbose=True)

    def validate_config(self):
        """
        Used to validate essential config properties
        """
        assert "population_size" in self.tuning_config, "Tuning config did not have population size!"
        assert "n_generations" in self.tuning_config, "Tuning config did not have n_generations!"
        # assert hasattr(self.tuning_config, "population_size"), "Tuning config did not have population size!"
        # assert hasattr(self.tuning_config, "population_size"), "Tuning config did not have population size!"

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
        new_config = copy.deepcopy(self.original_config)
        for index, name in enumerate(self.order_of_keys):
            current_value = chromosome[index]
            path_to_original = self.map_config[name]
            self.set_by_path(new_config, path_to_original, current_value)
        return new_config, len(chromosome)

    def get_by_path(self, root: dict, items: typing.List[str]):
        """Access a nested object in root by item sequence."""
        return reduce(operator.getitem, items, root)

    def set_by_path(self, root: dict, items: typing.List[str], value):
        """Set a value in a nested object in root by item sequence."""
        self.get_by_path(root, items[:-1])[items[-1]] = value

    def _instatiate_attributes(self):
        self.attributes = {}

        self.population_size: int = self.tuning_config["population_size"]
        self.n_generations: int = self.tuning_config["n_generations"]

        self.toolbox.register("evaluate", self.evaluate_function)
        self.toolbox.register("mate", self.getConfigOr("mate", tools.cxTwoPoint))
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select", self.getConfigOr("select", tools.selTournament), tournsize=3)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        chromosome = []
        self.order_of_keys = []
        for attribute_name, attribute in tuning_config["attributes"].items():
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
            if param_type == "float":
                self.toolbox.register("attr_float", self.rand_with_step, attr_values[0], attr_values[1], attr_values[2])
                chromosome.append(self.toolbox.attr_float)

        assert len(chromosome) != 0, "No values were added for the genetic algorithm"
        self.toolbox.register("individual", tools.initCycle, creator.Individual, chromosome, n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)



if __name__ == "__main__":
    with open("tuning_config_test.json", "r") as file:
        tuning_config = json.load(file)
    with open("real_config.json", "r") as file:
        model_config = json.load(file)

    def eval_function(real_config_updated, n_values):
        return DummyModel().predict(real_config_updated, n_values)

    tune = tuningDeap(eval_function, tuning_config, model_config)
    best_model = tune.run_evolutionary()