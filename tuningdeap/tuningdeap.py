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
import warnings

from tuningdeap.base_tuner import TuningBase
from tuningdeap.config_mapping import set_by_path, get_paths

logger = logging.getLogger(__name__)


class TuningDeap(TuningBase):
    """
    A class to genetically tune algorithms. Can handle a config file. Wraps the `deap` package.
    """

    def __init__(
        self, evaluate_outside_function: typing.Callable, tuning_config: dict, original_config: dict = None,
        output_dir: str = None, verbose: bool = False, minimize: bool = True, timeout: float = float("inf"),
        n_generations: int = None, population_size: int = None, random_seed: int = None, warm_start_path: str = None
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
        :param random_seed: set the random seed for deap
        :param warm_start_path: the path to a csv to warm start with
        """
        super().__init__(evaluate_outside_function, tuning_config, original_config, output_dir=output_dir, verbose=verbose, minimize=minimize, timeout=timeout, 
                        random_seed=random_seed, warm_start_path=warm_start_path)
        self.population_size = population_size
        self.n_generations = n_generations

        # override the base class and add the tuple that deap needs
        self.evaluate_function = self.create_evaluate_func_deap()

        self.validate_config()
        self.toolbox = base.Toolbox()
        self._instantiate_attributes()

        if self.verbose:
            logger.info("Finished initializing.")

    def create_evaluate_func_deap(self) -> typing.Callable:
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
                return (self.evaluate_outside_function(config), )
            except Exception as e:
                warnings.warn("there was an error while evaluating the function: {}".format(e))
                return tuple((float('inf'), )) if self.minimize else tuple((float("-inf"), ))
        return evaluate_function

    def run(self) -> (typing.List, float):
        """
        The main function to run the evoluationary algorithm
        """
        population = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)

        if self.warm_start_path is not None:
            population = self.create_population_from_csv(population)

        if self.output_dir is not None:
            halloffame = tools.HallOfFame(maxsize=self.population_size)
        topone = tools.HallOfFame(maxsize=1)

        ending_time = time.time() + self.timeout
        self.gen = 0
        while self.gen < self.n_generations and time.time() < ending_time:
            self.gen += 1
            if self.verbose:
                print("On generation {}".format(self.gen))

            # create and validate new generation by geneticism
            final_population = self.get_new_generation(population)
            
            # Evaluate the individuals with an invalid fitness
            population = final_population
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            if self.output_dir is not None:
                halloffame.update(population)
                # it's a list of lists
                individuals = halloffame.items
                # reverse halloffame items since they did it reverse
                individuals.reverse()
                scores = [item.wvalues[0] * self.optimize for item in halloffame.keys]
                if self.verbose:
                    print("The results from generation {} are:".format(self.gen))
                self.create_dataset_from_results(individuals, scores, self.output_dir is not None, self.verbose, prefix="generation-{}".format(self.gen))
                halloffame.clear()

            if len(population) != 0:
                topone.update(population)
            population = self.toolbox.select(population, k=len(population))

        if len(topone) == 0:
            return [], float("inf") if self.minimize else float("-inf")

        return topone.items[0], topone.keys[0].wvalues[0] * self.optimize

    def get_new_generation(self, population: list) -> list:
        """
        This function gets the new generation of the population, enforces, and verfies the limits and size
        :param population: the list of individuals, each a list
        :return the final population list of lists
        """
        # enforce our own limits on parameters regardless of mutations
        iterations = 0
        final_population = []
        while len(final_population) < self.population_size and iterations < 10:
            iterations += 1
            init_population = algorithms.varAnd(population, self.toolbox, cxpb=0.5, mutpb=0.2)
            final_population += self.enforce_limits(init_population)
            # if too big, scale it down
            final_population = final_population[:self.population_size]
        
        # verify population
        small_population = len(final_population) < self.population_size
        if iterations == 10 and small_population:
            raise EnvironmentError("bounds could not be enforced or population too small")
        assert small_population == False, "population was smaller than it should have been"
        assert len(population) == len(final_population), "populations are not the same size. Expected {} but got {}".format(len(population), len(final_population))
        return final_population

    def validate_config(self):
        """
        Used to validate essential config properties
        """
        if self.population_size is None:
            assert "population_size" in self.tuning_config, "Tuning config did not have population size!"
            self.population_size = self.tuning_config["population_size"]
        
        if self.n_generations is None:
            assert "n_generations" in self.tuning_config, "Tuning config did not have n_generations!"
            self.n_generations = self.tuning_config["n_generations"]

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

    def _instantiate_attributes(self):
        """
        This function sets the necessary class values and creates the layout for the genetic algorithm
        """
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
            # create the values in the program
            if param_type == "int":
                self.toolbox.register("attr_int", random.randrange, min_val, max_val, step_size)
                chromosome.append(self.toolbox.attr_int)
            if param_type == "bool":
                self.toolbox.register("attr_bool", random.randint, 0, 1)
                chromosome.append(self.toolbox.attr_bool)
                self.bool_values.append(attribute_name)
            if param_type == "float":
                self.toolbox.register("attr_float", self.rand_with_step, min_val, max_val, step_size)
                chromosome.append(self.toolbox.attr_float)
            if param_type == "categorical":
                self.toolbox.register("attr_int", random.randrange, 0, len(attribute_map["values"]), 1)
                chromosome.append(self.toolbox.attr_int)
                self.categorical_values[attribute_name] = attribute_map["values"]

        assert len(chromosome) != 0, "No values were added for the genetic algorithm"
        # finalize setting up the algorithm by specifying the chromosones, individual makeup, and the population
        self.toolbox.register("individual", tools.initCycle, creator.Individual, chromosome, n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

    def create_population_from_csv(self, init_population: typing.List[typing.List]):
        """
        This function is used to "warm-start" or initialize a population from previous values
        :param init_population: the deap generated list of individuals
        :return the init_population that has been populated by the csv file
        """
        previous_population = self.read_and_validate_previous(self.warm_start_path)

        # get the best given if more than population_size given
        previous_population.sort_values(by=["score"])
        starting_population = previous_population.iloc[:self.population_size, ].to_numpy().tolist()

        # fill in any missing inidivduals from the randomly created population
        if len(starting_population) < self.population_size:
            starting_population += init_population[:self.population_size - len(starting_population)]

        assert len(starting_population) == self.population_size, "could not create a population size of {}".format(self.population_size)
        return starting_population
