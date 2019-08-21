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


class TuningDeap:
    """
    A class to genetically tune algorithms. Can handle a config file. Wraps the `deap` package.
    """

    def __init__(
        self, evaluate_outside_function: typing.Callable, tuning_config: dict, original_config: dict = None,
        output_dir: str = None, verbose: bool = False, minimize: bool = True, timeout: float = float("inf")
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
        """
        self.tuning_config: dict = tuning_config
        self.using_config = False

        self.output_dir = output_dir
        if self.output_dir is not None and not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        self.minimize = minimize
        self.timeout = timeout

        self.verbose = verbose
        if self.verbose:
            logger.setLevel(logging.INFO)
            logger.info("verbose is set to: {}".format(self.verbose))

        if original_config is not None:
            self.original_config: dict = original_config
            self.using_config = True
            self.map_config: dict = self.map_tuning_config_to_original()

        self.validate_config()
        self.toolbox = base.Toolbox()
        self.evaluate_outside_function = evaluate_outside_function
        self.evaluate_function: typing.Callable = self.create_evaluate_func()
        self._instatiate_attributes()

        if self.verbose:
            logger.info("Finished initializing.")

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


    def run_evolutionary(self) -> (typing.List, float):
        """
        The main function to run the evoluationary algorithm
        """
        population = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)

        if self.output_dir is not None:
            halloffame = tools.HallOfFame(maxsize=self.population_size)
        topone = tools.HallOfFame(maxsize=1)

        ending_time = time.time() + self.timeout
        self.gen = 0
        while self.gen < self.n_generations and time.time() < ending_time:
            self.gen += 1
            if self.verbose:
                logger.info("On generation {}".format(self.gen))

            # create and validate new generation
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
                output_path = os.path.join(self.output_dir, 'generation-{}.csv'.format(self.gen))
                results.to_csv(output_path)
                halloffame.clear()

            if len(population) != 0:
                topone.update(population)
            population = self.toolbox.select(population, k=len(population))

        if len(topone) == 0:
            return [], float("inf") if self.minimize else float("-inf")

        return topone.items[0], float(topone.keys[0].wvalues[0] * self.optimize)

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
