import unittest
from test.dummy_model import DummyModel
from tuningdeap import TuningDeap
import json
import os
import logging

logging.basicConfig(level=logging.INFO)

class TestStartingPopulationInitialization(unittest.TestCase):
    def setUp(self):
        cwd = os.getcwd()
        with open(os.path.join(cwd, "test", "tuning_config_test.json"), "r") as file:
            self.tuning_config = json.load(file)
        with open(os.path.join(cwd, "test", "real_test_config.json"), "r") as file:
            self.model_config = json.load(file)

        # an example of a randomly generated population for the tuning_config
        self.random_generation = [
            [0.5, 1, 2, 1],
            [0.30000000000000004, 1, 3, 1],
            [0.4, 1, 1, 1], [0.2, 0, 3, 1],
            [0.4, 0, 1, 0],
            [0.1, 1, 3, 0],
            [0.1, 1, 3, 1],
            [0.0, 0, 2, 2],
            [0.30000000000000004, 1, 2, 2],
            [0.4, 1, 1, 1]
        ]
        self.init_population_path = os.path.join(cwd, "test", "previous_population.csv")

    def test_init_population_integration(self):
        def eval_function(real_config_updated):
            return DummyModel().predict(real_config_updated)
        tune = TuningDeap(eval_function, self.tuning_config, self.model_config, init_population_path=self.init_population_path)
        # change the generation number so we get the initial population
        warm_started_population = tune.create_population_from_csv(self.random_generation)
        assert warm_started_population != self.random_generation, "did not generate a new population from the csv file"
