import unittest
from tuningdeap import TuningDeap
import logging

logging.basicConfig(level=logging.INFO)


class TestTiming(unittest.TestCase):

    def setUp(self):
        self.tuning_config = {
            "population_size": 1,
            "n_generations": 1,
            "attributes": {
                "name1": {
                    "type": "float",
                    "min": 0,
                    "max": 1,
                    "step": 0.1
                },
                "name2": {
                    "type": "bool"
                },
                "name3": {
                    "type": "int",
                    "min": 1,
                    "max": 5,
                    "step": 1
                },
                "name4": {
                    "type": "categorical",
                    "values": ["a", "b", "c"]
                }
            }
        }

    def test_population_and_generations_from_config(self):
        def eval_function(chromosomes):
            return tuple(1, )
        tune = TuningDeap(eval_function, self.tuning_config, verbose=False)
        assert tune.n_generations == self.tuning_config["n_generations"], "did not read n_generations from the config file like it should have"
        assert tune.population_size == self.tuning_config["population_size"], "did not read n_generations from the config file like it should have"
    
    def test_population_and_generations_from_constructor(self):
        def eval_function(chromosomes):
            return tuple(1, )
        n_generations = 10
        population_size = 10
        tune = TuningDeap(eval_function, self.tuning_config, verbose=False, population_size=population_size, n_generations=n_generations)
        assert tune.n_generations == n_generations, "did not read n_generations from the constructor like it should have"
        assert tune.population_size == population_size, "did not read n_generations from the constructor like it should have"
    