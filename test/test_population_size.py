import unittest
from tuningdeap import TuningDeap
import json
import logging

logging.basicConfig(level=logging.INFO)

class TestPopulationSize(unittest.TestCase):

    def setUp(self):
        self.tuning_config = {
            "population_size": 100000,
            "n_generations": 10,
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

    def test_large_population_size(self):
        def eval_function(chromosomes):
            return tuple(1, )
        # run and see if it hits the assertion for too small of a population
        tune = TuningDeap(eval_function, self.tuning_config)
        _, _ = tune.run_evolutionary()
