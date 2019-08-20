import unittest
from tuningdeap import TuningDeap
import json
import os
import logging

logging.basicConfig(level=logging.INFO)

class TestMapConfigBack(unittest.TestCase):

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

    def test_gather_from_map_valid(self):
        with open(os.path.join(os.getcwd(), "test", "real_test_config.json"), "r") as file:
            self.model_config = json.load(file)
        def eval(values):
            return tuple(values[0],)
        tune = TuningDeap(eval, self.tuning_config, self.model_config, minimize=True)
        config = tune.map_tuning_config_back([1, 0, 1, 1])
        assert config.keys() == self.model_config.keys(), "keys were not mapped back correctly"
        assert config["name4"] == "b", "did not map back the categorical value"

    def test_gather_correct_mapping(self):
        with open(os.path.join(os.getcwd(), "test", "real_test_config.json"), "r") as file:
            self.model_config = json.load(file)
        def eval(values):
            return tuple(values[0],)
        tune = TuningDeap(eval, self.tuning_config, self.model_config, minimize=True)
        config = tune.map_tuning_config_to_original()
        for key, value in config.items():
            assert key == value[0], "config mapping did not map correctly. Expected {} but got {}".format(key, value[0])
