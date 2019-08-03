import unittest
from test.dummy_model import DummyModel
from tuning.main import tuningDeap
import json
import os

class TestConfig(unittest.TestCase):
    def setUp(self):
        cwd = os.getcwd()
        with open(os.path.join(cwd, "test", "tuning_config_test.json"), "r") as file:
            self.tuning_config = json.load(file)
        with open(os.path.join(cwd, "test", "real_test_config.json"), "r") as file:
            self.model_config = json.load(file)

    def test_basic_config(self):
        def eval_function(real_config_updated, n_values):
            return DummyModel().predict(real_config_updated, n_values)
        tune = tuningDeap(eval_function, self.tuning_config, self.model_config)
        tune.run_evolutionary()
    

    

    