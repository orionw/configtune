import unittest
from test.dummy_model import DummyModel
from tuning.main import TuningDeap
import json
import os

class TestConfig(unittest.TestCase):
    def setUp(self):
        cwd = os.getcwd()
        with open(os.path.join(cwd, "test", "tuning_config_test.json"), "r") as file:
            self.tuning_config = json.load(file)

    def test_basic_config(self):
        def eval_function(chromosomes, n_values):
            return DummyModel().predict_no_config(chromosomes, n_values)
        tune = TuningDeap(eval_function, self.tuning_config)
        best_config, best_score = tune.run_evolutionary()
        assert type(best_score) == float, "wrong type was returned, expected float was {}".format(type(best_score))
    

    

    