import unittest
from test.dummy_model import DummyModel
from tuningdeap import TuningDeap
import json
import os
import logging

logging.basicConfig(level=logging.INFO)

class TestNoConfig(unittest.TestCase):
    def setUp(self):
        cwd = os.getcwd()
        with open(os.path.join(cwd, "test", "tuning_config_test.json"), "r") as file:
            self.tuning_config = json.load(file)

    def test_no_config_basic(self):
        def eval_function(chromosomes):
            return DummyModel().predict_no_config(chromosomes)
        tune = TuningDeap(eval_function, self.tuning_config)
        best_config, best_score = tune.run_evolutionary()
        assert type(best_score) == float, "wrong type was returned, expected float was {}".format(type(best_score))
    

    

    
