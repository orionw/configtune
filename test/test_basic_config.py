import unittest
from test.dummy_model import DummyModel
from tuningdeap import TuningDeap, TuningBayes
import json
import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

class TestConfig(unittest.TestCase):
    def setUp(self):
        cwd = os.getcwd()
        with open(os.path.join(cwd, "test", "tuning_config_test.json"), "r") as file:
            self.tuning_config = json.load(file)
        with open(os.path.join(cwd, "test", "real_test_config.json"), "r") as file:
            self.model_config = json.load(file)

    def test_basic_config_deap(self):
        def eval_function(real_config_updated):
            return DummyModel().predict(real_config_updated)
        tune = TuningDeap(eval_function, self.tuning_config, self.model_config)
        best_config, best_score = tune.run()
        assert type(best_score) in [float, int], "wrong type was returned, expected float was {}".format(type(best_score))


    def test_basic_config_bayes(self):
        def eval_function(real_config_updated):
            return DummyModel().predict(real_config_updated)
        tune = TuningBayes(eval_function, self.tuning_config, self.model_config, verbose=True, n_calls=2)
        best_config, best_score = tune.run()
        assert type(best_score) == np.float64, "wrong type was returned, expected float was {}".format(type(best_score))
