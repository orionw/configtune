import unittest
from test.dummy_model import DummyModel
from configtune import TuningDeap, TuningBayes
import json
import os
import logging
import numpy as np

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
        best_config, best_score = tune.run()
        assert type(best_score) == float, "wrong type was returned, expected float was {}".format(type(best_score))

    def test_no_config_basic(self):
        def eval_function(chromosomes):
            return DummyModel().predict_no_config(chromosomes)
        tune = TuningBayes(eval_function, self.tuning_config, n_calls=2)
        best_config, best_score = tune.run()
        assert type(best_score) == np.float64, "wrong type was returned, expected float was {}".format(type(best_score))
