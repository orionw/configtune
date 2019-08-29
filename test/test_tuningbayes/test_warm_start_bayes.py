import unittest
from test.dummy_model import DummyModel
from configtune import TuningBayes
import json
import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)

class TestWarmStartInitialization(unittest.TestCase):
    def setUp(self):
        cwd = os.getcwd()
        with open(os.path.join(cwd, "test", "tuning_config_test.json"), "r") as file:
            self.tuning_config = json.load(file)
        with open(os.path.join(cwd, "test", "real_test_config.json"), "r") as file:
            self.model_config = json.load(file)

        self.warm_start_path = os.path.join(cwd, "test", "previous_population.csv")
        self.warm_start = pd.read_csv(self.warm_start_path, header=0, index_col=0)

    def test_init_runs_integration(self):
        def eval_function(real_config_updated):
            raise Exception("Should never call this")
        self.n_calls = 0
        tune = TuningBayes(eval_function, self.tuning_config, self.model_config, warm_start_path=self.warm_start_path, n_calls=self.n_calls)
        scores, configs = tune.init_warm_start()
        assert len(configs) == self.n_calls + self.warm_start.shape[0], "did not generate 1 + previously generated: got {} but expected {}".format(len(configs), self.n_calls + self.warm_start.shape[0])
