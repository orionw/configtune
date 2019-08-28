import unittest
from test.dummy_model import DummyModel
from tuningdeap import TuningDeap
import json
import os
import logging

logging.basicConfig(level=logging.INFO)

class TestRepoducibility(unittest.TestCase):
    def setUp(self):
        cwd = os.getcwd()
        with open(os.path.join(cwd, "test", "tuning_config_test.json"), "r") as file:
            self.tuning_config = json.load(file)
        with open(os.path.join(cwd, "test", "real_test_config.json"), "r") as file:
            self.model_config = json.load(file)


    # def test_same_output(self):
    #     # TODO: fix this test
    #     def eval_function(real_config_updated):
    #         return DummyModel().predict(real_config_updated)
    #     tune1 = TuningDeap(eval_function, self.tuning_config, self.model_config, minimize=False, random_seed=0)
    #     best_config1, best_score1 = tune1.run()

    #     tune2 = TuningDeap(eval_function, self.tuning_config, self.model_config, minimize=False, random_seed=0)
    #     best_config2, best_score2 = tune2.run()
    #     assert best_score1 == best_score2, "did not reproduce the same score: {} vs {}".format(best_score1, best_score2)
    #     assert best_config1 == best_config2, "did not reproduce the same best config: {} vs {}".format(best_config1, best_config2)
