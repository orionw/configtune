import unittest
from tuningdeap import TuningDeap
import json
import os
import logging

logging.basicConfig(level=logging.INFO)

class TestEvaluateError(unittest.TestCase):
    def setUp(self):
        cwd = os.getcwd()
        self.tuning_config = {
                                "population_size": 1,
                                "n_generations": 1,
                                "minimize": True,
                                "attributes": {
                                    "name1": {"float": [0, 1, 0.1]},
                                    "name2": {"bool": []},
                                    "name3": {"int": [1, 5, 1]}
                                }
                            }

    def test_minimize_error(self):
        def eval_function(chromosomes):
            raise Exception("test failure")
        tune = TuningDeap(eval_function, self.tuning_config)
        best_config, best_score = tune.run_evolutionary()
        assert best_score == float("inf"), "wrong default value was returned. Should be inf was {}".format(best_score)
    
    def test_maximize_error(self):
        def eval_function(chromosomes):
            raise Exception("test failure")
        self.tuning_config["minimize"] = False
        tune = TuningDeap(eval_function, self.tuning_config)
        best_config, best_score = tune.run_evolutionary()
        assert best_score == float("-inf"), "wrong default value was returned. Should be -inf was {}".format(best_score)
    
    

    

    
