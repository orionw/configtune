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
                                    "output": False,
                                    "minimize": True,
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
                                        }
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
    
    

    

    
