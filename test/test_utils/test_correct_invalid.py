import unittest
from tuningdeap import TuningDeap, TuningBayes
import logging

logging.basicConfig(level=logging.INFO)

class TestEvaluateError(unittest.TestCase):

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

    def test_minimize_error(self):
        def eval_function(chromosomes):
            raise Exception("test failure")
        tune = TuningDeap(eval_function, self.tuning_config, minimize=True)
        best_config, best_score = tune.run()
        assert best_score == float("inf"), "wrong default value was returned. Should be inf was {}".format(best_score)

    def test_maximize_error(self):
        def eval_function(chromosomes):
            raise Exception("test failure")
        tune = TuningDeap(eval_function, self.tuning_config, minimize=False)
        best_config, best_score = tune.run()
        assert best_score == float("-inf"), "wrong default value was returned. Should be -inf was {}".format(best_score)

    def test_bayes_no_maximize(self):
        try:
            def eval_function(chromosomes):
                raise 1
            tune = TuningBayes(eval_function, self.tuning_config, minimize=False, n_calls=2)
            best_config, best_score = tune.run()
            self.fail("Should not have allowed `minimize=False` to work")
        except Exception:
            pass