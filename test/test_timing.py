import unittest
from tuningdeap import TuningDeap, TuningBayes
import logging
import numpy as np
import time

logging.basicConfig(level=logging.INFO)


class TestTiming(unittest.TestCase):

    def setUp(self):
        self.tuning_config = {
            "population_size": 10,
            "n_generations": 10000000,
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

    def test_zero_second_one_generation_deap(self):
        def eval_function(chromosomes):
            return (1, )
        # small value to make sure it runs close to one generation only
        timeout_time = 0.001
        tune = TuningDeap(eval_function, self.tuning_config, verbose=False, timeout=timeout_time)
        start_time = time.time()
        _, _ = tune.run()
        assert abs((time.time() - start_time) - timeout_time) < 1, "timeout took longer than 1 second +- 1 second"
        assert abs(tune.gen - 1) < 3, "tuning ran {} generations when it should have run one".format(tune.gen)
    
    def test_completed_epochs_inf_time_deap(self):
        def eval_function(chromosomes):
            return (1, )
        self.tuning_config["n_generations"] = 5
        tune = TuningDeap(eval_function, self.tuning_config, verbose=False, timeout=float("inf"))
        start_time = time.time()
        _, _ = tune.run()
        assert tune.gen == 5, "tuning ran {} generations when it should have run 5".format(tune.gen)

    def test_time_longer_generations_deap(self):
        def eval_function(chromosomes):
            return (1, )
        timeout_time = 15
        tune = TuningDeap(eval_function, self.tuning_config, verbose=False, timeout=timeout_time)
        start_time = time.time()
        _, _ = tune.run()
        assert abs((time.time() - start_time) - timeout_time) < 1, "timeout took longer than 30 seconds +- 1 second"

    def test_zero_second_one_iter_bayes(self):
        def eval_function(chromosomes):
            return 1
        # small value to make sure it runs close to one iteration only
        timeout_time = 0.001
        tune = TuningBayes(eval_function, self.tuning_config, verbose=False, timeout=timeout_time, n_calls=100)
        start_time = time.time()
        _, _ = tune.run()
        assert abs((time.time() - start_time) - timeout_time) < 2, "timeout took longer than 1 second +- 1 second: {}s".format(abs((time.time() - start_time) - timeout_time))
        assert abs(len(tune.x_iters) - 1) < 3, "tuning ran {} generations when it should have run one".format(tune.gen)
    
    def test_completed_epochs_inf_time_bayes(self):
        def eval_function(chromosomes):
            return 1
        tune = TuningBayes(eval_function, self.tuning_config, verbose=False, timeout=float("inf"), n_calls=5)
        _, _ = tune.run()
        assert len(tune.x_iters) == 5, "tuning ran {} generations when it should have run 5".format(tune.gen)
