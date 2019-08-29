import unittest
from tuningdeap import TuningBayes
import json
import logging
import numpy as np
import pandas as pd
import math

logging.basicConfig(level=logging.INFO)

class TestPopulationValidityBayes(unittest.TestCase):

    def setUp(self):
        self.tuning_config = {
            "population_size": 100,
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

    def test_large_population_size_bayesian(self):
        def eval_function(chromosomes):
            return 1
        # run and see if it works.  Note that 100 is a long time - Bayes takes longer than the Deap version
        tune = TuningBayes(eval_function, self.tuning_config, n_calls=100)
        _, _ = tune.run()

    def test_population_output_correct_bayes(self):
        def eval_function(chromosomes):
            return np.sum(chromosomes[:len(chromosomes) - 1])

        n_generations = 10
        tune = TuningBayes(eval_function, self.tuning_config, output_dir="./tmp", n_calls=10)
        _, _ = tune.run()
        # check the generations
        for generations in range(n_generations):
            gen = pd.read_csv("./tmp/bayes.csv", header=0, index_col=0)
            for index, row in gen.iterrows():
                assert math.isclose(eval_function(row[:len(row) - 1].to_numpy()), row[-1]) == True, "did not get the score expected: output: {}, expected: {}".format(eval_function(row[:len(row) - 1].to_numpy()), row[-1])

