import unittest
from tuningdeap import TuningDeap
import json
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)

class TestPopulationSize(unittest.TestCase):

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

    def test_large_population_size(self):
        def eval_function(chromosomes):
            return (1, )
        # run and see if it hits the assertion for too small of a population
        tune = TuningDeap(eval_function, self.tuning_config, n_generations=10, population_size=10000)
        _, _ = tune.run()

    def test_population_output_mixup(self):
        print("Checking population params")
        def eval_function(chromosomes):
            return (np.sum(chromosomes[:len(chromosomes) - 1]), )

        n_generations = 10
        tune = TuningDeap(eval_function, self.tuning_config, minimize=False, output_dir="./tmp", n_generations=n_generations)
        _, _ = tune.run()
        # check the generations
        for generations in range(n_generations):
            gen = pd.read_csv("./tmp/generation-{}.csv".format(generations + 1), header=0, index_col=0)
            for index, row in gen.iterrows():
                assert eval_function(row[:len(row) - 1].to_numpy())[0] == row[-1], "did not get the score expected: output: {}, expected: {}".format(eval_function(row[:len(row) - 1].to_numpy())[0], row[-1])

