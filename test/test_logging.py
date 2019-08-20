import unittest
from tuningdeap import TuningDeap
import json
import os
import logging
import numpy as np


class TestLogging(unittest.TestCase):

    def setUp(self):
        cwd = os.getcwd()
        self.tuning_config = {
            "population_size": 1,
            "n_generations": 1,
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
                },
                "name4": {
                    "type": "categorical",
                    "values": ["a", "b", "c"]
                }
            }
        }

    @staticmethod
    def _eval_function(chromosomes):
        return tuple(np.sum(np.sqrt(chromosomes[:len(chromosomes) - 1])), )

    def test_logs(self):
        with self.assertLogs('tuningdeap.main', level='INFO') as cm:
            # tests that logging works
            tune = TuningDeap(self._eval_function, self.tuning_config, verbose=True)
            best_config, best_score = tune.run_evolutionary()

        assert len(cm.output) > 0, "Should have output more than zero logs: instead got zero"
        
    
    def test_no_logs(self):
        try:
            with self.assertLogs('tuningdeap.main', level='INFO') as cm:
                # tests that logging works
                tune = TuningDeap(self._eval_function, self.tuning_config, verbose=False)
                best_config, best_score = tune.run_evolutionary()

        except Exception:
            pass

    def test_output_dir(self):
        tune = TuningDeap(self._eval_function, self.tuning_config, output_dir='./tmp', verbose=False)
        best_config, best_score = tune.run_evolutionary()
        self.assertTrue(os.path.isfile('./tmp/generation-0.csv'))

    
    

    

    
