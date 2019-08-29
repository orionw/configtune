import unittest
from tuningdeap import TuningDeap, TuningBayes
import numpy as np
import os


class TestLogging(unittest.TestCase):

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

    @staticmethod
    def _eval_function(chromosomes):
        return np.sum(np.sqrt(chromosomes[:len(chromosomes) - 1]))

    def test_logs_deap(self):
        with self.assertLogs('tuningdeap.base_tuner', level='INFO') as cm:
            # tests that logging works
            tune = TuningDeap(self._eval_function, self.tuning_config, verbose=True, minimize=True)
            best_config, best_score = tune.run()

        assert len(cm.output) > 0, "Should have output more than zero logs: instead got zero"

    def test_logs_bayes(self):
        with self.assertLogs('tuningdeap.base_tuner', level='INFO') as cm:
            # tests that logging works
            tune = TuningBayes(self._eval_function, self.tuning_config, verbose=True, n_calls=2)
            best_config, best_score = tune.run()

        assert len(cm.output) > 0, "Should have output more than zero logs: instead got zero"

    def test_no_logs_deap(self):
        try:
            with self.assertLogs('tuningdeap.base_tuner', level='INFO') as cm:
                # tests that logging works
                tune = TuningDeap(self._eval_function, self.tuning_config, verbose=False, minimize=True)
                best_config, best_score = tune.run()

        except Exception:
            pass

    def test_no_logs_bayes(self):
        try:
            with self.assertLogs('tuningdeap.base_tuner', level='INFO') as cm:
                # tests that logging works
                tune = TuningBayes(self._eval_function, self.tuning_config, verbose=True, n_calls=2)
            best_config, best_score = tune.run()

        except Exception:
            pass

    def test_output_dir_deap(self):
        tune = TuningDeap(self._eval_function, self.tuning_config, output_dir='./tmp', verbose=False, minimize=True)
        best_config, best_score = tune.run()
        self.assertTrue(os.path.isfile('./tmp/generation-1.csv'))

    def test_output_dir_bayes(self):
        tune = TuningBayes(self._eval_function, self.tuning_config, output_dir='./tmp', verbose=False, minimize=True, n_calls=2)
        best_config, best_score = tune.run()
        self.assertTrue(os.path.isfile('./tmp/bayes.csv'))



    
    

    

    
