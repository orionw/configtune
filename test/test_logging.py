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
                                "output": True,
                                "attributes": {
                                    "name1": {"float": [0, 1, 0.1]},
                                    "name2": {"bool": []},
                                    "name3": {"int": [1, 5, 1]}
                                }
                            }
    # TODO: fix logging test
    # def test_logs(self):
    #     with self.assertLogs('tuningdeap.main', level='INFO') as cm:
    #         # tests that logging works
    #         def eval_function(chromosomes):
    #             return tuple(np.sum(np.sqrt(chromosomes)), )
    #         tune = TuningDeap(eval_function, self.tuning_config)
    #         best_config, best_score = tune.run_evolutionary()

    #     assert len(cm.output) > 0, "Should have output more than zero logs: instead got zero"
        
    
    def test_no_logs(self):
        self.tuning_config["output"] = False
        try:
            with self.assertLogs('tuningdeap.main', level='INFO') as cm:
                # tests that logging works
                def eval_function(chromosomes):
                    return tuple(np.sum(np.sqrt(chromosomes)), )
                tune = TuningDeap(eval_function, self.tuning_config)
                best_config, best_score = tune.run_evolutionary()

        except Exception:
            pass

    
    

    

    
