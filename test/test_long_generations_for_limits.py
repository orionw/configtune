# import unittest
# from tuningdeap import TuningDeap
# import json
# import os
# import logging

# logging.basicConfig(level=logging.INFO)

# class TestCheckLimits(unittest.TestCase):
#     def setUp(self):
#         cwd = os.getcwd()
#         self.tuning_config = {
#                                 "population_size": 100,
#                                 "n_generations": 1000,
#                                 "minimize": True,
#                                 "output": True,
#                                 "attributes": {
#                                     "name1": {"float": [0, 1, 0.1]},
#                                     "name2": {"bool": []},
#                                     "name3": {"int": [1, 5, 1]}
#                                 }
#                             }

#     def test_long_running(self):
#         # this test does not implicitly assert anything but those should be caught in the mutation
#         def eval_function(chromosomes):
#             return tuple(np.sum(np.sqrt(chromosomes)), )
#         tune = TuningDeap(eval_function, self.tuning_config)
#         best_config, best_score = tune.run_evolutionary()

#     def test_enforce_limits_no_population_left(self):
#         def eval_function(chromosomes):
#             # not used
#             return 1
#         tune = TuningDeap(eval_function, self.tuning_config)
#         init_population = [[0.1, 0.5, 1], [0.1, 1, 6], [1.1, 0, 2]]
#         final_population = tune.enforce_limits(init_population)
#         print(final_population)
#         assert len(final_population) == 0, "should have elimated all inividuals for being unfit, instead had {}".format(len(final_population))

#     def test_enforce_limits_all_left(self):
#         def eval_function(chromosomes):
#             # not used
#             return 1
#         tune = TuningDeap(eval_function, self.tuning_config)
#         init_population = [[0.1, 0, 1], [0.1, 1, 5], [1.0, 0, 1]]
#         final_population = tune.enforce_limits(init_population)
#         print(final_population)
#         assert len(final_population) == len(init_population), "should have all inividuals left: {}, instead had {}".format(len(init_population), 
#                                                                 len(final_population))                                 
#     def test_enforce_limits_some(self):
#         def eval_function(chromosomes):
#             # not used
#             return 1
#         tune = TuningDeap(eval_function, self.tuning_config)
#         init_population = [[0.0, 0, 1], [0.1, 1, 5], [1.0, 0, 1]]
#         final_population = tune.enforce_limits(init_population)
#         print(final_population)
#         assert len(final_population) == len(init_population), "should have {} inividuals left: instead had {}".format(len(init_population) - 1, 
#                                                                     len(final_population))

        
    

    

    
