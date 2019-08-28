import unittest
import logging
from examples.tune_basic_mlp import execute_example_deap

logging.basicConfig(level=logging.INFO)

class TestFullMLP(unittest.TestCase):

    def test_example_run_full_mlp(self):
        # make sure the example runs successfully
        execute_example_deap()
