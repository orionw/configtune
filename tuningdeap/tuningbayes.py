import pandas as pd
import numpy as np
import skopt


from skopt.space import Real, Integer
from skopt.utils import use_named_args

from tuningdeap.base_tuner import TuningBase

class TuningBayes(TuningBase):
    def __init__(self):
        pass