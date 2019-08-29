[![Build Status](https://travis-ci.com/orionw/tuningDEAP.svg?branch=master)](https://travis-ci.com/orionw/tuningDEAP)
[![PyPI version](https://badge.fury.io/py/configtune.svg)](https://badge.fury.io/py/configtune)
[![codecov](https://codecov.io/gh/orionw/tuningDEAP/branch/master/graph/badge.svg)](https://codecov.io/gh/orionw/tuningDEAP)

# configtune
A package for tuning machine learning models (bayesian or genetic tuning), with or without a config file.

## How to Use:
0. Create your model and config files (if desired)
1. Create your tuning config in the format as follows (json):
```
{
    "attributes": {
        "generic_param_example: {
            "type": <"int"/"float"/"bool">,
            "min": <min_value if int or float>,
            "max": <max_value if int or float>,
            <this is an optional param for configtune but NOT tuningbayes, default=1: "step": <step size value>,
            <this is an optional param to enforce step limits (for configtune but NOT tuningbayes): "strict": <True/False>>
        },
        "int_you_want_to_tune_example": {
            "type": "int",
            "min": 1,
            "max": 10,
            "step": 2
        },
        "float_you_want_to_tune_example": {
            "type": "int",
            "min": 0,
            "max": 1,
            "step": 0.1
        },
        "bool_you_want_to_tune_example": {
            "type": "bool"
        },
        "categorical_values_you_want_to_tune_example": {
            "type": "categorical",
            "values": ["a", "b", "c"]
    }
}
```
Boolean values don't need any bounds.  The parameter names should match those found in your model config file, if you have one.  Categorical values will be randomly selected for initialization.

2. Create your evaluation function.  This function needs to take in a config file or a list of values being tuned if you're not using a config.  It should output a scalar value.

Example overall usage of `TuningDeap`:
```
from configtune import TuningDeap

def eval_function(config_file):
    return your_eval_function(config_file)

tune = TuningDeap(eval_function, tuning_config, model_config, n_generation=5, population_size=10, 
                  minimize=True, output_dir="/tmp", verbose=False)
best_config, best_score = tune.run()
```

Example overall usage of `TuningBayes`:
```
from configtune import TuningBayes

def eval_function(config_file):
    return your_eval_function(config_file)

tune = TuningBayes(eval_function, tuning_config, model_config, n_calls=10, n_random_starts=2, 
                   output_dir="/tmp", verbose=True)
best_config, best_score = tune.run()
```
