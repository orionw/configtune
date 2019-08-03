[![Build Status](https://travis-ci.com/orionw/tuningDEAP.svg?branch=master)](https://travis-ci.com/orionw/tuningDEAP)
# TuningDEAP
A package for tuning machine learning models genetically, with or without a config file. 

## How to Use:
0. Create your model and config files (if desired)
1. Create your tuning config in the format as follows (json):
```
{
    "parameter_name_you_want_to_tune": {
        "<type ="float", "bool", "int">": [lower_bound, upper_bound, step_size]
    },
    "another_parameter": {
        "bool": [] 
    }
}
```
Boolean values don't need any bounds.  The parameter names should match those found in your model config file, if you have one.
2. Create your evaluation function.  This function needs to take in two arguments: a config file/list of values being tuned if no config, and a `n_values` parameter.  Your function needs to return a list of values of size `n_values`.

Example overall usage:
```
from tuningdeep import TuningDeap

def eval_function(config_file, n_values):
    return your_eval_function(config_file, n_values)

tune = TuningDeap(eval_function, tuning_config, model_config)
best_config = tune.run_evolutionary()
```