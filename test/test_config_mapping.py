import unittest
from tuningdeap.config_mapping import get_by_path, set_by_path, get_paths
import json
import os
import logging

logging.basicConfig(level=logging.INFO)

class TestConfigMapping(unittest.TestCase):
    def setUp(self):
        cwd = os.getcwd()
        self.tuning_config = {
                                        "population_size": 1,
                                        "n_generations": 1,
                                        "output": False,
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
                                            }
                                        }
                                    }

    def test_gather_from_map_valid(self):
        test_value = "three_levels"
        test_config = {
            "a": 1,
            "b": True,
            "c": {
                "d": {
                    "e": test_value
                }
            }
        }
        path = ["c", "d", "e"]
        value = get_by_path(test_config, path)
        assert value == "three_levels", "did not gather the correct value from the map, expected {} but got {}".format(value, test_value)

    def test_gather_from_map_invalid(self):
        test_value = "three_levels"
        test_config = {
            "a": 1,
            "b": True,
            "c": {
                "d": {
                    "e": test_value
                }
            }
        }
        path = ["c", "d", "e"]
        try:
            value = get_by_path(test_config, path)
            self.fail("Grabbed a value for an incorrect key/path combination")
        except Exception as e:
            pass

    def test_set_from_map_valid(self):
        test_value = "three_levels"
        new_value = "new_value"
        test_config = {
            "a": 1,
            "b": True,
            "c": {
                "d": {
                    "e": test_value
                }
            }
        }
        path = ["c", "d", "e"]
        set_by_path(test_config, path, new_value, is_bool=False)
        value = get_by_path(test_config, path)
        assert value == new_value != test_value, "did not set the correct value from the map, expected {} but got {} (which is not the original {})".format(new_value, value, test_value)

    def test_set_from_map_invalid_dict(self):
        path = ["c", "d", "e"]
        try:
            set_by_path("a", path, "new_value", is_bool=False)
            self.fail("Should not allow a string to be passed in as the dict_object")
        except Exception:
            pass
    
    def test_set_from_map_valid_bool(self):
        test_value = True
        new_value = 0
        test_config = {
            "a": 1,
            "b": test_value,
            "c": {
                "d": {
                    "e": "some_value"
                }
            }
        }
        path = ["b"]
        set_by_path(test_config, path, new_value, is_bool=True)
        value = get_by_path(test_config, path)
        assert value == bool(new_value) and type(value) == bool, "did not set the correct value from the map, expected {} but got {} (which is not the original {})".format("bool", bool(new_value), type(value), value)
        set_by_path(test_config, path, new_value, is_bool=False)
        int_value = get_by_path(test_config, path)
        assert value == int_value and type(int_value) == int, "did not set the correct value from the map, expected type {} = {} but got type {} = {})".format("int", int_value, type(int_value), new_value)

    def test_set_from_map_invalid(self):
        test_value = "three_levels"
        new_value = "new_value"
        test_config = {
            "a": 1,
            "b": True,
            "c": {
                "d": {
                    "e": test_value
                }
            }
        }
        path = ["c", "d", "e"]
        try:
            set_by_path(test_config, path, new_value)
            self.fail("Should not have been able to set value with incorrect path/key combo")
        except Exception:
            pass
    
    def test_get_paths_short(self):
        expected_output = [
            ["a"], ["b"], ["c"], ["c", "d"], ["c", "d", "e"]
        ]
        test_config = {
            "a": 1,
            "b": True,
            "c": {
                "d": {
                    "e": "value"
                }
            }
        }
        output = get_paths(test_config)
        assert output == expected_output, "path output was not what was expected, failed"

    def test_get_paths_long(self):
        expected_output = [
            ["a"], ["b"], ["c"], ["c", "d"], ["c", "d", "e"], ["c", "d", "e", "f"], ["c", "d", "e", "f", "g"]
        ]
        test_config = {
            "a": 1,
            "b": True,
            "c": {
                "d": {
                    "e": {
                        "f": {
                            "g":  "done!"
                        }
                    }
                }
            }
        }
        output = get_paths(test_config)
        assert output == expected_output, "path output was not what was expected for the long version, failed"

    def test_get_paths_empty(self):
        expected_output = []
        test_config = {}
        output = get_paths(test_config)
        assert output == expected_output, "path output was not what was expected for the empty version, failed"

    

    

    
