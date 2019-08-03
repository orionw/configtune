import typing

class DummyModel:
    def __init__(self):
        pass

    def predict(self, real_config_updated: dict, n_values: int) -> typing.List:
        """
        A basic evaluation function that takes in a config file
        :param real_config_updated: the "real" like config file
        :param n_values: an int of the number of values, so we can return the right number
        :return a list of n values
        """
        value = real_config_updated["name1"] * real_config_updated["name3"] if real_config_updated["name2"] else 0
        return [value * n_values]

    def predict_no_config(self, values: typing.List, n_values: int) -> typing.List:
        """
        A basic evaluation function with no config file
        :param values: the values from the genetic algorithm
        :param n_values: an int of the number of values, so we can return the right number
        :return a list of n_values
        """
        return [(values[0] * 2) * n_values] if values[1] is True else [(1 / (values[2] + 0.001)) * n_values]