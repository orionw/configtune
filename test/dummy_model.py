import typing

class DummyModel:
    def __init__(self):
        pass

    def predict(self, real_config_updated: dict) -> typing.Tuple[int]:
        """
        A basic evaluation function that takes in a config file
        :param real_config_updated: the "real" like config file
        :return the scores as a tuple
        """
        value = real_config_updated["name1"] * real_config_updated["name3"] if real_config_updated["name2"] else 0
        return tuple((value, ))

    def predict_no_config(self, values: typing.List) -> typing.Tuple[int]:
        """
        A basic evaluation function with no config file
        :param values: the values from the genetic algorithm
        :return the scores as a tuple
        """
        return tuple((values[0] * 2), ) if values[1] is True else tuple((1 / (values[2] + 0.001), ))