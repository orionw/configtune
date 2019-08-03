

class DummyModel:
    def __init__(self):
        pass

    def predict(self, real_config_updated, n_values):
        value = real_config_updated["name1"] * real_config_updated["name3"] if real_config_updated["name2"] else 0
        return [value * n_values]