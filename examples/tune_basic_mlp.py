import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from configtune import TuningDeap, TuningBayes
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import os


class MLP(nn.Module):

    def __init__(self, input_layer, hidden_units=100, hidden_layers=1):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.extend([nn.Linear(input_layer, hidden_units),
                               nn.ReLU()])
        # add additional layers
        for k in range(hidden_layers - 1):
            self.layers.extend([nn.Linear(hidden_units, hidden_units),
                               nn.ReLU()])
        # final layer
        self.layers.extend([nn.Linear(hidden_units, 3), nn.Softmax(dim=1)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def fit(config, train_loader):
    model = MLP(config["input_shape"], config["hidden_units"], config["hidden_layers"])
    device = "gpu:0" if torch.cuda.is_available() else "cpu"

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(config["n_epochs"]):
        for batch, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

    return model


def evaluate(model, validate_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (data, labels) in enumerate(validate_loader):
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return correct/total


def make_loader(X, y, batch_size=10, shuffle=False, num_workers=1):
    data_x = torch.Tensor(X.values).float()
    data_y = torch.Tensor(y.values).long()
    ds = TensorDataset(data_x, data_y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader


def execute_example_tuning():
     # prepare dataset/dataloaders
    iris = pd.read_csv(os.path.join("examples", "example_data", "iris.csv"), header=0)
    train, validate = train_test_split(iris, test_size=0.33, random_state=42)
    y_train = train["species"].map({"versicolor": 0, "virginica": 1, "setosa": 2})
    X_train = train.drop("species", axis=1)
    y_validate = validate["species"].map({"versicolor": 0, "virginica": 1, "setosa": 2})
    X_validate = validate.drop("species", axis=1)
    train_loader = make_loader(X_train, y_train)
    validate_loader = make_loader(X_validate, y_validate)

    # define configs
    tuning_config = {
        "attributes": {
            "hidden_units": {
                    "type": "int",
                    "min": 1,
                    "max": 512,
                    "step": 16
                },
            "hidden_layers": {
                "type": "int",
                "min": 1,
                "max": 10
            }
        }
    }

    model_config = {
        # the first two are tuned
        "hidden_units": 5,
        "hidden_layers": 2,
        # these next three are unused in tuning
        "learning_rate": 0.001,
        "n_epochs": 3,
        "input_shape": 4
    }

    # define evaluation function
    def accuracy_function(config):
        try:
            model = fit(config, train_loader)
            score = evaluate(model, validate_loader)
            return score
        except Exception as e:
            print("Failed because of {}".format(e))
            raise(e)

    # now we tune gentically
    tune = TuningDeap(accuracy_function, tuning_config, model_config, minimize=False, verbose=True, population_size=10, n_generations=5)
    params, score = tune.run()
    print("The best scores with TuningDeap was {} with parameters: {}".format(score, ' '.join([str(x) for x in params])))

    # bayesian optimization can only do mins
    def loss_function(config: dict):
        return 1 - accuracy_function(config)
    tune_bayes = TuningBayes(loss_function, tuning_config, model_config, minimize=True, n_calls=50, verbose=True)
    params, score = tune_bayes.run()
    print("The best scores with TuningBayes was {} with parameters: {}".format(score, ' '.join([str(x) for x in params])))

if __name__ == "__main__":
    execute_example_tuning()
