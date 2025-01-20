import torch

class MnistMlp(torch.nn.Module):
    def __init__(self, dataset_shape=784, n_classes=2, dropout_probability=0.2):
        super().__init__()
        self.dataset_shape = dataset_shape
        self.n_classes = n_classes
        self.dropout_probability = dropout_probability
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.dataset_shape, 600),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_probability),
            torch.nn.Linear(600, 600),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_probability),
            torch.nn.Linear(600, 600),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_probability),
            torch.nn.Linear(600, 600),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_probability),
            torch.nn.Linear(600, self.n_classes)
        )

    def forward(self, input):
        x = input.flatten(1)
        return self.layers(x)


class MotherNetMlp(torch.nn.Module):
    def __init__(self, dataset_shape=10, n_classes=2):
        super().__init__()
        self.dataset_shape = dataset_shape
        self.n_classes = n_classes
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.dataset_shape, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, self.n_classes),
        )

    def forward(self, input):
        x = input.flatten(1)
        return self.layers(x)

    def update_weights(self, weights):
        with torch.no_grad():
            j = 0
            for i in range(len(self.layers)):
                if not isinstance(self.layers[i], torch.nn.ReLU):
                    b, w = weights[j]
                    self.layers[i].weight = torch.nn.Parameter(torch.tensor(w.T))
                    self.layers[i].bias = torch.nn.Parameter(torch.tensor(b))
                    j += 1