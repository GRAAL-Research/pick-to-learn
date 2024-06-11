from torch import nn

class MnistCnn(nn.Module):
    def __init__(self, n_classes=2, dropout_probability=0.2):
        super().__init__()
        self.n_classes = n_classes
        self.dropout_probability = dropout_probability
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_probability),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_probability),
            nn.MaxPool2d(2)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(128, self.n_classes),
        )

    def forward(self, input):
        out = self.cnn_layers(input)
        return self.linear_layers(out.flatten(1))


class Cifar10Cnn9l(nn.Module):
    def __init__(self, n_classes=2, dropout_probability=0.2):
        super().__init__()
        self.n_classes = n_classes
        self.dropout_probability = dropout_probability
        self.cnn_layers = nn.Sequential(
            # layer 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_probability),
            # layer 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_probability),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #layer 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_probability),
            # layer 4
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_probability),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # layer 5
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_probability),
            # layer 6
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_probability),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(512, n_classes)
        )

    def forward(self, input):
        out = self.cnn_layers(input)
        return self.linear_layers(out.flatten(1))
