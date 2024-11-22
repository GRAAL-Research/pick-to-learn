import lightning as L
from torchmetrics.classification import MulticlassAccuracy
import math
from torch import optim, nn


class MnistAutoEncoder(nn.Module):
    def __init__(self, input_size=784, latent_dims=32, dropout_probability=0.2):
        super().__init__()
        self.input_size = input_size
        self.latent_dims = latent_dims
        self.dropout_probability = dropout_probability
        self.encoder = nn.Sequential(nn.Linear(self.input_size, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, self.latent_dims)
        )
        self.decoder = nn.Sequential(nn.Linear(self.latent_dims, 256),
                                    nn.ReLU(),
                                     nn.Linear(256, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 784),
                                     nn.Sigmoid()
        )   

    def forward(self, input):
        out = self.encoder(input)
        return self.decoder(out)


class AutoEncoderModel(L.LightningModule):
    def __init__(self, model, optimizer="Adam", lr=1e-3, momentum=0.95, batch_size=64):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.model = model
        self.loss = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.flatten(1)
        out = self.model(x)
        loss = self.loss(out, x)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.flatten(1)
        out = self.model(x)
        validation_loss = self.loss(out, x)
        self.log("validation_loss", validation_loss)
        return validation_loss

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        x = x.flatten(1)
        out = self.model.encoder(x)
        return out

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x = x.flatten(1)
        out = self.model(x)

        test_loss = self.loss(out, x)
        self.log("test_loss", test_loss)
    
    def configure_optimizers(self):
        if self.optimizer == "Adam":
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == "SGD":
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        
        return optimizer


def create_autoencoder(config):
    if config['dataset'] == "mnist":
        return AutoEncoderModel(MnistAutoEncoder())
    
    raise f"There is no implementation for the autoencoder with the dataset {config['dataset']}"