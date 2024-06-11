from torch import optim, nn
import lightning as L
import torch
from torchmetrics.classification import MulticlassAccuracy


class ClassificationModel(L.LightningModule):
    def __init__(self, model, optimizer="Adam", lr=1e-3, momentum=0.95, batch_size=64):
        super().__init__()
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.model = model
        self.loss = nn.CrossEntropyLoss()
        self.metric = MulticlassAccuracy(num_classes=self.model.n_classes).to(self.device)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        train_acc = self.metric(torch.argmax(y_hat, dim=1), y)
        self.log("train_acc", train_acc, prog_bar=True)

        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss =  nn.CrossEntropyLoss(reduction='none')(y_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        validation_loss = self.loss(y_hat, y)
        self.log("validation_loss", validation_loss)
        validation_acc = self.metric(torch.argmax(y_hat, dim=1), y)
        self.log("validation_acc", validation_acc)
        validation_error = 1 - validation_acc
        self.log("validation_error", validation_error)


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        test_loss = self.loss(y_hat, y)
        self.log("test_loss", test_loss)
        test_acc = self.metric(torch.argmax(y_hat, dim=1), y)
        self.log("test_acc", test_acc)
        test_error = 1 - test_acc
        self.log("test_error", test_error)
    
    def configure_optimizers(self):
        if self.optimizer == "Adam":
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == "SGD":
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        
        return optimizer

