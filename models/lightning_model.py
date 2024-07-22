import lightning as L
import torch
from torchmetrics.classification import MulticlassAccuracy
import math

class ClampedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, clamping=False, pmin=1e-5, reduction='mean'):
        super().__init__()
        self.clamping=clamping
        self.pmin = pmin
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.loss = torch.nn.NLLLoss(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out = self.log_softmax(input)
        if self.clamping:
            out = torch.clamp(out, min=math.log(self.pmin))
        return self.loss(out, target)
    

class ClassificationModel(L.LightningModule):
    def __init__(self, model, optimizer="Adam", lr=1e-3, momentum=0.95, batch_size=64):
        super().__init__()
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.model = model
        self.configure_loss(clamping=False)
        self.metric = MulticlassAccuracy(num_classes=self.model.n_classes).to(self.device)
        self.no_reduction_loss = torch.nn.CrossEntropyLoss(reduction='none')

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
        loss = self.no_reduction_loss(y_hat, y)
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
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        
        return optimizer
    
    def configure_loss(self, clamping : bool = False, pmin : float = 1e-5):
        self.loss = ClampedCrossEntropyLoss(clamping=clamping, pmin=pmin, reduction='mean')
    
