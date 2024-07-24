import lightning as L
import torch

class ClampedMSELoss(torch.nn.Module):
    def __init__(self, clamping=False, min_val=0, max_val=torch.inf):
        super().__init__()
        self.clamping=clamping
        self.min_val = min_val
        self.max_val = max_val
        self.max_error = (self.max_val - self.min_val)**2
        self.loss = torch.nn.MSELoss(reduction='none')

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out = self.loss(input, target)
        if self.clamping:
            out = torch.clamp(out, max=self.max_error)
        return out.mean()
    

class RegressionModel(L.LightningModule):
    def __init__(self, model, optimizer="Adam", lr=1e-3, momentum=0.95, batch_size=64):
        super().__init__()
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.model = model
        self.configure_loss(clamping=False)
        self.no_reduction_loss = torch.nn.MSELoss(reduction='none')

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

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


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        test_loss = self.loss(y_hat, y)
        self.log("test_loss", test_loss)
    
    def configure_optimizers(self):
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        
        return optimizer
    
    def configure_loss(self, clamping : bool = False, min_val : float = 0, max_val=torch.inf):
        self.loss = ClampedMSELoss(clamping=clamping, min_val=min_val, max_val=max_val)
    
