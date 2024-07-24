import torch
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from models.classification_model import RegressionModel
from torchmetrics.classification import MulticlassAccuracy
from sklearn.utils.validation import check_is_fitted

# WARNING : This class has not been tested yet. See Issue #8
class DecisionTree(torch.nn.Module):
    def __init__(self, n_classes=2, max_depth=10, min_samples_split=2, min_samples_leaf=1, seed=42):
        super().__init__()
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.seed = seed
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                        min_samples_split=self.min_samples_split,
                                        min_samples_leaf=self.min_samples_leaf,
                                        random_state=self.seed)

    def fit(self, X, y):
        self.tree.fit(X, y)
    
    def forward(self, input):
        if not check_is_fitted(self.tree):
            return torch.ones((input.shape[0], self.n_classes))
        return torch.tensor(self.tree.predict_proba(input))
    
class RegressionTree(torch.nn.Module):
    def __init__(self, n_classes=2, max_depth=10, min_samples_split=2, min_samples_leaf=1, seed=42):
        super().__init__()
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.seed = seed
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                        min_samples_split=self.min_samples_split,
                                        min_samples_leaf=self.min_samples_leaf,
                                        random_state=self.seed)

    def fit(self, X, y):
        self.tree.fit(X, y)
    
    def forward(self, input):
        if not check_is_fitted(self.tree):
            return torch.ones((input.shape[0], self.n_classes))
        return torch.tensor(self.tree.predict(input))

class RegressionTreeModel(RegressionModel):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.model.fit(x,y)
        y_hat = self.model(x)

        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return None