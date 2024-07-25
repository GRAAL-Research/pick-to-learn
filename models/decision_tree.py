import torch
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
from models.regression_model import RegressionModel

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
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, seed=42):
        super().__init__()
        self.max_depth = max_depth
        self.seed = seed
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                        min_samples_split=self.min_samples_split,
                                        min_samples_leaf=self.min_samples_leaf,
                                        random_state=self.seed)
        self.is_fitted = False

    def squeeze(self, X):
        X = X.squeeze()
        if len(X.shape) < 2:
            X = X.reshape(1, -1)
        return X
    
    def fit(self, X, y):
        X = self.squeeze(X)
        self.tree.fit(X, y)
        self.is_fitted = True
    
    def forward(self, input):
        if not self.is_fitted:
            return torch.ones(input.shape[0])
        input = self.squeeze(input)
        return torch.tensor(self.tree.predict(input))
    
class RegressionForest(torch.nn.Module):
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, seed=42):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.forest = RandomForestRegressor(n_estimators=self.n_estimators, 
                                        max_depth=self.max_depth,
                                        min_samples_split=self.min_samples_split,
                                        min_samples_leaf=self.min_samples_leaf,
                                        random_state=self.seed)
        self.is_fitted = False

    def squeeze(self, X):
        X = X.squeeze()
        if len(X.shape) < 2:
            X = X.reshape(1, -1)
        return X
    
    def fit(self, X, y):
        X = self.squeeze(X)
        self.forest.fit(X, y)
        self.is_fitted = True
    
    def forward(self, input):
        if not self.is_fitted:
            return torch.ones(input.shape[0])
        input = self.squeeze(input)
        return torch.tensor(self.forest.predict(input))
    
class RegressionTreeModel(RegressionModel):
    def __init__(self, model):
        super().__init__(model)
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.model.fit(x, y)
        return super().training_step(batch, batch_idx)

    def configure_optimizers(self):
        return None