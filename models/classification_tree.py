import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from models.classification_model import ClassificationModel

# WARNING : This class has not been tested yet. See Issue #8
class ClassificationTree(torch.nn.Module):
    """
    WARNING : Expects the dataset to have targets in [0, ..., n_classes -1]. 
    """

    def __init__(self, n_classes=2,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                seed=42,
                ccp_alpha=0.0):
        super().__init__()
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.seed = seed
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.ccp_alpha = ccp_alpha
        self.tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                        min_samples_split=self.min_samples_split,
                                        min_samples_leaf=self.min_samples_leaf,
                                        random_state=self.seed,
                                        ccp_alpha=self.ccp_alpha)
        self.is_fitted = False
        self.observed_classes = None

    def fit(self, X, y):
        self.tree.fit(X, y)
        self.is_fitted = True
        if self.observed_classes is None or self.observed_classes.shape[0] < self.n_classes:
            self.observed_classes = torch.unique(y)
    
    def forward(self, input):
        if not self.is_fitted:
            return torch.ones((input.shape[0], self.n_classes))
        
        out = torch.tensor(self.tree.predict_proba(input))
        if self.observed_classes.shape[0] < self.n_classes:
            zero_matrix = torch.zeros((out.shape[0], self.n_classes))
            for i in range(self.observed_classes.shape[0]):
                target = self.observed_classes
                zero_matrix[:, target] += out[i, :]
            out = zero_matrix
        return out


class ClassificationForest(torch.nn.Module):
    def __init__(self, n_classes=2, n_estimators=100, max_depth=10, min_samples_split=2,
                  min_samples_leaf=1, seed=42, n_jobs=5, ccp_alpha=0.0, warm_start=False, n_add_estimators=10):
        super().__init__()
        self.n_classes = n_classes
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.warm_start = warm_start
        self.ccp_alpha = ccp_alpha
        self.n_add_estimators = n_add_estimators
        start_nbr_estimators = self.n_add_estimators if self.warm_start else self.n_estimators
        self.forest = RandomForestClassifier(n_estimators=start_nbr_estimators, 
                                        max_depth=self.max_depth,
                                        min_samples_split=self.min_samples_split,
                                        min_samples_leaf=self.min_samples_leaf,
                                        n_jobs=self.n_jobs, 
                                        warm_start=self.warm_start,
                                        ccp_alpha=self.ccp_alpha,
                                        random_state=self.seed)
        self.is_fitted = False
        self.observed_classes = None
    
    def set_warm_start(self):
        if len(self.forest.estimators_) < self.n_estimators:
            n_estimators = len(self.forest.estimators_) + self.n_add_estimators
            self.forest.set_params(n_estimators=n_estimators)
        else:
            self.forest.set_params(warm_start=False)

    def fit(self, X, y):
        if self.warm_start:
            self.set_warm_start()
        self.forest.fit(X, y)
        self.is_fitted = True
        if self.observed_classes is None or self.observed_classes.shape[0] < self.n_classes:
            self.observed_classes = torch.unique(y)
    
    def forward(self, input):
        if not self.is_fitted:
            return torch.ones((input.shape[0], self.n_classes))
        
        out = torch.tensor(self.forest.predict_proba(input))
        if self.observed_classes.shape[0] < self.n_classes:
            zero_matrix = torch.zeros((out.shape[0], self.n_classes))
            for i in range(self.observed_classes.shape[0]):
                target = self.observed_classes
                zero_matrix[:, target] += out[i, :]
            out = zero_matrix
        return out

class ClassificationTreeModel(ClassificationModel):
    def __init__(self, model):
        super().__init__(model)
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.model.fit(x, y)
        return super().training_step(batch, batch_idx)

    def configure_optimizers(self):
        return None
