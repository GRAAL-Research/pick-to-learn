from ucimlrepo import fetch_ucirepo 
from utils import CustomDataset, split_train_validation_dataset
import torch

def load_concrete(test_size:float=0.1):
    concrete_compressive_strength = fetch_ucirepo(id=165)
    X = torch.tensor(concrete_compressive_strength.data.features.to_numpy())
    y = torch.tensor(concrete_compressive_strength.data.targets.to_numpy())
    dataset = CustomDataset(X, y)
    return split_train_validation_dataset(dataset, test_size)
