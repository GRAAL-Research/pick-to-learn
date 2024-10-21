from sklearn.datasets import make_moons
from utilities.utils_datasets  import CustomDataset, split_train_validation_dataset
import torch

def load_moons(random_state, test_size):
    X, y = make_moons(n_samples=500, noise=0.2, random_state=random_state)
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    dataset = CustomDataset(X_tensor, y_tensor, transform=None, real_targets=False, is_an_image=False)

    train_set, test_set = split_train_validation_dataset(dataset, test_size)
    collate_fn = None
    return train_set, test_set, collate_fn

