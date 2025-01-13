from ucimlrepo import fetch_ucirepo 
from utilities.utils_datasets import CustomDataset, split_train_validation_dataset
import torch
import pandas as pd
import numpy as np
import time

def fetch_repo(id):
    while True:
        try:
            uci_dataset = fetch_ucirepo(id=id)
            break
        except ConnectionError:
            time.sleep(20)
    return uci_dataset


def load_uci_repo(id:int, test_size:float=0.1, target_name=None):
    uci_dataset = fetch_repo(id=id)
    X = torch.tensor(uci_dataset.data.features.to_numpy())
    if target_name is None:
        y = torch.tensor(uci_dataset.data.targets.to_numpy())
    else:
        y = torch.tensor(uci_dataset.data.targets[target_name].to_numpy())
    dataset = CustomDataset(X, y.reshape(-1,), transform=None, real_targets=True, is_an_image=False)
    train_set, test_set = split_train_validation_dataset(dataset, test_size)
    collate_fn = None
    return train_set, test_set, collate_fn


def load_rice(test_size:float=0.1):
    # https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik
    uci_dataset = fetch_repo(id=545)
    X = torch.tensor(uci_dataset.data.features.to_numpy())
    y = torch.tensor((uci_dataset.data.targets == 'Cammeo').to_numpy()).to(int)
    dataset = CustomDataset(X, y.reshape(-1,), transform=None, real_targets=True, is_an_image=False)
    train_set, test_set = split_train_validation_dataset(dataset, test_size)
    collate_fn = None
    return train_set, test_set, collate_fn

def load_wine(test_size:float=0.1):
    # https://archive.ics.uci.edu/dataset/186/wine+quality
    return load_uci_repo(id=186, test_size=test_size)

def load_statlog(test_size:float=0.1):
    # https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite
    return load_uci_repo(id=146, test_size=test_size)




