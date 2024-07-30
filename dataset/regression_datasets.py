from ucimlrepo import fetch_ucirepo 
from utils import CustomDataset, split_train_validation_dataset
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
            time.sleep(5)
    return uci_dataset


def load_uci_repo(id:int, test_size:float=0.1, target_name=None):
    uci_dataset = fetch_repo(id=id)
    X = torch.tensor(uci_dataset.data.features.to_numpy())
    if target_name is None:
        y = torch.tensor(uci_dataset.data.targets.to_numpy())
    else:
        y = torch.tensor(uci_dataset.data.targets[target_name].to_numpy())
    dataset = CustomDataset(X, y.reshape(-1,), real_targets=True)
    return split_train_validation_dataset(dataset, test_size)


def load_concrete(test_size:float=0.1):
    # https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
    return load_uci_repo(id=165, test_size=test_size)

def load_parkinson(test_size: float = 0.1):
    # https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring
    return load_uci_repo(id=189, test_size=test_size, target_name='motor_UPDRS')

def load_airfoil(test_size: float = 0.1):
    # https://archive.ics.uci.edu/dataset/291/airfoil+self+noise
    return load_uci_repo(id=291, test_size=test_size)

def load_powerplant(test_size: float = 0.1):
    # https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant
    return load_uci_repo(id=294, test_size=test_size)

def load_infrared(test_size: float = 0.1):
    # https://archive.ics.uci.edu/dataset/925/infrared+thermography+temperature+dataset
    uci_dataset = fetch_repo(id=925)
    X = pd.get_dummies(uci_dataset.data.features, columns=["Gender", 'Ethnicity', 'Age'])
    X = torch.tensor(X.to_numpy(dtype=np.float64))
    y = torch.tensor(uci_dataset.data.targets['aveOralF'].to_numpy())
    dataset = CustomDataset(X, y.reshape(-1,), real_targets=True)
    return split_train_validation_dataset(dataset, test_size)

