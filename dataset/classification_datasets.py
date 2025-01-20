from ucimlrepo import fetch_ucirepo 
from utilities.utils_datasets import CustomDataset, split_train_validation_dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
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

def normalize_data(train_set, test_set):
    mean_ = np.nan_to_num(np.nanmean(train_set.data, axis=0), 0)
    std_ = np.nanstd(train_set.data, axis=0, ddof=1) + .000001
    std_[np.isnan(std_)] = 1

    train_set.data = (train_set.data - mean_) / std_
    test_set.data = (test_set.data - mean_) / std_

    return train_set, test_set

def load_uci_repo(id:int, test_size:float=0.1, target_name=None):
    uci_dataset = fetch_repo(id=id)
    X = torch.tensor(uci_dataset.data.features.to_numpy(), dtype=torch.float32)
    if target_name is None:
        try:
            y = torch.tensor(uci_dataset.data.targets.to_numpy())
        except TypeError:
            le = LabelEncoder()
            labels = uci_dataset.data.targets.to_numpy(dtype=np.object_).reshape(-1,)
            y = torch.tensor(le.fit_transform(labels))
    else:
        y = torch.tensor(uci_dataset.data.targets[target_name].to_numpy())
    
    le = LabelEncoder()
    le.fit(y.unique())
    y = le.transform(y)
    dataset = CustomDataset(X, y.reshape(-1,), transform=None, real_targets=True, is_an_image=False)
    train_set, test_set = split_train_validation_dataset(dataset, test_size)

    train_set, test_set = normalize_data(train_set, test_set)
    collate_fn = None
    return train_set, test_set, collate_fn


def load_rice(test_size:float=0.1):
    # https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik
    return load_uci_repo(id=545, test_size=test_size)
    # uci_dataset = fetch_repo(id=545)
    # X = torch.tensor(uci_dataset.data.features.to_numpy(), dtype=torch.float32)
    # y = torch.tensor((uci_dataset.data.targets == 'Cammeo').to_numpy()).to(int)
    # dataset = CustomDataset(X, y.reshape(-1,), transform=None, real_targets=True, is_an_image=False)
    # train_set, test_set = split_train_validation_dataset(dataset, test_size)

    # train_set, test_set = normalize_data(train_set, test_set)

    # collate_fn = None
    # return train_set, test_set, collate_fn

def load_wine(test_size:float=0.1):
    # https://archive.ics.uci.edu/dataset/186/wine+quality
    return load_uci_repo(id=186, test_size=test_size)

def load_statlog(test_size:float=0.1):
    # https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite
    return load_uci_repo(id=146, test_size=test_size)

def load_breast_cancer(test_size:float=0.1):
    # https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
    return load_uci_repo(id=17, test_size=test_size)

def load_image_segmentation(test_size:float=0.1):
    # https://archive.ics.uci.edu/dataset/50/image+segmentation
    return load_uci_repo(id=50, test_size=test_size)

def load_mice_protein(test_size:float=0.1):
    # https://archive.ics.uci.edu/dataset/342/mice+protein+expression
    uci_dataset = fetch_repo(id=342)
    enc = OneHotEncoder(drop='first', sparse_output=False)
    X = uci_dataset.data.features
    X_encoded = enc.fit_transform(X[['Genotype', 'Treatment', 'Behavior']])
    X = X.drop(columns=['Genotype', 'Treatment', 'Behavior'])
    X = pd.concat((X,pd.DataFrame(X_encoded, columns=['Genotype', 'Treatment', 'Behavior'])),axis=1)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imp.fit_transform(X)
    X = torch.tensor(X, dtype=torch.float32)
    le = LabelEncoder()
    labels = uci_dataset.data.targets.to_numpy(dtype=np.object_).reshape(-1,)
    y = torch.tensor(le.fit_transform(labels))

    dataset = CustomDataset(X, y.reshape(-1,), transform=None, real_targets=True, is_an_image=False)
    train_set, test_set = split_train_validation_dataset(dataset, test_size)

    train_set, test_set = normalize_data(train_set, test_set)
    collate_fn = None
    return train_set, test_set, collate_fn

def load_htru2(test_size:float=0.1):
    # https://archive.ics.uci.edu/dataset/372/htru2
    return load_uci_repo(id=372, test_size=test_size)