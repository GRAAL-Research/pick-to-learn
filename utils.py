import torch
from PIL import Image
from torchvision.transforms import ToTensor
from models.linear_model import MnistMlp
from models.convolutional_model import MnistCnn, Cifar10Cnn9l
from models.lightning_model import ClassificationModel
import numpy as np

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=ToTensor()):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.targets[idx])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy()) #, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        return img, target

def get_max_error_idx(errors):
    error_tensor = torch.tensor([errors[i][0].item() for i in range(len(errors))])
    error_idx = torch.tensor([errors[i][1].item() for i in range(len(errors))])
    return error_tensor.max(), error_idx[error_tensor.argmax()]

def create_model(config):
    if config['dataset'] == "mnist":
        if config['model_type'] == "mlp":
            return ClassificationModel(MnistMlp(dataset_shape=784,
                                                n_classes=config['n_classes'],
                                                dropout_probability=config['dropout_probability']),
                                                optimizer=config['optimizer'],
                                                lr=config['learning_rate'],
                                                momentum=config['momentum'],
                                                batch_size=config['batch_size']
                                                )
        elif config['model_type'] == "cnn":
            return ClassificationModel(MnistCnn(n_classes=config['n_classes'],
                                                dropout_probability=config['dropout_probability']),
                                                optimizer=config['optimizer'],
                                                lr=config['learning_rate'],
                                                momentum=config['momentum'],
                                                batch_size=config['batch_size']
                                                )
    elif config['dataset'] == "cifar10":
        return ClassificationModel(Cifar10Cnn9l(n_classes=config['n_classes'],
                                                dropout_probability=config['dropout_probability']),
                                                optimizer=config['optimizer'],
                                                lr=config['learning_rate'],
                                                momentum=config['momentum'],
                                                batch_size=config['batch_size']
                                                )
    
    raise NotImplementedError(f"Model type = {config['model_type']} with dataset {config['dataset']} is not implemented yet.")

def split_prior_train_dataset(dataset, prior_size):
    if prior_size == 0.0:
        return None, CustomDataset(dataset.data, dataset.targets)
    
    prior_data, train_data = torch.utils.data.random_split(dataset, [prior_size, 1-prior_size])
    prior_set = CustomDataset(dataset.data[prior_data.indices], dataset.targets[prior_data.indices])
    train_set = CustomDataset(dataset.data[train_data.indices], dataset.targets[train_data.indices])
    
    assert len(prior_set) + len(train_set) == len(dataset)

    return prior_set, train_set