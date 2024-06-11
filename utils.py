import torch
from PIL import Image
from torchvision.transforms import ToTensor
from models.linear_model import MnistMlp
from models.convolutional_model import MnistCnn, Cifar10Cnn9l
from models.lightning_model import ClassificationModel
import numpy as np

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, indices=None, transform=ToTensor()):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if indices is not None:
            self.data = data[indices]
            self.targets = targets[indices]
        else:
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

class CompressionSetIndexes(torch.Tensor):
    def __init__(self, n:int ):
        super().__init__()
        self.validation_set = torch.ones(n, dtype=torch.bool) # True if the data is in the validation set

    def get_validation_set_length(self):
        return self.validation_set.sum()
    
    def get_compression_set_length(self):
        return self.validation_set.shape[0] - self.get_validation_set_length()
    
    def get_validation_set(self):
        return self.validation_set
    
    def get_compression_set(self):
        return ~self.validation_set
    
    def update_compression_set(self, indices) -> None:
        self.validation_set[indices] = False

    def correct_idx(self,indices):
        return self.validation_set.nonzero()[indices]


def get_max_error_idx(errors, k):
    error_tensor = torch.cat(errors)
    values, indices = torch.topk(error_tensor, k)
    return values.max(), indices

def create_model(config):
    if config['prior_size'] == 0.0:
        lr = config['learning_rate']
    else:
        lr = config['pretraining_lr']

    if config['dataset'] == "mnist":
        if config['model_type'] == "mlp":
            return ClassificationModel(MnistMlp(dataset_shape=784,
                                                n_classes=config['n_classes'],
                                                dropout_probability=config['dropout_probability']),
                                                optimizer=config['optimizer'],
                                                lr=lr,
                                                momentum=config['momentum'],
                                                batch_size=config['batch_size']
                                                )
        elif config['model_type'] == "cnn":
            return ClassificationModel(MnistCnn(n_classes=config['n_classes'],
                                                dropout_probability=config['dropout_probability']),
                                                optimizer=config['optimizer'],
                                                lr=lr,
                                                momentum=config['momentum'],
                                                batch_size=config['batch_size']
                                                )
    elif config['dataset'] == "cifar10":
        return ClassificationModel(Cifar10Cnn9l(n_classes=config['n_classes'],
                                                dropout_probability=config['dropout_probability']),
                                                optimizer=config['optimizer'],
                                                lr=lr,
                                                momentum=config['momentum'],
                                                batch_size=config['batch_size']
                                                )
    
    raise NotImplementedError(f"Model type = {config['model_type']} with dataset {config['dataset']} is not implemented yet.")

def update_learning_rate(model, lr) -> None:
    model.lr = lr

def split_prior_train_dataset(dataset, prior_size):
    if prior_size == 0.0:
        return None, CustomDataset(dataset.data, dataset.targets)
    
    prior_data, train_data = torch.utils.data.random_split(dataset, [prior_size, 1-prior_size])
    prior_set = CustomDataset(dataset.data, dataset.targets, indices=prior_data.indices)
    train_set = CustomDataset(dataset.data, dataset.targets, indices=train_data.indices)
    
    assert len(prior_set) + len(train_set) == len(dataset)

    return prior_set, train_set