from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from utils import CustomDataset

def load_binary_mnist(low=1, high=7):
    transform = ToTensor()
    train_set = MNIST(root="MNIST", download=True, train=True, transform=transform)
    train_labels = (train_set.targets == low) | (train_set.targets == high)
    train_set.data = train_set.data[train_labels]
    train_set.targets = (train_set.targets[train_labels] == low).to(int)
    
    assert train_set.data.shape[0] == train_set.targets.shape[0]

    test_set = MNIST(root="MNIST", download=True, train=False, transform=transform)
    test_labels = (test_set.targets == low) | (test_set.targets == high)
    test_set.data = test_set.data[test_labels]
    test_set.targets = (test_set.targets[test_labels] == low).to(int)

    assert test_set.data.shape[0] == test_set.targets.shape[0]

    train_dataset = CustomDataset(data=train_set.data, targets=train_set.targets, transform=transform, real_targets=False, is_an_image=True)
    test_dataset = CustomDataset(data=test_set.data, targets=test_set.targets, transform=transform, real_targets=False, is_an_image=True)

    return train_dataset, test_dataset

def load_low_high_mnist():
    separator = 4

    transform = ToTensor()
    train_set = MNIST(root="MNIST", download=True, train=True, transform=transform)
    train_set.targets = (train_set.targets > separator).to(int)

    test_set = MNIST(root="MNIST", download=True, train=False, transform=transform)
    test_set.targets = (test_set.targets > separator).to(int)

    train_dataset = CustomDataset(data=train_set.data, targets=train_set.targets, transform=transform, real_targets=False, is_an_image=True)
    test_dataset = CustomDataset(data=test_set.data, targets=test_set.targets, transform=transform, real_targets=False, is_an_image=True)

    return train_dataset, test_dataset

def load_mnist():
    transform = ToTensor()
    train_set = MNIST(root="MNIST", download=True, train=True, transform=transform)
    test_set = MNIST(root="MNIST", download=True, train=False, transform=transform)

    train_dataset = CustomDataset(data=train_set.data, targets=train_set.targets, transform=transform, real_targets=False, is_an_image=True)
    test_dataset = CustomDataset(data=test_set.data, targets=test_set.targets, transform=transform, real_targets=False, is_an_image=True)

    return train_dataset, test_dataset


if __name__ == "__main__":
    load_binary_mnist()