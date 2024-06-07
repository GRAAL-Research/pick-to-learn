from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

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

    return train_set, test_set

def load_mnist():
    transform = ToTensor()
    train_set = MNIST(root="MNIST", download=True, train=True, transform=transform)
    test_set = MNIST(root="MNIST", download=True, train=False, transform=transform)
    return train_set, test_set


if __name__ == "__main__":
    load_binary_mnist()