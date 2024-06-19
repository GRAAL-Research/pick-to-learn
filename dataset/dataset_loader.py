from dataset.mnist import load_binary_mnist, load_low_high_mnist
from dataset.cifar10 import load_binary_cifar10


def load_dataset(config):
    if config['dataset'] == "mnist":
        if config['n_classes'] == 2:
            if config['first_class'] == -1:
                return load_low_high_mnist()
            else:
                return load_binary_mnist(low=config['first_class'],
                                                    high=config['second_class'])
        else:
            raise NotImplementedError(f"{config['dataset']} with {config['n_classes']} is not supported.")
    elif config['dataset'] == "cifar10":
        if config['n_classes'] == 2:
            return load_binary_cifar10(low=config['first_class'],
                                                high=config['second_class'])
        else:
            raise NotImplementedError(f"{config['dataset']} with {config['n_classes']} is not supported.")
    else:
        raise NotImplementedError(f"The dataset {config['dataset']} is not supported yet.")