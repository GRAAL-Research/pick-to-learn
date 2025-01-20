from dataset.mnist import load_binary_mnist, load_low_high_mnist, load_mnist, load_random_mnist
from dataset.cifar10 import load_binary_cifar10
from dataset.regression_datasets import load_concrete, load_parkinson, load_powerplant, load_infrared, load_airfoil
from dataset.classification_datasets import *
from dataset.amazon_polarity import load_amazon_polarity
from dataset.toys import load_moons


def load_dataset(config):
    if config['dataset'] == "mnist":
        if config['n_classes'] == 2:
            if config['first_class'] == -1:
                return load_low_high_mnist()
            else:
                return load_binary_mnist(low=config['first_class'],
                                                    high=config['second_class'])
        elif config['n_classes'] == 10:
            return load_mnist()
        else:
            raise NotImplementedError(f"{config['dataset']} with {config['n_classes']} is not supported.")
    elif config['dataset'] == "randomMnist":
        return load_random_mnist()
    elif config['dataset'] == "cifar10":
        if config['n_classes'] == 2:
            return load_binary_cifar10(low=config['first_class'],
                                                high=config['second_class'])
        else:
            raise NotImplementedError(f"{config['dataset']} with {config['n_classes']} is not supported.")
    elif config['dataset'] == "concrete":
        return load_concrete(config['test_size'])
    elif config['dataset'] == "parkinson":
        return load_parkinson(config['test_size'])
    elif config['dataset'] == "airfoil":
        return load_airfoil(config['test_size'])
    elif config['dataset'] == "infrared":
        return load_infrared(config['test_size'])
    elif config['dataset'] == "powerplant":
        return load_powerplant(config['test_size'])
    elif config['dataset'] == "amazon":
        return load_amazon_polarity(config['n_shards'])
    elif config['dataset'] == "moons":
        return load_moons(config['seed'], config['test_size'])
    elif config['dataset'] == 'rice':
        return load_rice(config['test_size'])
    elif config['dataset'] == 'wine':
        return load_wine(config['test_size'])
    elif config['dataset'] == 'statlog':
        return load_statlog(config['test_size'])
    elif config['dataset'] == 'breast_cancer':
        return load_breast_cancer(config['test_size'])
    elif config['dataset'] == 'image_segmentation':
        return load_image_segmentation(config['test_size'])
    elif config['dataset'] == 'mice_protein':
        return load_mice_protein(config['test_size'])
    elif config['dataset'] == 'htru2':
        return load_htru2(config['test_size'])
    else:
        raise NotImplementedError(f"The dataset {config['dataset']} is not supported yet.")