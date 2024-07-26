import torch
import lightning as L
from PIL import Image
import torch.utils
from torchvision.transforms import ToTensor
from models.linear_network import MnistMlp
from models.convolutional_network import MnistCnn, Cifar10Cnn9l
from models.classification_model import ClassificationModel
from models.decision_tree import RegressionTree, RegressionTreeModel, RegressionForest
from itertools import product
import wandb
from bounds.real_valued_bounds import compute_real_valued_bounds

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
    def __init__(self, n : int ):
        super().__init__()
        self.complement_set = torch.ones(n, dtype=torch.bool) # True if the data is in the validation set

    def get_complement_size(self):
        return int(self.complement_set.sum())
    
    def get_compression_size(self):
        return int(self.complement_set.shape[0] - self.get_complement_size())
    
    def get_complement_data(self):
        return self.complement_set
    
    def get_compression_data(self):
        return ~self.complement_set
    
    def update_compression_set(self, indices) -> None:
        self.complement_set[indices] = False

    def correct_idx(self,indices):
        return self.complement_set.nonzero()[indices]
    

def get_max_error_idx(errors, k):
    error_tensor = torch.cat(errors)
    # on gère le cas où il reste moins de k données dans le jeu de données.
    if error_tensor.shape[0] < k:
        k = error_tensor.shape[0]
    values, indices = torch.topk(error_tensor, k)
    return values.max(), indices

def create_model(config):
    if config.get('prior_size', 0.0) == 0.0:
        lr = config['training_lr']
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
    elif config['dataset'] == "concrete":
        if config['model_type'] == "tree":
            return RegressionTreeModel(RegressionTree(
                max_depth=config['max_depth'],
                min_samples_split=config['min_samples_split'],
                min_samples_leaf=config['min_samples_leaf'],
                seed=config['seed']
            ))
        elif config['model_type'] == "forest":
            return RegressionTreeModel(RegressionForest(
                n_estimators=config['n_estimators'],
                max_depth=config['max_depth'],
                min_samples_split=config['min_samples_split'],
                min_samples_leaf=config['min_samples_leaf'],
                seed=config['seed']
            ))
    
    raise NotImplementedError(f"Model type = {config['model_type']} with dataset {config['dataset']} is not implemented yet.")

def update_learning_rate(model, lr:float) -> None:
    model.lr = lr

def add_clamping_to_model(model, config) -> None:
    if config['regression']:
        model.configure_loss(clamping=True, min_val=config['min_val'], max_val=config['max_val'])
    else:
        model.configure_loss(clamping=True, pmin=config['min_probability'])

def split_prior_train_validation_dataset(dataset : CustomDataset, prior_size : float, validation_size : float):
    if prior_size == 0.0:
        train_data, val_data = torch.utils.data.random_split(dataset, [1-validation_size, validation_size])
        train_set = CustomDataset(dataset.data, dataset.targets, indices=train_data.indices)
        validation_set = CustomDataset(dataset.data, dataset.targets, indices=val_data.indices)

        assert len(train_set) + len(validation_set) == len(dataset)
        return None, train_set, validation_set
    
    splits = [prior_size, 1-prior_size - validation_size, validation_size]
    prior_data, train_data, val_data = torch.utils.data.random_split(dataset, splits)
    prior_set = CustomDataset(dataset.data, dataset.targets, indices=prior_data.indices)
    train_set = CustomDataset(dataset.data, dataset.targets, indices=train_data.indices)
    validation_set = CustomDataset(dataset.data, dataset.targets, indices=val_data.indices)
    
    assert len(prior_set) + len(train_set) + len(validation_set) == len(dataset)

    return prior_set, train_set, validation_set

def split_train_validation_dataset(dataset : CustomDataset, validation_size : float):
    train_data, val_data = torch.utils.data.random_split(dataset, [1-validation_size, validation_size])
    train_set = CustomDataset(dataset.data, dataset.targets, indices=train_data.indices)
    validation_set = CustomDataset(dataset.data, dataset.targets, indices=val_data.indices)

    assert len(train_set) + len(validation_set) == len(dataset)
    return train_set, validation_set

def check_can_be_converted_to_float(entry) -> bool:
    try:
        float_entry = float(entry)
        return True
    except ValueError:
        return False

def correct_type_of_entry(entry):
    if isinstance(entry, list):
        return [correct_type_of_entry(entry_) for entry_ in entry]
    elif isinstance(entry, float):
        return entry
    elif isinstance(entry, str):
        if entry == 'None':
            return None
        elif check_can_be_converted_to_float(entry):
            return float(entry)
        else:
            return entry
    elif isinstance(entry, int):
        return entry
    else:
        raise ValueError(f'The entry type of {entry} is not recognised.')


def create_all_configs(config):
    if config['method'] != 'grid':
        raise NotImplementedError(f'The hyperparameter tuning method {config['method']} is not supported.')
    
    list_of_keys = []
    list_of_hyperparams = []
    for key, item in config['parameters'].items():
        list_of_keys.append(key)
        if item.get('values', None) is not None:
            val_ = correct_type_of_entry(item['values'])
            list_of_hyperparams.append(val_)
        elif item.get('value', None) is not None:
            val_ = correct_type_of_entry(item['value'])
            list_of_hyperparams.append([val_])
        else:
            raise ValueError(f"The parameter {key} doesn't have an item 'value' or 'values'. Please specify one.")
    
    list_of_configs = list(product(*list_of_hyperparams))
    return [dict(zip(list_of_keys, config_)) for config_ in list_of_configs]

def get_exp_file_name(config):
    list_of_params = list(config.values())
    file_name = "exp_"
    for param in list_of_params:
        file_name += str(param) + "_"
    file_name += ".json"
    return "./experiment_logs/" + file_name

def get_updated_batch_size(batch_size, model_type, dataset_length):
    """
    When batch_size == -1, we want to train on the whole dataset.
    """
    if model_type in ['tree', 'forest']:
        return dataset_length
    return batch_size

def get_dataloader(dataset, batch_size, shuffle=False, num_workers=5, persistent_workers=True):
    return torch.utils.data.DataLoader(dataset,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        num_workers=num_workers, 
                                        persistent_workers=persistent_workers)

def get_trainer(accelerator='auto', devices=1, max_epochs=None, logger=False,
                enable_checkpointing=False, callbacks=None):
    return L.Trainer(accelerator=accelerator,
                     devices=devices,
                    max_epochs=max_epochs,
                    logger=logger,
                    enable_checkpointing=enable_checkpointing,
                    callbacks=callbacks)

def get_accelerator(model_type:str):
    return 'cpu' if model_type in ["tree", 'forest'] else 'auto'

def log_metrics(trainer, model, complement_loader, valset_loader, test_loader, compression_set_length, train_set_length, n_sigma):
    complement_res = trainer.validate(model=model, dataloaders=complement_loader)
    validation_res = trainer.validate(model=model, dataloaders=valset_loader)
    test_results = trainer.test(model, dataloaders=test_loader)

    if wandb.config['regression']:
        metrics = {'complement_loss' : complement_res[0]['validation_loss'],
                'validation_loss': validation_res[0]['validation_loss'],
                'test_loss': test_results[0]['test_loss']}

        compute_real_valued_bounds(compression_set_length,
                                    n_sigma,
                                    train_set_length,
                                    complement_res[0]['validation_loss'],
                                    wandb.config['delta'],
                                    wandb.config['nbr_parameter_bounds'],
                                    metrics,
                                    min_val=wandb.config['min_val'],
                                    max_val=wandb.config['max_val'])
    else:
        metrics = {'complement_error' : complement_res[0]['validation_error'],
                'validation_error': validation_res[0]['validation_error'],
                'test_error': test_results[0]['test_error']}

        compute_real_valued_bounds(compression_set_length,
                                    n_sigma,
                                    train_set_length,
                                    complement_res[0]['validation_error'],
                                    wandb.config['delta'],
                                    wandb.config['nbr_parameter_bounds'],
                                    metrics)

    wandb.log(metrics)