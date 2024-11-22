import torch
import lightning as L
from itertools import product
import wandb
from bounds.real_valued_bounds import compute_real_valued_bounds
import numpy as np
import os
from models.autoencoder import AutoEncoderModel, create_autoencoder


def get_max_error_idx(errors, k):
    error_tensor = torch.cat(errors)
    # on gère le cas où il reste moins de k données dans le jeu de données.
    if error_tensor.shape[0] < k:
        k = error_tensor.shape[0]
    values, indices = torch.topk(error_tensor, k)
    return values.max(), indices

def update_learning_rate(model, lr:float) -> None:
    model.lr = lr

def add_clamping_to_model(model, config) -> None:
    if config['regression']:
        model.configure_loss(clamping=True, min_val=config['min_val'], max_val=config['max_val'])
    else:
        model.configure_loss(clamping=True, pmin=config['min_probability'])

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

def get_exp_file_name(config, path="./experiment_logs/"):
    list_of_params = list(config.values())
    file_name = "exp_"
    for param in list_of_params:
        file_name += str(param) + "_"
    file_name += ".json"
    return path + file_name

def get_updated_batch_size(batch_size, model_type, dataset_length):
    """
    When batch_size == -1, we want to train on the whole dataset.
    """
    if model_type in ['tree', 'forest']:
        return dataset_length
    return batch_size

def get_dataloader(dataset, batch_size, shuffle=False, num_workers=5, persistent_workers=True, collate_fn=None):
    return torch.utils.data.DataLoader(dataset,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        num_workers=num_workers, 
                                        persistent_workers=persistent_workers, 
                                        collate_fn=collate_fn)

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

def log_metrics(trainer, model, complement_loader, valset_loader, test_loader, compression_set_length, train_set_length, n_sigma, return_validation_loss=False):
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
        wandb.log(metrics)
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
        if wandb.config['clamping']:
            compute_real_valued_bounds(compression_set_length, n_sigma, train_set_length, complement_res[0]['validation_loss'], wandb.config['delta'],
                                        wandb.config['nbr_parameter_bounds'], metrics, min_val=0, 
                                        max_val=-np.log(wandb.config['min_probability']), prefix="CE")
        wandb.log(metrics)

    if return_validation_loss:
        return complement_res[0]['validation_loss']


def gaussian_kernel(dist, sigma):
    return torch.exp(- dist.pow(2) / (sigma**2))

def unbiased_mmd(X, Y, sigmas=[1, 2, 5, 10, 25]):
    XX = torch.cdist(X, X)
    YY = torch.cdist(Y, Y)
    XY = torch.cdist(X, Y)

    KXX = torch.zeros(XX.shape)
    KYY = torch.zeros(YY.shape)
    KXY = torch.zeros(XY.shape)

    for sigma in sigmas:
      KXX += gaussian_kernel(XX, sigma)
      KYY += gaussian_kernel(YY, sigma)
      KXY +=  gaussian_kernel(XY, sigma)

    KXX = (KXX.sum() - (len(sigmas) * X.shape[0])) / (len(sigmas) * X.shape[0] * (X.shape[0] - 1))
    KYY = (KYY.sum() - (len(sigmas) * Y.shape[0])) / (len(sigmas) * Y.shape[0] * (Y.shape[0] - 1))
    KXY = KXY.sum() / (len(sigmas) * X.shape[0] * Y.shape[0])

    return KXX + KYY - 2 * KXY

def biased_mmd(X, Y, sigmas=[1, 2, 5, 10, 25]):
    XX = torch.cdist(X, X)
    YY = torch.cdist(Y, Y)
    XY = torch.cdist(X, Y)

    KXX = torch.zeros(XX.shape)
    KYY = torch.zeros(YY.shape)
    KXY = torch.zeros(XY.shape)

    for sigma in sigmas:
      KXX += gaussian_kernel(XX, sigma)
      KYY += gaussian_kernel(YY, sigma)
      KXY +=  gaussian_kernel(XY, sigma)

    KXX = KXX.sum() / (len(sigmas) * X.shape[0]**2 )
    KYY = KYY.sum() / (len(sigmas) * (Y.shape[0]**2))
    KXY = KXY.sum() / (len(sigmas) * (X.shape[0] * Y.shape[0]))

    return KXX + KYY - 2 * KXY


def compute_mmd(trainset_loader, compression_loader, information_dict):
    file_path = "./autoencoders_mmd/"
    if not os.path.isdir(file_path):
        os.mkdir(file_path)

    ae_trainer = get_trainer(accelerator=accelerator, max_epochs=wandb.config.get("max_epochs_AE", 200))
        
    model_name = f"autoencoder_{wandb.config['dataset']}" +\
                    f"{wandb.config['first_class'] + wandb.config['second_class'] if wandb.config['n_classes'] == 2 else ''}" +\
                    f"_{wandb.config['seed']}.ckpt"
    ae_file_path = file_path + model_name
    if os.path.isfile(ae_file_path):
        autoencoder = AutoEncoderModel.load_from_checkpoint(checkpoint_path)
    else:
        autoencoder = create_autoencoder(wandb.config)
        ae_trainer.fit(model=autoencoder, train_dataloaders=trainset_loader)
        ae_trainer.save_checkpoint(ae_file_path)

    encoded_dataset = torch.concat(ae_trainer.predict(model=autoencoder, dataloaders=trainset_loader))
    encoded_compression_set = torch.concat(ae_trainer.predict(model=autoencoder, dataloaders=compression_loader))
    information_dict['mmd_u'] = unbiased_mmd(encoded_dataset, encoded_compression_set).item()
    information_dict['mmd_b'] = biased_mmd(encoded_dataset, encoded_compression_set).item()