import torch
from lightning.pytorch import seed_everything
import lightning as L
from utils import *
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from dataset.dataset_loader import load_dataset
import os
import json
import argparse
import wandb
import datetime
from functools import partial
import yaml

def baseline(config):
    wandb.init(project="p2l", config=config)
    seed_everything(42, workers=True)

    # constants to be used later 
    STOP = torch.log(torch.tensor(wandb.config['n_classes']))
    batch_size = wandb.config['batch_size']
    information_dict = {}

    # create models, load dataset and split it if necessary
    model = create_model(wandb.config)
    train_set, test_set = load_dataset(wandb.config)
    train_set, validation_set = split_train_validation_dataset(train_set, wandb.config['validation_size'])

    train_loader= torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=5, persistent_workers=True)
    valset_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False,  num_workers=5, persistent_workers=True)
    callbacks = [EarlyStopping(monitor="validation_loss", mode="min", patience=wandb.config['patience'])]
    trainer = L.Trainer(max_epochs=wandb.config['max_epochs'], callbacks=callbacks)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valset_loader)
    
    # create the dataloaders for the validation and test data. 
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,  num_workers=5, persistent_workers=True)
    train_results = trainer.validate(model=model, dataloaders=train_loader)
    val_results = trainer.validate(model=model, dataloaders=valset_loader)
    test_results = trainer.test(model=model, dataloaders=test_loader)

    information_dict['train_set_size'] = len(train_set)
    information_dict['test_set_size'] = len(test_set)
    information_dict['validation_set_size'] = len(validation_set)

    information_dict['train_error'] = train_results[0]['validation_error']
    information_dict['train_loss'] = train_results[0]['validation_loss']

    information_dict['validation_error'] = val_results[0]['validation_error']
    information_dict['validation_loss'] = val_results[0]['validation_loss']

    information_dict['test_error'] = test_results[0]['test_error']
    information_dict['test_loss'] = test_results[0]['test_loss']

    information_dict['config'] = dict(wandb.config)

    # save the experiment informations in a json
    if not os.path.isdir("./baseline_logs"):
        os.mkdir("./baseline_logs")

    file_name = f"exp_{wandb.config['dataset']}_{wandb.config['model_type']}_{str(datetime.datetime.now()).replace(' ', '_')}.json"
    file_dir = "./baseline_logs/" + file_name
    with open(file_dir, "w") as outfile: 
        json.dump(information_dict, outfile)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baseline_config', type=str, default="default", help="Name of the config for the hyperparameters of the sweep.")
    parser.add_argument('-p', '--params_config', type=str, default="mnist", help="Name of the config for the parameters.")
    args = parser.parse_args()

    baseline_config_name = "./configs/baseline_configs/" + args.baseline_config + ".yaml"
    with open(baseline_config_name) as file:
        sweep_configuration = yaml.safe_load(file)
    
    params_config_name = "./configs/parameter_configs/" + args.params_config + ".yaml"
    with open(params_config_name) as file:
        config = yaml.safe_load(file)

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="p2l")

    start_sweep = partial(baseline, config=config)
    wandb.agent(sweep_id, function=start_sweep)