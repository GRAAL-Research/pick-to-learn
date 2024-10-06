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
from copy import deepcopy
from pytorch_lightning.loggers import WandbLogger

class CustomCallback(EarlyStopping):
    def __init__(self, monitor, stop):
        super().__init__(monitor=monitor)
        self.monitor = monitor
        self.stop = stop
        self.mode = "="
        self.mode_dict = {"=": torch.eq}

    def _evaluate_stopping_criteria(self, current):
        should_stop = self.monitor_op(current, self.stop)
        return should_stop, None

def baseline(config, name):
    wandb.init(project=name, config=config)
    logger = WandbLogger(project=name, experiment=wandb.run, prefix="(train)")

    seed_everything(wandb.config['seed'], workers=True)
    accelerator = get_accelerator(wandb.config['model_type'])

    # create models, load dataset and split it if necessary
    
    train_set, test_set, collate_fn = load_dataset(wandb.config)
    if wandb.config['prior_size'] != 0.0:
        prior_set, train_set, validation_set = split_prior_train_validation_dataset(train_set, wandb.config['prior_size'], wandb.config['validation_size'])
    else:
        train_set, validation_set = split_train_validation_dataset(train_set, wandb.config['validation_size'])

    trainset_loader = get_dataloader(dataset=train_set, batch_size=wandb.config['batch_size'], collate_fn=collate_fn)
    valset_loader = get_dataloader(dataset=validation_set, batch_size=wandb.config['batch_size'], collate_fn=collate_fn)
    test_loader = get_dataloader(dataset=test_set, batch_size=wandb.config['batch_size'], collate_fn=collate_fn)

    if wandb.config['prior_size'] == 0.0:
         model = create_model(wandb.config)
    else:
        file_path = "./prior_models/"
        if not os.path.isdir(file_path):
            os.mkdir(file_path)
            
        model_name = f"prior_model_{wandb.config['model_type']}_{wandb.config['prior_size']}_{wandb.config['pretraining_lr']}_{wandb.config['pretraining_epochs']}_{wandb.config['seed']}.ckpt"
        file_path = file_path + model_name
        if os.path.isfile(file_path):
            model = load_pretrained_model(file_path, wandb.config)
        else:
            model = create_model(wandb.config)
            prior_loader= get_dataloader(dataset=prior_set, batch_size=wandb.config['batch_size'] , collate_fn=collate_fn)
            prior_trainer = get_trainer(accelerator=accelerator, max_epochs=wandb.config['pretraining_epochs'])
            prior_trainer.fit(model=model, train_dataloaders=prior_loader)
            prior_trainer.save_checkpoint(file_path)

    update_learning_rate(model, wandb.config.get('training_lr', None))
    
    trainer = get_trainer(max_epochs=wandb.config['max_epochs'], 
                          callbacks=[CustomCallback(monitor="validation_error", stop=0.0)], logger=logger)
    trainer.fit(model=model, train_dataloaders=trainset_loader, val_dataloaders=trainset_loader)

    train_results = trainer.validate(model=model, dataloaders=trainset_loader)
    validation_results = trainer.validate(model=model, dataloaders=valset_loader)
    test_results = trainer.test(model, dataloaders=test_loader)

    logger._prefix = ""
    information_dict = {}

    information_dict['train_set_size'] = len(train_set)
    information_dict['validation_set_size'] = len(validation_set)
    information_dict['test_set_size'] = len(test_set)

    information_dict['complement_error'] = train_results[0]['validation_error']
    information_dict['validation_error'] = validation_results[0]['validation_error']
    information_dict['test_error'] = test_results[0]['test_error']

    information_dict['complement_loss'] = train_results[0]['validation_loss']
    information_dict['validation_loss'] = validation_results[0]['validation_loss']
    information_dict['test_loss'] = test_results[0]['test_loss']

    wandb.log(information_dict)
    information_dict['config'] = dict(wandb.config)

    # save the experiment informations in a json
    if not os.path.isdir("./baseline_logs"):
        os.mkdir("./baseline_logs")

    file_name = get_exp_file_name(dict(wandb.config), path="./baseline_logs/")
    with open(file_name, "w") as outfile: 
        json.dump(information_dict, outfile)
    wandb.finish()

def hyperparameter_loop(list_of_sweep_configs, dataset_config):
    for sweep_config_ in list_of_sweep_configs:
        exp_config = sweep_config_ | dataset_config
        config_name = get_exp_file_name(exp_config, path="./baseline_logs/")
        if not os.path.isfile(config_name):
            exp_name = 'baseline_' + dataset_config['dataset']
            if dataset_config.get('n_classes', -1) == 2 and dataset_config['dataset'] == "mnist":
                exp_name += str(dataset_config['first_class']) + str(dataset_config['second_class'])
            baseline(exp_config, name=exp_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sweep_config', type=str, default="pretraining", help="Name of the config for the hyperparameters of the sweep.")
    parser.add_argument('-p', '--params_config', type=str, default="mnist", help="Name of the config for the parameters.")
    parser.add_argument('--online', action='store_true', help="Use if you want to run the code offline with wandb.")
    args = parser.parse_args()

    sweep_config_name = "./configs/sweep_configs/" + args.sweep_config + ".yaml"
    with open(sweep_config_name) as file:
        sweep_configuration = yaml.safe_load(file)

    params_config_name = "./configs/parameter_configs/" + args.params_config + ".yaml"
    with open(params_config_name) as file:
        config = yaml.safe_load(file)

    # correct types of entries to make sure all floats/ints are parsed as such
    for key, value in config.items():
        config[key] = correct_type_of_entry(value)

    if args.online:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="p2l")

        start_sweep = partial(baseline, config=config, name=sweep_configuration['name'])
        wandb.agent(sweep_id, function=start_sweep)
    else:
        list_of_configs = create_all_configs(sweep_configuration)
        if config.get('n_classes', -1) != 2 or config['dataset'] != "mnist":
            hyperparameter_loop(list_of_configs, config)
        else:
            if not isinstance(config['first_class'], list):
                config['first_class'] = [config['first_class']]
                config['second_class'] = [config['second_class']]
            
            for idx in range(len(config['first_class'])):
                new_config = deepcopy(config)
                new_config['first_class'] = config['first_class'][idx]
                new_config['second_class'] = config['second_class'][idx]
                hyperparameter_loop(list_of_configs, new_config)