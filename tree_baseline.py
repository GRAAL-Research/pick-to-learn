from lightning.pytorch import seed_everything
from utils import *
from dataset.dataset_loader import load_dataset
import os
import json
import argparse
import wandb
from functools import partial
import yaml

def baseline(config, name):
    wandb.init(project=name, config=config)
    seed_everything(wandb.config['seed'], workers=True)

    # create models, load dataset and split it if necessary
    
    train_set, test_set, _ = load_dataset(wandb.config)
    train_set, validation_set = split_train_validation_dataset(train_set, wandb.config['validation_size'])

    trainset_loader = get_dataloader(dataset=train_set,
                                      batch_size=get_updated_batch_size(None, wandb.config['model_type'], len(train_set)))
    valset_loader = get_dataloader(dataset=validation_set,
                                batch_size=get_updated_batch_size(None, wandb.config['model_type'], len(validation_set)))
    test_loader = get_dataloader(dataset=test_set,
                                batch_size=get_updated_batch_size(None, wandb.config['model_type'], len(test_set)))

    model = create_model(wandb.config)

    trainer = get_trainer(accelerator='cpu', max_epochs=1)

    trainer.fit(model=model, train_dataloaders=trainset_loader, val_dataloaders=valset_loader)   

    train_results = trainer.validate(model=model, dataloaders=trainset_loader)
    validation_results = trainer.validate(model=model, dataloaders=valset_loader)
    test_results = trainer.test(model, dataloaders=test_loader)

    information_dict = {}

    information_dict['train_set_size'] = len(train_set)
    information_dict['validation_set_size'] = len(validation_set)
    information_dict['test_set_size'] = len(test_set)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sweep_config', type=str, default="tree", help="Name of the config for the hyperparameters of the sweep.")
    parser.add_argument('-p', '--params_config', type=str, default="airfoil", help="Name of the config for the parameters.")
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
        for sweep_config_ in list_of_configs:
            exp_config = sweep_config_ | config
            config_name = get_exp_file_name(exp_config, path="./baseline_logs/")
            if not os.path.isfile(config_name):
                exp_name = "baseline_" + config['dataset']
                baseline(exp_config, name=exp_name)