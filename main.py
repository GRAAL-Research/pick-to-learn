import wandb
from p2l import p2l_algorithm
import argparse
import yaml
from functools import partial
from utils import create_all_configs, get_exp_file_name, correct_type_of_entry
import os
from copy import deepcopy

def run_sweep(config, name='p2l'):
    wandb.init(project=name, config=config)
    p2l_algorithm()
    wandb.finish()

def hyperparameter_loop(list_of_sweep_configs, dataset_config):
    for sweep_config_ in list_of_sweep_configs:
        exp_config = sweep_config_ | dataset_config
        config_name = get_exp_file_name(exp_config)
        if not os.path.isfile(config_name):
            exp_name = sweep_configuration['name'] + dataset_config['dataset']
            if dataset_config.get('n_classes', -1) == 2:
                exp_name += str(dataset_config['first_class']) + str(dataset_config['second_class'])
            run_sweep(exp_config, name=exp_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sweep_config', type=str, default="tree_based", help="Name of the config for the hyperparameters of the sweep.")
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

        start_sweep = partial(run_sweep, config=config, name=sweep_configuration['name'])
        wandb.agent(sweep_id, function=start_sweep)
    else:
        list_of_configs = create_all_configs(sweep_configuration)
        if config.get('regression', False) or config.get('n_classes', -1) != 2 :
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