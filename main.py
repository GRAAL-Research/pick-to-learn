import wandb
from p2l import p2l_algorithm
import argparse
import yaml
from functools import partial
from utils import create_all_configs, get_exp_file_name
import os

def run_sweep(config, name='p2l'):
    wandb.init(project=name, config=config)
    p2l_algorithm()
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sweep_config', type=str, default="default", help="Name of the config for the hyperparameters of the sweep.")
    parser.add_argument('-p', '--params_config', type=str, default="mnist", help="Name of the config for the parameters.")
    parser.add_argument('--online', action='store_true', help="Use if you want to run the code offline with wandb.")
    args = parser.parse_args()

    sweep_config_name = "./configs/sweep_configs/" + args.sweep_config + ".yaml"
    with open(sweep_config_name) as file:
        sweep_configuration = yaml.safe_load(file)

    params_config_name = "./configs/parameter_configs/" + args.params_config + ".yaml"
    with open(params_config_name) as file:
        config = yaml.safe_load(file)

    if args.online:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="p2l")

        start_sweep = partial(run_sweep, config=config, name=sweep_configuration['name'])
        wandb.agent(sweep_id, function=start_sweep)
    else:
        list_of_configs = create_all_configs(sweep_configuration)
        for sweep_config_ in list_of_configs:
            config_name = get_exp_file_name(sweep_config_)
            if not os.path.isfile(config_name):
                exp_name = sweep_configuration['name'] + config['dataset']+ str(config['first_class']) + str(config['second_class'])
                run_sweep(sweep_config_ | config, name=exp_name)