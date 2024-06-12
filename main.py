import wandb
from p2l import p2l_algorithm
import argparse
import yaml
from functools import partial

def run_sweep(config):
    wandb.init(project="p2l", config=config)
    p2l_algorithm()
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sweep_config', type=str, default="default", help="Name of the config for the hyperparameters of the sweep.")
    parser.add_argument('-p', '--params_config', type=str, default="mnist", help="Name of the config for the parameters.")
    args = parser.parse_args()

    sweep_config_name = "./configs/sweep_configs/" + args.sweep_config + ".yaml"
    with open(sweep_config_name) as file:
        sweep_configuration = yaml.safe_load(file)
    
    params_config_name = "./configs/parameter_configs/" + args.params_config + ".yaml"
    with open(params_config_name) as file:
        config = yaml.safe_load(file)

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="p2l")

    start_sweep = partial(run_sweep, config=config)
    wandb.agent(sweep_id, function=start_sweep)