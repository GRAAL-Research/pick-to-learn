import wandb
from p2l import p2l_algorithm


def main():
    wandb.init(project="p2l")
    p2l_algorithm()
    wandb.finish()

sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

wandb.agent(sweep_id, function=main, count=4)