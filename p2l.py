import torch
from lightning.pytorch import seed_everything
from dataset.mnist import load_binary_mnist
from dataset.cifar10 import load_binary_cifar10
import lightning as L
from utils import get_max_error_idx, CustomDataset, create_model, split_prior_train_dataset
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from bounds.p2l_bounds import compute_all_p2l_bounds
from bounds.real_valued_bounds import compute_real_valued_bounds
from bounds.classical_bounds import compute_classical_compression_bounds
from dataset.dataset_loader import load_dataset
import os
import json
import argparse
import wandb
import datetime

def p2l_algorithm():
    seed_everything(42, workers=True)

    model = create_model(wandb.config)
    
    STOP = -torch.log(torch.tensor(0.5))
    batch_size = wandb.config['batch_size'] 
    n_sigma = 1

    train_set, test_set = load_dataset(wandb.config)
    
    prior_set, train_set = split_prior_train_dataset(train_set, wandb.config['prior_size'])

    if prior_set is not None:
        prior_loader= torch.utils.data.DataLoader(prior_set, batch_size=batch_size, shuffle=False,  num_workers=5, persistent_workers=True)
        prior_trainer = L.Trainer(max_epochs=200)
        prior_trainer.fit(model=model, train_dataloaders=prior_loader)

    information_dict = {}
    information_dict['config'] = dict(wandb.config)
    information_dict['train_set_size'] = len(train_set)
    information_dict['test_set_size'] = len(test_set)

    val_dataset_idx = torch.arange(len(train_set))
    
    predict_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False,  num_workers=5, persistent_workers=True)
    prediction_trainer = L.Trainer(max_epochs=1)
    errors = prediction_trainer.predict(model=model, dataloaders=predict_loader)

    z, idx = get_max_error_idx(errors)
    
    compression_set_idx = torch.tensor([idx])
    len_val_idx = val_dataset_idx.shape[0]
    val_dataset_idx = val_dataset_idx[val_dataset_idx != idx]

    assert len_val_idx -1 == val_dataset_idx.shape[0]

    i = 0
    max_iterations = len(train_set) if wandb.config['max_iterations'] == -1 else wandb.config['max_iterations']

    while STOP <= z and i < max_iterations:
        compression_set = CustomDataset(train_set.data[compression_set_idx], train_set.targets[compression_set_idx])
        compression_loader = torch.utils.data.DataLoader(compression_set, batch_size=batch_size,  num_workers=5, persistent_workers=True)
        trainer = L.Trainer(max_epochs=200, callbacks=[EarlyStopping(monitor="train_loss", mode="min", patience=wandb.config['patience'])])
        trainer.fit(model=model, train_dataloaders=compression_loader)

        predict_set = CustomDataset(train_set.data[val_dataset_idx], train_set.targets[val_dataset_idx])
        predict_loader = torch.utils.data.DataLoader(predict_set, batch_size=batch_size, shuffle=False, num_workers=5, persistent_workers=True)
        errors = prediction_trainer.predict(model=model, dataloaders=predict_loader)
        z, idx = get_max_error_idx(errors)

        wandb.log({'max_error': z})
        # we have the index in the modified dataset, we need to normalize it to be the one in the original dataset
        idx = val_dataset_idx[idx]
        i += 1 
        if i % wandb.config['log_iterations'] == 0:
            val_res = prediction_trainer.validate(model=model, dataloaders=predict_loader)
            
            metrics = {'val_error' : val_res[0]['validation_error']}
            compute_real_valued_bounds(len(compression_set),
                                        n_sigma,
                                        len(predict_set),
                                        val_res[0]['validation_error'],
                                        wandb.config['delta'],
                                        metrics)
            wandb.log(metrics)
        if STOP <= z and i < max_iterations:
            print(z, len(compression_set_idx))
            len_comp = compression_set_idx.shape[0]
            len_val = val_dataset_idx.shape[0]

            compression_set_idx = torch.cat((compression_set_idx, torch.tensor([idx])))
            val_dataset_idx = val_dataset_idx[val_dataset_idx != idx]

            assert len_comp + 1 == compression_set_idx.shape[0]
            assert len_val - 1 == val_dataset_idx.shape[0]
        else:
            print(f"P2l ended with max error {z.item():.2f} and a compression set of size {len(compression_set)}")


    validation_set = CustomDataset(train_set.data[val_dataset_idx], train_set.targets[val_dataset_idx])
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=5, persistent_workers=True)
    validation_results = prediction_trainer.validate(model, dataloaders=validation_loader)

    information_dict['val_error'] = validation_results[0]['validation_error']
    information_dict['val_loss'] = validation_results[0]['validation_loss']

    test_dataset = CustomDataset(test_set.data, test_set.targets)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,  num_workers=5, persistent_workers=True)
    test_results = prediction_trainer.test(model, dataloaders=test_loader)

    information_dict['test_error'] = test_results[0]['test_error']
    information_dict['test_loss'] = test_results[0]['test_loss']

    n = len(train_set)
    m = compression_set_idx.shape[0]
    information_dict['compression_set_size'] = m

    val_error = validation_results[0]['validation_error']
    k = val_error * (n-m)

    print(("-"*20) + " Classical compression bounds " + "-"*20)
    compute_classical_compression_bounds(m, n_sigma, n, k, wandb.config['delta'], information_dict)
    print(("-"*20) + " Pick-to-learn bounds " + "-"*20)
    compute_all_p2l_bounds(m, n, wandb.config['delta'], information_dict)
    print(("-"*20) + " Real valued bounds " + "-"*20)
    compute_real_valued_bounds(m, n_sigma, n, val_error, wandb.config['delta'], information_dict)

    wandb.log(information_dict)

    if not os.path.isdir("./experiment_logs"):
        os.mkdir("./experiment_logs")

    file_name = f"exp_{wandb.config['dataset']}_{wandb.config['model_type']}_{str(datetime.datetime.now()).replace(' ', '_')}.json"
    file_dir = "./experiment_logs/" + file_name
    with open(file_dir, "w") as outfile: 
        json.dump(information_dict, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--optimizer', type=str, default="Adam")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-mm', '--momentum', type=float, default=0.95)
    parser.add_argument('-m', '--model_type', type=str, default="cnn")
    parser.add_argument('-mx', '--max_iterations', type=int, default=-1)
    parser.add_argument('-f', '--first_class', type=int, default=1)
    parser.add_argument('-s', '--second_class', type=int, default=7)
    parser.add_argument('-d', '--dataset', type=str, default="mnist")
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-del', '--delta', type=float, default=0.01)
    parser.add_argument('-dp', '--dropout_probability', type=float, default=0.2)
    parser.add_argument('-p', '--prior_size', type=float, default=0.0)
    parser.add_argument('-lg', '--log_iterations', type=int, default=50)
    parser.add_argument('-pt', '--patience', type=int, default=3)
    parser.add_argument('-nc', '--n_classes', type=int, default=2)
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()
    
    wandb.init(project="p2l", config=args)
    p2l_algorithm()
    wandb.finish()