import torch
from lightning.pytorch import seed_everything
import lightning as L
from utils import get_max_error_idx, CustomDataset, create_model, split_prior_train_dataset, CompressionSetIndexes, update_learning_rate
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
    
    STOP = torch.log(torch.tensor(wandb.config['n_classes']))
    batch_size = wandb.config['batch_size'] 
    n_sigma = 1

    train_set, test_set = load_dataset(wandb.config)
    
    prior_set, train_set = split_prior_train_dataset(train_set, wandb.config['prior_size'])

    if prior_set is not None:
        prior_loader= torch.utils.data.DataLoader(prior_set, batch_size=batch_size, shuffle=False,  num_workers=5, persistent_workers=True)
        prior_trainer = L.Trainer(max_epochs=wandb.config['pretraining_epochs'])
        prior_trainer.fit(model=model, train_dataloaders=prior_loader)

    information_dict = {}
    information_dict['config'] = dict(wandb.config)
    information_dict['train_set_size'] = len(train_set)
    information_dict['test_set_size'] = len(test_set)

    dataset_idx = CompressionSetIndexes(len(train_set))
    
    predict_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False,  num_workers=5, persistent_workers=True)
    prediction_trainer = L.Trainer(max_epochs=1)
    errors = prediction_trainer.predict(model=model, dataloaders=predict_loader)

    z, idx = get_max_error_idx(errors, wandb.config['block_training'])
    
    idx = dataset_idx.correct_idx(idx)

    # Updates the lr, as it might not be the same in the pretraining and training
    update_learning_rate(model, wandb.config['learning_rate'])
    dataset_idx.update_compression_set(idx)

    i = 0
    max_iterations = len(train_set) if wandb.config['max_iterations'] == -1 else wandb.config['max_iterations']

    while STOP <= z and i < max_iterations:
        compression_set = CustomDataset(train_set.data, train_set.targets, indices=dataset_idx.get_compression_set())
        compression_loader = torch.utils.data.DataLoader(compression_set, batch_size=batch_size,  num_workers=5, persistent_workers=True)
        trainer = L.Trainer(max_epochs=wandb.config['max_epochs'], callbacks=[EarlyStopping(monitor="train_loss", mode="min", patience=wandb.config['patience'])])
        trainer.fit(model=model, train_dataloaders=compression_loader)

        predict_set = CustomDataset(train_set.data, train_set.targets, indices=dataset_idx.get_validation_set())
        predict_loader = torch.utils.data.DataLoader(predict_set, batch_size=batch_size, shuffle=False, num_workers=5, persistent_workers=True)
        errors = prediction_trainer.predict(model=model, dataloaders=predict_loader)
        z, idx = get_max_error_idx(errors, wandb.config['block_training'])

        idx = dataset_idx.correct_idx(idx)

        wandb.log({'max_error': z})
        i += 1 
        if i % wandb.config['log_iterations'] == 0:
            val_res = prediction_trainer.validate(model=model, dataloaders=predict_loader)
            
            metrics = {'val_error' : val_res[0]['validation_error']}
            compute_real_valued_bounds(len(compression_set),
                                        n_sigma,
                                        len(predict_set),
                                        val_res[0]['validation_error'],
                                        wandb.config['delta'],
                                        wandb.config['nbr_parameter_bounds'],
                                        metrics)
            wandb.log(metrics)
        if STOP <= z and i < max_iterations:
            print(z.item(), len(compression_set))
            dataset_idx.update_compression_set(idx)
        else:
            print(f"P2l ended with max error {z.item():.2f} and a compression set of size {len(compression_set)}")


    validation_set = CustomDataset(train_set.data, train_set.targets, indices=dataset_idx.get_validation_set())
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
    m = int(dataset_idx.get_compression_set_length())
    information_dict['compression_set_size'] = m

    val_error = validation_results[0]['validation_error']
    k = val_error * (n-m)

    if wandb.config['classic_bounds']:   
        print(("-"*20) + " Classical compression bounds " + "-"*20)
        compute_classical_compression_bounds(m, n_sigma, n, k, wandb.config['delta'], information_dict)

    if wandb.config['p2l_bounds']:
        print(("-"*20) + " Pick-to-learn bounds " + "-"*20)
        compute_all_p2l_bounds(m, n, wandb.config['delta'], information_dict)
    
    if wandb.config['real_bounds']:
        print(("-"*20) + " Real valued bounds " + "-"*20)
        compute_real_valued_bounds(m, n_sigma, n, val_error, wandb.config['delta'], wandb.config['nbr_parameter_bounds'], information_dict)

    wandb.log(information_dict)

    if not os.path.isdir("./experiment_logs"):
        os.mkdir("./experiment_logs")

    file_name = f"exp_{wandb.config['dataset']}_{wandb.config['model_type']}_{str(datetime.datetime.now()).replace(' ', '_')}.json"
    file_dir = "./experiment_logs/" + file_name
    with open(file_dir, "w") as outfile: 
        json.dump(information_dict, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset details 
    parser.add_argument('-d', '--dataset', type=str, default="cifar10", help="Name of the dataset.")
    parser.add_argument('-nc', '--n_classes', type=int, default=2, help="Number of classes used in the training set.")
    parser.add_argument('-f', '--first_class', type=int, default=1, help="When the problem is binary classification, the first class used in the training set.")
    parser.add_argument('-s', '--second_class', type=int, default=7, help="When the problem is binary classification, the second class used in the training set.")

    # pretraining details
    parser.add_argument('-p', '--prior_size', type=float, default=0.01, help="Portion of the training set that is used to pre-train the model.")
    parser.add_argument('-pti', '--pretraining_epochs', type=int, default=100, help="Number of epochs used to pretrain the model.")
    parser.add_argument('-plr', '--pretraining_lr', type=int, default=100, help="Learning rate used by the optimizer to pretrain the model.")

    # training details
    parser.add_argument('-m', '--model_type', type=str, default="cnn", help="Type of model to train.")
    parser.add_argument('-me', '--max_epochs', type=int, default=200, help="Maximum number of epochs to train the model at each step of P2L.")
    parser.add_argument('-b', '--batch_size', type=int, default=64, help="Batch size used to train the model.")
    parser.add_argument('-dp', '--dropout_probability', type=float, default=0.2, help="Dropout probability for the layers of the model.")

    # optimizer
    parser.add_argument('-o', '--optimizer', type=str, default="Adam", help="Optimizer used to train the model.")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2, help="Learning rate used to train the model.")
    parser.add_argument('-mm', '--momentum', type=float, default=0.95, help="Momentum used by the SGD optimizer. This parameter is not used by Adam.")

    # p2l params
    parser.add_argument('-mx', '--max_iterations', type=int, default=-1, help="Maximum number of iterations of the P2L algorithm.")
    parser.add_argument('-bt', '--block_training', type=int, default=32, help="Number of data added to the compression set at each iterations.")
    parser.add_argument('-pt', '--patience', type=int, default=3, help="Patience of the EarlyStopping Callback used to train on the compression set.")
    
    # bound parameters
    parser.add_argument('-del', '--delta', type=float, default=0.01, help="Delta used to compute the bounds.")
    parser.add_argument('-npb', '--nbr_parameter_bounds', type=int, default=20, help="Number of parameters used to compute the Catoni and linear bounds.")
    parser.add_argument('--classic_bounds', action='store_false', help="Use if you do not want to compute the classic bounds.")
    parser.add_argument('--p2l_bounds', action='store_false', help="Use if you do not want to compute the P2L bounds.")
    parser.add_argument('--real_bounds', action='store_false', help="Use if you do not want to compute the real valued bounds.")

    # miscellaneous
    parser.add_argument('-lg', '--log_iterations', type=int, default=5, help="Log the real valued bounds after this number of iterations.")
    parser.add_argument('--deterministic', action='store_true',
                            help="Use if you want reproducible results, but not when you want to test the code, as it slows down the code.")

    args = parser.parse_args()
    
    wandb.init(project="p2l", config=args)
    p2l_algorithm()
    wandb.finish()