import torch
from lightning.pytorch import seed_everything
import lightning as L
from utils import get_max_error_idx, CustomDataset, create_model, split_prior_train_validation_dataset, CompressionSetIndexes, update_learning_rate
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

    # constants to be used later 
    STOP = torch.log(torch.tensor(wandb.config['n_classes']))
    batch_size = wandb.config['batch_size'] 
    n_sigma = 1
    information_dict = {}

    # create models, load dataset and split it if necessary
    model = create_model(wandb.config)
    train_set, test_set = load_dataset(wandb.config)
    prior_set, train_set, validation_set = split_prior_train_validation_dataset(train_set, wandb.config['prior_size'], wandb.config['validation_size'])

    # if there is pretraining, train the model on the prior set.
    if wandb.config['prior_size'] != 0.0:
        prior_loader= torch.utils.data.DataLoader(prior_set, batch_size=batch_size, shuffle=False,  num_workers=5, persistent_workers=True)
        prior_trainer = L.Trainer(max_epochs=wandb.config['pretraining_epochs'])
        prior_trainer.fit(model=model, train_dataloaders=prior_loader)

    # Instantiate the mask that will deal with the indexes
    dataset_idx = CompressionSetIndexes(len(train_set))
    
    # create the dataloaders for the validation and test data. 
    valset_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False,  num_workers=5, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,  num_workers=5, persistent_workers=True)

    # Forward pass of prediction to find on which data we do the most error
    trainset_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False,  num_workers=5, persistent_workers=True)
    prediction_trainer = L.Trainer()
    errors = prediction_trainer.predict(model=model, dataloaders=trainset_loader)
    z, idx = get_max_error_idx(errors, wandb.config['data_groupsize'])
    
    # We need to correct the indices, as the continuously changing indices cause index shift
    idx = dataset_idx.correct_idx(idx)

    # Updates the lr, as it might not be the same in the pretraining and training
    update_learning_rate(model, wandb.config['training_lr'])

    max_compression_size = len(train_set) if wandb.config['max_compression_size'] == -1 else wandb.config['max_compression_size']

    compression_set_size = dataset_idx.get_compression_size()

    # main loop of p2l
    while STOP <= z and compression_set_size < max_compression_size:
        print(z.item(), compression_set_size)

        # update the compression set
        dataset_idx.update_compression_set(idx)
        compression_set_size = dataset_idx.get_compression_size()

        # train on the compression set
        compression_set = CustomDataset(train_set.data, train_set.targets, indices=dataset_idx.get_compression_data())
        compression_loader = torch.utils.data.DataLoader(compression_set, batch_size=batch_size,  num_workers=5, persistent_workers=True)
        trainer = L.Trainer(max_epochs=wandb.config['max_epochs'], callbacks=[EarlyStopping(monitor="train_loss", mode="min", patience=wandb.config['patience'])])
        trainer.fit(model=model, train_dataloaders=compression_loader)

        # predict on the complement set
        complement_set = CustomDataset(train_set.data, train_set.targets, indices=dataset_idx.get_complement_data())
        complement_loader = torch.utils.data.DataLoader(complement_set, batch_size=batch_size, shuffle=False, num_workers=5, persistent_workers=True)
        errors = prediction_trainer.predict(model=model, dataloaders=complement_loader)
        z, idx = get_max_error_idx(errors, wandb.config['data_groupsize'])

        # We need to correct the indices, as the continuously changing indices cause index shift
        idx = dataset_idx.correct_idx(idx)
        
        wandb.log({'max_error': z})
        # On va tester le modèle sur le complement et validation set, ainsi que calculer les bornes
        # On met le -1 pour qu'il log à l'itération 0, puis à toutes les log_iterations
        if compression_set_size - 1 % (wandb.config['data_groupsize'] * wandb.config['log_iterations']) == 0:
            complement_res = prediction_trainer.validate(model=model, dataloaders=complement_loader)
            validation_res = prediction_trainer.validate(model=model, dataloaders=valset_loader)
            metrics = {'complement_error' : complement_res[0]['validation_error'], 'val_error': validation_res[0]['validation_error']}

            compute_real_valued_bounds(len(compression_set),
                                        n_sigma,
                                        len(complement_set),
                                        complement_res[0]['validation_error'],
                                        wandb.config['delta'],
                                        wandb.config['nbr_parameter_bounds'],
                                        metrics)
            wandb.log(metrics)


    print(f"P2l ended with max error {z.item():.2f} and a compression set of size {compression_set_size}")

    # Test the model on the complement set
    complement_set = CustomDataset(train_set.data, train_set.targets, indices=dataset_idx.get_complement_data())
    complement_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=5, persistent_workers=True)
    complement_results = prediction_trainer.validate(model, dataloaders=complement_loader)

    # Test the model on the validation and test sets
    validation_results = prediction_trainer.validate(model, dataloaders=valset_loader)
    test_results = prediction_trainer.test(model, dataloaders=test_loader)

    # log informations
    information_dict['train_set_size'] = len(train_set)
    information_dict['test_set_size'] = len(test_set)
    information_dict['compression_set_size'] = compression_set_size

    information_dict['complement_error'] = complement_results[0]['validation_error']
    information_dict['complement_loss'] = complement_results[0]['validation_loss']

    information_dict['validation_error'] = validation_results[0]['validation_error']
    information_dict['validation_loss'] = validation_results[0]['validation_loss']

    information_dict['test_error'] = test_results[0]['test_error']
    information_dict['test_loss'] = test_results[0]['test_loss']

    # compute the bounds
    n = len(train_set)
    val_error = complement_results[0]['validation_error']
    k = val_error * (n-compression_set_size)

    if wandb.config['classic_bounds']:   
        print(("-"*20) + " Classical compression bounds " + "-"*20)
        compute_classical_compression_bounds(compression_set_size, n_sigma, n, k, wandb.config['delta'], information_dict)

    if wandb.config['p2l_bounds']:
        print(("-"*20) + " Pick-to-learn bounds " + "-"*20)
        compute_all_p2l_bounds(compression_set_size, n, wandb.config['delta'], information_dict)
    
    if wandb.config['real_bounds']:
        print(("-"*20) + " Real valued bounds " + "-"*20)
        compute_real_valued_bounds(compression_set_size, n_sigma, n, val_error, wandb.config['delta'], wandb.config['nbr_parameter_bounds'], information_dict)

    wandb.log(information_dict)

    information_dict['config'] = dict(wandb.config)

    # save the experiment informations in a json
    if not os.path.isdir("./experiment_logs"):
        os.mkdir("./experiment_logs")

    file_name = f"exp_{wandb.config['dataset']}_{wandb.config['model_type']}_{str(datetime.datetime.now()).replace(' ', '_')}.json"
    file_dir = "./experiment_logs/" + file_name
    with open(file_dir, "w") as outfile: 
        json.dump(information_dict, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset details 
    parser.add_argument('-d', '--dataset', type=str, default="mnist", help="Name of the dataset.")
    parser.add_argument('-nc', '--n_classes', type=int, default=2, help="Number of classes used in the training set.")
    parser.add_argument('-f', '--first_class', type=int, default=1, help="When the problem is binary classification, the first class used in the training set.")
    parser.add_argument('-s', '--second_class', type=int, default=7, help="When the problem is binary classification, the second class used in the training set.")

    # pretraining details
    parser.add_argument('-p', '--prior_size', type=float, default=0.0, help="Portion of the training set that is used to pre-train the model.")
    parser.add_argument('-v', '--validation_size', type=float, default=0.1, help="Portion of the dataset that is used to validate the model.")
    parser.add_argument('-pti', '--pretraining_epochs', type=int, default=50, help="Number of epochs used to pretrain the model.")
    parser.add_argument('-plr', '--pretraining_lr', type=float, default=1e-3, help="Learning rate used by the optimizer to pretrain the model.")

    # training details
    parser.add_argument('-m', '--model_type', type=str, default="cnn", help="Type of model to train.")
    parser.add_argument('-me', '--max_epochs', type=int, default=200, help="Maximum number of epochs to train the model at each step of P2L.")
    parser.add_argument('-b', '--batch_size', type=int, default=64, help="Batch size used to train the model.")
    parser.add_argument('-dp', '--dropout_probability', type=float, default=0.2, help="Dropout probability for the layers of the model.")

    # optimizer
    parser.add_argument('-o', '--optimizer', type=str, default="Adam", help="Optimizer used to train the model.")
    parser.add_argument('-lr', '--training_lr', type=float, default=1e-3, help="Learning rate used to train the model.")
    parser.add_argument('-mm', '--momentum', type=float, default=0.95, help="Momentum used by the SGD optimizer. This parameter is not used by Adam.")
    parser.add_argument('--nesterov', action='store_false', help="If the SGD optimizer should use Nesterov acceleration.")

    # p2l params
    parser.add_argument('-mx', '--max_compression_size', type=int, default=2,
                     help="Maximum size of the compression set added by the P2L algorithm. -1 if everything can be added")
    parser.add_argument('-dg', '--data_groupsize', type=int, default=1, help="Number of data added to the compression set at each iterations.")
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