import torch
from lightning.pytorch import seed_everything
import lightning as L
from utilities.utils import *
from utilities.utils_compression_set import *
from utilities.utils_datasets import *
from utilities.utils_early_stopping import *
from utilities.utils_models import *
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from bounds.p2l_bounds import compute_all_p2l_bounds
from bounds.real_valued_bounds import compute_real_valued_bounds
from bounds.classical_bounds import compute_classical_compression_bounds
from dataset.dataset_loader import load_dataset
import os
import json
import argparse
import wandb
import numpy as np

def p2l_algorithm():
    seed_everything(wandb.config['seed'], workers=True)

    # constants to be used later 
    STOP = torch.log(torch.tensor(2))
    batch_size = wandb.config['batch_size'] 
    n_sigma = 1
    information_dict = {}
    accelerator = get_accelerator(wandb.config['model_type'])

    ######################## DATASET ########################
    # Load dataset and split it if necessary
    train_set, test_set, collate_fn = load_dataset(wandb.config)

    # if there is pretraining, train the model on the prior set.
    if wandb.config['prior_size'] != 0.0:
        prior_set, train_set, validation_set = split_prior_train_validation_dataset(train_set, wandb.config['prior_size'], wandb.config['validation_size'])
    else:
        train_set, validation_set = split_train_validation_dataset(train_set, wandb.config['validation_size'])

    # We check everything is a CustomDataset and will work correctly
    assert isinstance(train_set, CustomDataset)
    assert isinstance(validation_set, CustomDataset)
    assert isinstance(test_set, CustomDataset)

    # Instantiate the mask that will deal with the indexes
    dataset_idx = CompressionSetIndexes(len(train_set))
    
    # create the dataloaders for the validation and test data. 
    trainset_loader = get_dataloader(dataset=train_set, batch_size=batch_size, collate_fn=collate_fn)
    valset_loader = get_dataloader(dataset=validation_set, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = get_dataloader(dataset=test_set, batch_size=batch_size, collate_fn=collate_fn)

    ######################## MODEL ########################
    if wandb.config['prior_size'] == 0.0:
         model = create_model(wandb.config)
    else:
        file_path = "./prior_models/"
        if not os.path.isdir(file_path):
            os.mkdir(file_path)
            
        model_name = f"prior_model_{wandb.config['model_type']}_{wandb.config['prior_size']}_{wandb.config['pretraining_lr']}_{wandb.config['pretraining_epochs']}_{wandb.config['n_classes']}_{wandb.config['seed']}.ckpt"
        file_path = file_path + model_name
        if os.path.isfile(file_path):
            model = load_pretrained_model(file_path, wandb.config)
        else:
            model = create_model(wandb.config)
            prior_loader= get_dataloader(dataset=prior_set, batch_size=batch_size, collate_fn=collate_fn)
            prior_trainer = get_trainer(accelerator=accelerator, max_epochs=wandb.config['pretraining_epochs'])
            prior_trainer.fit(model=model, train_dataloaders=prior_loader)
            prior_trainer.save_checkpoint(file_path)
        
    # Updates the lr, as it might not be the same in the pretraining and training
    update_learning_rate(model, wandb.config.get('training_lr', None))
    if wandb.config['clamping']:
        add_clamping_to_model(model, config=wandb.config)
    
    # Forward pass of prediction to find on which data we do the most error
    prediction_trainer = get_trainer(accelerator=accelerator)
    validation_loss = log_metrics(prediction_trainer,
                model,
                trainset_loader,
                valset_loader,
                test_loader,
                0,
                len(train_set),
                n_sigma,
                return_validation_loss=True)
    errors = prediction_trainer.predict(model=model, dataloaders=trainset_loader)
    z, idx = get_max_error_idx(errors, wandb.config['data_groupsize'])
    
    # We need to correct the indices, as the continuously changing indices cause index shift
    idx = dataset_idx.correct_idx(idx)

    max_compression_size = len(train_set) if wandb.config['max_compression_size'] == -1 else wandb.config['max_compression_size']
    early_stopper = StoppingCriterion(max_compression_size,
                                    stop_criterion=STOP,
                                    patience=wandb.config['early_stopping_patience'],
                                    use_early_stopping=wandb.config['early_stopping'],
                                    use_p2l_stopping=not wandb.config['regression'])
    
    compression_set_size = dataset_idx.get_compression_size()

    early_stopper.check_stop(loss=validation_loss, max_error=z, compression_set_size=compression_set_size)
    # main loop of p2l
    while not early_stopper.stop:
        print(z.item(), compression_set_size)

        # update the compression set
        dataset_idx.update_compression_set(idx)
        compression_set_size = dataset_idx.get_compression_size()

        # train on the compression set
        compression_set = train_set.clone_dataset(dataset_idx.get_compression_data())
        
        compression_loader = get_dataloader(dataset=compression_set,
                             batch_size=get_updated_batch_size(batch_size, wandb.config['model_type'], len(compression_set)),
                             collate_fn=collate_fn)
        trainer = get_trainer(accelerator=accelerator,
                            max_epochs=wandb.config['max_epochs'],
                            callbacks=[EarlyStopping(monitor="validation_loss", mode="min", patience=wandb.config['patience'])])

        trainer.fit(model=model, train_dataloaders=compression_loader, val_dataloaders=valset_loader)   

        # predict on the complement set
        complement_set = train_set.clone_dataset(dataset_idx.get_complement_data())
        complement_loader = get_dataloader(dataset=complement_set, batch_size=batch_size, collate_fn=collate_fn)
        errors = prediction_trainer.predict(model=model, dataloaders=complement_loader)
        z, idx = get_max_error_idx(errors, wandb.config['data_groupsize'])

        # We need to correct the indices, as the continuously changing indices cause index shift
        idx = dataset_idx.correct_idx(idx)
        
        wandb.log({'max_error': z})

        early_stopper.check_stop(loss=trainer.callback_metrics['validation_loss'],
                                max_error=z,
                                compression_set_size=compression_set_size)

        # On va tester le modèle sur le complement et validation set, ainsi que calculer les bornes
        # On met le -1 pour qu'il log à l'itération 0, puis à toutes les log_iterations
        if (compression_set_size ) % (wandb.config['data_groupsize'] * wandb.config['log_iterations']) == 0:
            log_metrics(prediction_trainer,
                        model,
                        complement_loader,
                        valset_loader,
                        test_loader,
                        len(compression_set),
                        len(train_set),
                        n_sigma)


    print(f"P2l ended with max error {z.item():.2f} and a compression set of size {compression_set_size}")

    # Test the model on the complement set
    complement_set = train_set.clone_dataset(dataset_idx.get_complement_data())
    complement_loader = get_dataloader(dataset=complement_set, batch_size=batch_size, collate_fn=collate_fn)
    complement_results = prediction_trainer.validate(model, dataloaders=complement_loader)

    # Test the model on the validation and test sets
    validation_results = prediction_trainer.validate(model, dataloaders=valset_loader)
    test_results = prediction_trainer.test(model, dataloaders=test_loader)

    # log informations
    if wandb.config['prior_size'] != 0.0:
        information_dict['prior_set_size'] = len(prior_set)
    information_dict['train_set_size'] = len(train_set)
    information_dict['validation_set_size'] = len(validation_set)
    information_dict['test_set_size'] = len(test_set)
    information_dict['compression_set_size'] = compression_set_size

    if not wandb.config['regression']:
        information_dict['complement_error'] = complement_results[0]['validation_error']
        information_dict['validation_error'] = validation_results[0]['validation_error']
        information_dict['test_error'] = test_results[0]['test_error']
    
    information_dict['complement_loss'] = complement_results[0]['validation_loss']
    information_dict['validation_loss'] = validation_results[0]['validation_loss']
    information_dict['test_loss'] = test_results[0]['test_loss']

    # compute the bounds
    n = len(train_set)
    
    if wandb.config['classic_bounds']:   
        print(("-"*20) + " Classical compression bounds " + "-"*20)
        k = complement_results[0]['validation_error'] * (n-compression_set_size)
        compute_classical_compression_bounds(compression_set_size, n_sigma, n, k, wandb.config['delta'], information_dict)

    if wandb.config['p2l_bounds']:
        print(("-"*20) + " Pick-to-learn bounds " + "-"*20)
        compute_all_p2l_bounds(compression_set_size, n, wandb.config['delta'], information_dict)
    
    if wandb.config['real_bounds']:
        print(("-"*20) + " Real valued bounds " + "-"*20)

        if wandb.config['regression']:
            compute_real_valued_bounds(compression_set_size, n_sigma, n, complement_results[0]['validation_loss'],
                                        wandb.config['delta'], wandb.config['nbr_parameter_bounds'], information_dict, 
                                        min_val=wandb.config['min_val'], max_val=wandb.config['max_val'])
        else:
            compute_real_valued_bounds(compression_set_size, n_sigma, n, complement_results[0]['validation_error'],
                                        wandb.config['delta'], wandb.config['nbr_parameter_bounds'], information_dict)
            if wandb.config['clamping']:
                print(("-"*20) + " Real valued bounds for bounded cross entropy" + "-"*20)
                compute_real_valued_bounds(compression_set_size, n_sigma, n, complement_results[0]['validation_loss'], wandb.config['delta'],
                                            wandb.config['nbr_parameter_bounds'], information_dict, min_val=0, 
                                            max_val=-np.log(wandb.config['min_probability']), prefix="CE")

    wandb.log(information_dict)

    information_dict['config'] = dict(wandb.config)
    
    if wandb.config.get("estimate_mmd", False):
        compute_mmd(trainset_loader, compression_loader, information_dict)
        
    # save the experiment informations in a json
    if not os.path.isdir("./experiment_logs"):
        os.mkdir("./experiment_logs")

    file_dir = get_exp_file_name(information_dict['config'])
    with open(file_dir, "w") as outfile: 
        json.dump(information_dict, outfile)

    if not os.path.isdir("./compression_sets_log"):
        os.mkdir("./compression_sets_log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset details 
    parser.add_argument('-d', '--dataset', type=str, default="mnist", help="Name of the dataset.")
    parser.add_argument('-r', '--regression', action='store_true', help="If the dataset is a regression problem.")
    parser.add_argument('-nc', '--n_classes', type=int, default=2, help="Number of classes used in the training set.")
    parser.add_argument('-f', '--first_class', type=int, default=1,
                 help="When the problem is binary classification, the first class used in the training set. Use -1 for low_high problems. The second class is ignored.")
    parser.add_argument('-s', '--second_class', type=int, default=7, help="When the problem is binary classification, the second class used in the training set.")

    # pretraining details
    parser.add_argument('-p', '--prior_size', type=float, default=0.0, help="Portion of the training set that is used to pre-train the model.")
    parser.add_argument('-v', '--validation_size', type=float, default=0.1, help="Portion of the dataset that is used to validate the model.")
    parser.add_argument('-t', '--test_size', type=float, default=0.1, help="Portion of the dataset that is used to test the model, when the test dataset is not defined.")
    parser.add_argument('-pti', '--pretraining_epochs', type=int, default=50, help="Number of epochs used to pretrain the model.")
    parser.add_argument('-plr', '--pretraining_lr', type=float, default=1e-3, help="Learning rate used by the optimizer to pretrain the model.")

    # training details
    parser.add_argument('-m', '--model_type', type=str, default="mlp", help="Type of model to train.")
    parser.add_argument('-me', '--max_epochs', type=int, default=1, help="Maximum number of epochs to train the model at each step of P2L.")
    parser.add_argument('-b', '--batch_size', type=int, default=128, help="Batch size used to train the model.")
    parser.add_argument('-dp', '--dropout_probability', type=float, default=0.2, help="Dropout probability for the layers of the model.")
    parser.add_argument('--early_stopping', action='store_false', help="Should the model early stop.")
    parser.add_argument('-esp', '--early_stopping_patience', type=int, default=20, help="Number of iterations before early stopping.")

    # optimizer
    parser.add_argument('-o', '--optimizer', type=str, default="Adam", help="Optimizer used to train the model.")
    parser.add_argument('-lr', '--training_lr', type=float, default=1e-3, help="Learning rate used to train the model.")
    parser.add_argument('-mm', '--momentum', type=float, default=0.95, help="Momentum used by the SGD optimizer. This parameter is not used by Adam.")
    parser.add_argument('--nesterov', action='store_false', help="If the SGD optimizer should use Nesterov acceleration.")

    # p2l params
    parser.add_argument('-mx', '--max_compression_size', type=int, default=1,
                     help="Maximum size of the compression set added by the P2L algorithm. -1 if everything can be added")
    parser.add_argument('-dg', '--data_groupsize', type=int, default=1, help="Number of data added to the compression set at each iterations.")
    parser.add_argument('-pt', '--patience', type=int, default=3, help="Patience of the EarlyStopping Callback used to train on the compression set.")
    parser.add_argument('--clamping', action='store_false', help="If you want to clamp the cross-entropy loss during training.")
    parser.add_argument('-pmin', '--min_probability', type=float, default=1e-5, help="Minimum probability bound for clamping.")
    parser.add_argument('-miv', '--min_val', type=float, default=0, help=" ")
    parser.add_argument('-mav', '--max_val', type=float, default=90, help=" ")
    
    # tree parameters (only used if model_type in ["tree", "forest"] )
    parser.add_argument('-mxd', '--max_depth', type=int, default=10, help="Max depth of the tree.")
    parser.add_argument('-mss', '--min_samples_split', type=int, default=2)
    parser.add_argument('-msl', '--min_samples_leaf', type=int, default=1)
    parser.add_argument('-ca', '--ccp_alpha', type=float, default=0.0)

    # forest parameters (only used if model_type == "forest")
    parser.add_argument('-nes', '--n_estimators', type=int, default=100)
    parser.add_argument('-nj', '--n_jobs', type=int, default=5)
    parser.add_argument('--warm_start', action='store_true')

    # transformer parameters
    parser.add_argument('-sh', '--n_shards', type=int, default=10)

    # bound parameters
    parser.add_argument('-del', '--delta', type=float, default=0.01, help="Delta used to compute the bounds.")
    parser.add_argument('-npb', '--nbr_parameter_bounds', type=int, default=20, help="Number of parameters used to compute the Catoni and linear bounds.")
    parser.add_argument('--classic_bounds', action='store_true', help="Use if you do not want to compute the classic bounds.")
    parser.add_argument('--p2l_bounds', action='store_true', help="Use if you do not want to compute the P2L bounds.")
    parser.add_argument('--real_bounds', action='store_false', help="Use if you do not want to compute the real valued bounds.")

    # autoencoder parameters for mmd estimation
    parser.add_argument('--estimate_mmd', action='store_false',
                        help="Use if you want to compute the mmd distance between the compression set and dataset.")
    

    # miscellaneous
    parser.add_argument('-lg', '--log_iterations', type=int, default=1, help="Log the real valued bounds after this number of iterations.")
    parser.add_argument('-sd', '--seed', type=int, default=42, help="The se1d for the experiment.")
    parser.add_argument('--deterministic', action='store_true',
                            help="Use if you want reproducible results, but not when you want to test the code, as it slows down the code.")

    args = parser.parse_args()
    
    wandb.init(project="p2l", config=args)
    p2l_algorithm()
    wandb.finish()