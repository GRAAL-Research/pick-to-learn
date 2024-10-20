from models.linear_network import MnistMlp
from models.convolutional_network import MnistCnn, Cifar10Cnn9l
from models.transformer import DistilBert, ClassificationTransformerModel
from models.classification_model import ClassificationModel
from models.classification_tree import ClassificationTree, ClassificationForest, ClassificationTreeModel
from models.regression_tree import RegressionTree, RegressionForest, RegressionTreeModel

def create_model(config):
    if config.get('prior_size', 0.0) == 0.0:
        lr = config.get('training_lr', None)
    else:
        lr = config.get('pretraining_lr', None)

    if config['dataset'] == "mnist":
        if config['model_type'] == "mlp":
            return ClassificationModel(MnistMlp(dataset_shape=784,
                                                n_classes=config['n_classes'],
                                                dropout_probability=config['dropout_probability']),
                                                optimizer=config['optimizer'],
                                                lr=lr,
                                                momentum=config['momentum'],
                                                batch_size=config['batch_size']
                                                )
        elif config['model_type'] == "cnn":
            return ClassificationModel(MnistCnn(n_classes=config['n_classes'],
                                                dropout_probability=config['dropout_probability']),
                                                optimizer=config['optimizer'],
                                                lr=lr,
                                                momentum=config['momentum'],
                                                batch_size=config['batch_size']
                                                )
    elif config['dataset'] == "randomMnist":
        return ClassificationModel(MnistCnn(n_classes=config['n_classes'],
                                    dropout_probability=config['dropout_probability']),
                                    optimizer=config['optimizer'],
                                    lr=lr,
                                    momentum=config['momentum'],
                                    batch_size=config['batch_size']
                                    )
    elif config['dataset'] == "cifar10":
        return ClassificationModel(Cifar10Cnn9l(n_classes=config['n_classes'],
                                                dropout_probability=config['dropout_probability']),
                                                optimizer=config['optimizer'],
                                                lr=lr,
                                                momentum=config['momentum'],
                                                batch_size=config['batch_size']
                                                )
    elif config['dataset'] == "amazon":
        if config['model_type'] == "transformer":
            return ClassificationTransformerModel(DistilBert(n_classes=config['n_classes'],
                                                dropout_probability=config['dropout_probability']),
                                    optimizer=config['optimizer'],
                                    lr=lr,
                                    momentum=config['momentum'],
                                    batch_size=config['batch_size'])
    elif config['dataset'] in ["concrete", "airfoil", "parkinson", "infrared", "powerplant"]:
        if config['model_type'] == "tree":
            return RegressionTreeModel(RegressionTree(
                max_depth=config['max_depth'],
                min_samples_split=config['min_samples_split'],
                min_samples_leaf=config['min_samples_leaf'],
                seed=config['seed'],
                ccp_alpha=config['ccp_alpha']
            ))
        elif config['model_type'] == "forest":
            return RegressionTreeModel(RegressionForest(
                n_estimators=config['n_estimators'],
                max_depth=config['max_depth'],
                min_samples_split=config['min_samples_split'],
                min_samples_leaf=config['min_samples_leaf'],
                seed=config['seed'],
                ccp_alpha=config['ccp_alpha']
            ))
    elif config['dataset'] == "moons":
        if config['model_type'] == "tree":
            return ClassificationTreeModel(ClassificationTree(
                n_classes=config['n_classes'],
                max_depth=config['max_depth'],
                min_samples_split=config['min_samples_split'],
                min_samples_leaf=config['min_samples_leaf'],
                seed=config['seed'],
                ccp_alpha=config['ccp_alpha']
            ))
        elif config['model_type'] == "forest":
            return ClassificationTreeModel(ClassificationForest(
                n_classes=config['n_classes'],
                n_estimators=config['n_estimators'],
                max_depth=config['max_depth'],
                min_samples_split=config['min_samples_split'],
                min_samples_leaf=config['min_samples_leaf'],
                seed=config['seed'],
                ccp_alpha=config['ccp_alpha']
            ))
    
    raise NotImplementedError(f"Model type = {config['model_type']} with dataset {config['dataset']} is not implemented yet.")

def load_pretrained_model(checkpoint_path, config):
    if config.get('regression', False):
        if config['model_type'] in ['tree', 'forest']:
            return RegressionTreeModel.load_from_checkpoint(checkpoint_path)
    else:
        if config['model_type'] in ['mlp', 'cnn']:
            return ClassificationModel.load_from_checkpoint(checkpoint_path)
        elif config['model_type'] == "transformer":
            return ClassificationTransformerModel.load_from_checkpoint(checkpoint_path)
    setting = "regression" if config['regression'] else "classification"
    raise NotImplementedError(f"Loading checkpoints for a {config['model_type']} in a {setting} setting is not supported yet.")