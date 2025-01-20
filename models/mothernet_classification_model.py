import lightning as L
import torch
from torchmetrics.classification import MulticlassAccuracy
import math
from ticl.prediction import MotherNetClassifier
from ticl.prediction.mothernet import extract_mlp_model
from ticl.utils import get_mn_model, normalize_by_used_features_f
from ticl.model_builder import load_model
from sklearn.preprocessing import LabelEncoder
import numpy as np
from einops import rearrange, repeat

def extract_mlp_model_with_fixed_classes(model, config, X_train, y_train, n_classes, device="cpu", inference_device="cpu", scale=True):
    #### This code is licensed under Apache 2.0 License and comes from https://github.com/microsoft/ticl/tree/main
    if "cuda" in inference_device and device == "cpu":
        raise ValueError("Cannot run inference on cuda when model is on cpu")
    try:
        max_features = config['prior']['num_features']
    except KeyError:
        max_features = 100
    eval_position = X_train.shape[0]
    #### Original
    # n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]
    if torch.is_tensor(X_train):
        xs = X_train.to(device)
    else:
        xs = torch.Tensor(X_train.astype(float)).to(device)
    if torch.is_tensor(y_train):
        ys = y_train.to(device)
    else:
        ys = torch.Tensor(y_train.astype(float)).to(device)

    if scale:
        eval_xs_ = normalize_data(xs, eval_position)
    else:
        eval_xs_ = torch.clip(xs, min=-100, max=100)

    eval_xs = normalize_by_used_features_f(
        eval_xs_, X_train.shape[-1], max_features)
    if X_train.shape[1] > max_features:
        raise ValueError(f"Cannot run inference on data with more than {max_features} features")
    x_all_torch = torch.concat([eval_xs, torch.zeros((X_train.shape[0], max_features - X_train.shape[1]), device=device)], axis=1)
    x_src = model.encoder(x_all_torch.unsqueeze(1))

    if model.y_encoder is not None:
        y_src = model.y_encoder(ys.unsqueeze(1).unsqueeze(-1))
        train_x = x_src + y_src
    else:
        train_x = x_src

    if hasattr(model, "transformer_encoder"):
        # tabpfn mlp model maker
        output = model.transformer_encoder(train_x)
    elif hasattr(model, "ssm"):
        # ssm model maker
        output = model.ssm(train_x)
    else:
        # perceiver
        data = rearrange(train_x, 'n b d -> b n d')
        x = repeat(model.latents, 'n d -> b n d', b=data.shape[0])

        # layers
        for cross_attn, cross_ff, self_attns in model.layers:
            x = cross_attn(x, context=data) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        output = rearrange(x, 'b n d -> n b d')
    (b1, w1), *layers = model.decoder(output, ys)

    w1_data_space_prenorm = w1.squeeze()[:n_features, :]
    b1_data_space = b1.squeeze()

    w1_data_space = w1_data_space_prenorm / (n_features / max_features)

    if model.decoder.weight_embedding_rank is not None and len(layers):
        w1_data_space = torch.matmul(w1_data_space, model.decoder.shared_weights[0])

    layers_result = [(b1_data_space, w1_data_space)]

    for i, (b, w) in enumerate(layers[:-1]):
        if model.decoder.weight_embedding_rank is not None:
            w = torch.matmul(w, model.decoder.shared_weights[i + 1])
        layers_result.append((b.squeeze(), w.squeeze()))

    # remove extra classes on output layer
    if len(layers):
        layers_result.append((layers[-1][0].squeeze()[:n_classes], layers[-1][1].squeeze()[:, :n_classes]))
    else:
        layers_result = [(b1_data_space[:n_classes], w1_data_space[:, :n_classes])]

    if inference_device == "cpu":
        def detach(x):
            return x.detach().cpu().numpy()
    else:
        def detach(x):
            return x.detach()

    return [(detach(b), detach(w)) for (b, w) in layers_result]

class MotherNetClassifierWithFixedClasses(MotherNetClassifier):
    #### This code is licensed under Apache 2.0 License and comes from https://github.com/microsoft/ticl/tree/main
    def fit(self, X, y, n_classes):
        self.X_train_ = X
        le = LabelEncoder()
        #### Original
        # y = le.fit_transform(y)
        #### Modification
        le.fit(np.arange(n_classes))
        y = le.transform(y)
        ####
        if len(le.classes_) > 10:
            raise ValueError(f"Only 10 classes supported, found {len(le.classes_)}")
        if self.model is not None:
            model = self.model
            config = self.config
        else:
            model, config = load_model(self.path, device=self.device)
            self.config = config
        if "model_type" not in config:
            config['model_type'] = config.get("model_maker", 'tabpfn')
        if config['model_type'] not in ["mlp", "mothernet", 'ssm_mothernet']:
            raise ValueError(f"Incompatible model_type: {config['model_type']}")
        model.to(self.device)
        n_classes = len(le.classes_)
        indices = np.mod(np.arange(n_classes) + self.label_offset, n_classes)
        layers = extract_mlp_model_with_fixed_classes(model, config, X, np.mod(y + self.label_offset, n_classes), n_classes, device=self.device,
                                   inference_device=self.inference_device, scale=self.scale)
        if self.label_offset == 0:
            self.parameters_ = layers
        else:
            *lower_layers, b_last, w_last = layers
            self.parameters_ = (*lower_layers, (b_last[indices], w_last[:, indices]))
        self.classes_ = le.classes_
        self.mean_ = np.nan_to_num(np.nanmean(X, axis=0), 0)
        self.std_ = np.nanstd(X, axis=0, ddof=1) + .000001
        self.std_[np.isnan(self.std_)] = 1

        return self


class ClampedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, clamping=False, pmin=1e-5, reduction='mean'):
        super().__init__()
        self.clamping=clamping
        self.pmin = pmin
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.loss = torch.nn.NLLLoss(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out = self.log_softmax(input)
        if self.clamping:
            out = torch.clamp(out, min=math.log(self.pmin))
        return self.loss(out, target)
    

class MotherNetClassificationModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.model = model
        self.configure_loss(clamping=False)
        self.metric = MulticlassAccuracy(num_classes=self.model.n_classes).to(self.device)
        self.no_reduction_loss = torch.nn.CrossEntropyLoss(reduction='none')

        model_string = "mn_d2048_H4096_L2_W32_P512_1_gpu_warm_08_25_2023_21_46_25_epoch_3940_no_optimizer.pickle"
        model_path = get_mn_model(model_string)
        self.mothernet = MotherNetClassifierWithFixedClasses(path=model_path, device=self.device.type, inference_device=self.device.type, scale=False)

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.mothernet.fit(x.cpu(),y.cpu(), self.model.n_classes)
        self.model.update_weights(self.mothernet.parameters_)
        self.model.to(self.device)
        
        y_hat = self.model(x)
        train_acc = self.metric(torch.argmax(y_hat, dim=1), y)
        self.log("train_acc", train_acc, prog_bar=True)

        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.no_reduction_loss(y_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        validation_loss = self.loss(y_hat, y)
        self.log("validation_loss", validation_loss)
        validation_acc = self.metric(torch.argmax(y_hat, dim=1), y)
        self.log("validation_acc", validation_acc)
        validation_error = 1 - validation_acc
        self.log("validation_error", validation_error)
        return torch.nn.CrossEntropyLoss(reduction='sum')(y_hat, y)


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        test_loss = self.loss(y_hat, y)
        self.log("test_loss", test_loss)
        test_acc = self.metric(torch.argmax(y_hat, dim=1), y)
        self.log("test_acc", test_acc)
        test_error = 1 - test_acc
        self.log("test_error", test_error)
    
    def configure_optimizers(self):
        return None
    
    def configure_loss(self, clamping : bool = False, pmin : float = 1e-5):
        self.loss = ClampedCrossEntropyLoss(clamping=clamping, pmin=pmin, reduction='mean')
    
