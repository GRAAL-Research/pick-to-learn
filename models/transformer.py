from torch import nn
from transformers import DistilBertForSequenceClassification, DistilBertConfig
from models.classification_model import ClassificationModel

class DistilBert(nn.Module):
    def __init__(self, n_classes=2, dropout_probability=0.2):
        super().__init__()
        self.n_classes = n_classes
        self.dropout_probability = dropout_probability
        distilbert_config = DistilBertConfig(seq_classif_dropout=self.dropout_probability, num_labels=self.n_classes)
        self.model = DistilBertForSequenceClassification.from_pretrained(pretrained_model_name_or_path="distilbert-base-uncased",
                                                                          config=distilbert_config)
        # for p in self.model.distilbert.parameters():
        #     p.requires_grad = False

    def forward(self, input):
        return self.model(**input).logits

class ClassificationTransformerModel(ClassificationModel):

    def training_step(self, batch, batch_idx):
        y = batch['labels']
        return super().training_step((batch, y), batch_idx)
    
    def predict_step(self, batch, batch_idx):
        y = batch['labels']
        return super().predict_step((batch, y), batch_idx)
    
    def validation_step(self, batch, batch_idx):
        y = batch['labels']
        super().validation_step((batch, y), batch_idx)

    def test_step(self, batch, batch_idx):
        y = batch['labels']
        super().test_step((batch, y), batch_idx)