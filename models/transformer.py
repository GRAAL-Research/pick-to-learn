from torch import nn
from transformers import DistilBertForSequenceClassification, DistilBertConfig

class DistilBert(nn.Module):
    def __init__(self, n_classes=2, dropout_probability=0.2):
        super().__init__()
        self.n_classes = n_classes
        self.dropout_probability = dropout_probability
        distilbert_config = DistilBertConfig(seq_classif_dropout=self.dropout_probability, num_labels=self.n_classes)
        self.model = DistilBertForSequenceClassification.from_pretrained(pretrained_model_name_or_path="distilbert-base-uncased",
                                                                          config=distilbert_config)
        for p in self.model.distilbert.parameters():
            p.requires_grad = False

    def forward(self, input):
        return self.model(**input).logits

