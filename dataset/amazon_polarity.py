from datasets import load_dataset, load_from_disk
from transformers import DistilBertTokenizerFast, DataCollatorWithPadding
import os
from utils import CustomDataset

def load_amazon_polarity():
    file_path = "./amazon_polarity"
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
    
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    file_name = "/amazon_polarity_dataset"
    file_path = file_path + file_name
    if not os.path.isdir(file_path):
        ds = load_dataset("mteb/amazon_polarity")
        tokenized_datasets = ds.map(lambda dataset: tokenizer(dataset['text'], truncation=True), batched=True)
        tokenized_datasets.set_format(type="torch", columns=["input_ids", "label"])
        tokenized_datasets.save_to_disk(file_path)
    else:
        tokenized_datasets = load_from_disk(file_path)
    
    collate_fn = DataCollatorWithPadding(tokenizer)
    train_set = CustomDataset(data=tokenized_datasets['train']['input_ids'],
                            targets=tokenized_datasets['train']['label'],
                            transform=collate_fn,
                            real_targets=False,
                            is_an_image=False)
    test_set = CustomDataset(data=tokenized_datasets['test']['input_ids'],
                        targets=tokenized_datasets['test']['label'],
                        transform=collate_fn,
                        real_targets=False,
                        is_an_image=False)
    
    return train_set, test_set


