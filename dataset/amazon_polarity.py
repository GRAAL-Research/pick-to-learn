from datasets import load_dataset, load_from_disk
from transformers import DistilBertTokenizerFast, DataCollatorWithPadding
import os
from utils import CustomDataset
from functools import partial

def collate_function(input, data_collator=None):
    return data_collator([{'input_ids':i[0], 'labels':i[1]}for i in input])
    
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
    
    data_collator = DataCollatorWithPadding(tokenizer)
    collate_fn = partial(collate_function, data_collator=data_collator)
    
    train_set = CustomDataset(data=tokenized_datasets['train']['input_ids'],
                            targets=tokenized_datasets['train']['label'],
                            transform=None,
                            real_targets=False,
                            is_an_image=False)
    test_set = CustomDataset(data=tokenized_datasets['test']['input_ids'],
                        targets=tokenized_datasets['test']['label'],
                        transform=None,
                        real_targets=False,
                        is_an_image=False)
    
    return train_set, test_set, collate_fn


