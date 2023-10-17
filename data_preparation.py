import os, json
import pandas as pd
import random
import math
import torch
import numpy as np
from tqdm.notebook import tqdm
from datasets import Dataset, DatasetDict
from clulab.clu_tokenizer import CluTokenizer
from transformers import AutoTokenizer
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from transformers import AutoConfig
from transformers import TrainingArguments
from sklearn.metrics import accuracy_score
from transformers import Trainer


transformer_name = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(transformer_name)

def main():
    tqdm.pandas()
    use_gpu = True
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    # print(f'device: {device.type}')
    seed = 1234
    if seed is not None:
        # print(f'random seed: {seed}')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    data = read_data()
    # pretty_print(data)

    # Take the first 60% as train, next 20% as development, last 20% as test (Don't use test for now)
    size = len(data)
    train_size = round(size * 0.6)
    dev_size = round(size * 0.2)
    test_size = round(size * 0.2)

    # print("size: {}  train_size: {}  dev_size: {}  test_size: {}\n".format(size, train_size, dev_size, test_size))
    assert(train_size + dev_size + test_size == size)

    train_list = []
    dev_list = []
    test_list = []

    for i in range(0, train_size):
        train_list.append(data[i])
        
    for i in range(train_size, (train_size + dev_size)):
        dev_list.append(data[i])

    for i in range(dev_size, (dev_size + test_size)):
        test_list.append(data[i])

    # create train dataset
    train_df = pd.DataFrame(train_list)
    print(train_df)

    dev_df = pd.DataFrame(dev_list)
    test_df = pd.DataFrame(test_list)

    print(f'train rows: {len(train_df.index):,}')
    print(f'eval rows: {len(dev_df.index):,}')
    print(f'test rows: {len(test_df.index):,}')

    ds = DatasetDict()
    ds['train'] = Dataset.from_pandas(train_df)
    ds['validation'] = Dataset.from_pandas(dev_df)
    ds['test'] = Dataset.from_pandas(test_df)

    print(ds)

    train_ds = ds['train'].map(
        tokenize, batched=True,
        # TODO should I keep columns?
        remove_columns=['sentence_tokens', 'event_indices', 'type', 'polarity', 'controller_indices', 'controlled_indices', 'trigger_indices']
    )   
    train_ds.to_pandas()

    eval_ds = ds['validation'].map(
        tokenize,
        batched=True,
        remove_columns=['sentence_tokens', 'event_indices', 'type', 'polarity', 'controller_indices', 'controlled_indices', 'trigger_indices']
    )
    eval_ds.to_pandas() # maybe not yet...

    # -- [9] --
    labels = train_ds.num_columns
    print("labels = ", labels)

    config = AutoConfig.from_pretrained(transformer_name, num_labels = labels)
    model = (BertForSequenceClassification.from_pretrained(transformer_name, config=config))
    
    # -- [10] --
    print("\n")
    num_epochs = 2
    batch_size = 24
    weight_decay = 0.01
    model_name = f'{transformer_name}-sequence-classification'
    training_args = TrainingArguments(
        output_dir=model_name,
        log_level='error',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy='epoch',
        weight_decay=weight_decay,
        label_names=['input_ids', 'token_type_ids', 'attention_mask']
    )
    
    # -- [12] -- 
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )    
    trainer.train()


def tokenize(examples):
    output = tokenizer(examples['sentence_tokens'], is_split_into_words=True)
    return output

def read_data():
    """ 
    Read data from sample_training_data folder and remove duplicates.
    """
    json_data = []
    directory = 'sample_training_data'
    count = 0
    
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename)) as f:
                data = json.load(f)
                if (len(data) > 0):
                    json_data.append(data)
                    count += 1

        if (count == 3): # just for testing smaller data
            break

    list_no_dups = remove_duplicates(json_data)
    random.shuffle(list_no_dups)
    return list_no_dups

def remove_duplicates(list):
    # concatenate all nested items into single list
    single_list = []
    for e1 in list:
        for e2 in e1:
            single_list.append(e2)

    list_no_duplicates = []
    seen = set()

    for e in single_list:
        joined_string = ''.join(e['sentence_tokens'])
        if (joined_string not in seen):
            list_no_duplicates.append(e)

        seen.add(joined_string)
    return list_no_duplicates

def pretty_print(list):
    for e in list:
        print(e, "\n")

# https://github.com/huggingface/transformers/blob/65659a29cf5a079842e61a63d57fa24474288998/src/transformers/models/bert/modeling_bert.py#L1486
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        cls_outputs = outputs.last_hidden_state[:, 0, :]
        cls_outputs = self.dropout(cls_outputs)
        logits = self.classifier(cls_outputs)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
def compute_metrics(eval_pred):
    y_true = eval_pred.label_ids
    y_pred = np.argmax(eval_pred.predictions, axis=-1)
    return {'accuracy': accuracy_score(y_true, y_pred)}

main()
