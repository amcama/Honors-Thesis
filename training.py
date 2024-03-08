import os, json
import pandas as pd
import random
import torch
import numpy as np
from tqdm.notebook import tqdm
from datasets import Dataset, DatasetDict, load_metric
from transformers import AutoTokenizer
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from transformers import AutoConfig
from transformers import TrainingArguments
from sklearn.metrics import accuracy_score
from transformers import Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import copy
import tensorflow as tf

transformer_name = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(transformer_name)

configuration = 3
ignore_index = -100

relation_types = ['Negative_activation', 'Positive_activation', 'No_relation']

def main():
    init()
    data = read_data()
    x = data.values()
    data_list = []
    for e in x:
       for f in e:
           data_list.append(f)

    data = data_list

    # Take the first 60% as train, next 20% as development, last 20% as test 
    train_df, eval_df = train_test_split(data, train_size=0.6)
    eval_df, test_df = train_test_split(eval_df, train_size=0.5)

    # print(f'train rows: {len(train_df):,}')
    # print(f'eval rows: {len(eval_df):,}')
    # print(f'test rows: {len(test_df):,}')

    # create train dataset
    train_df = pd.DataFrame(train_df)
    eval_df = pd.DataFrame(eval_df)
    test_df = pd.DataFrame(test_df)

    ds = DatasetDict()
    ds['train'] = Dataset.from_pandas(train_df)
    ds['validation'] = Dataset.from_pandas(eval_df)
    ds['test'] = Dataset.from_pandas(test_df)


    train_ds = ds['train'].map(
        tokenize, batched=True,
        remove_columns=['event_indices', 'polarity', 'controller_indices', 'controlled_indices', 'trigger_indices']
    )   
    if configuration == 3:
        maxpool(train_ds)

    exit(0)
    
    # TODO add condition for configuration 

    eval_ds = ds['validation'].map(
        tokenize,
        batched=True,
        remove_columns=['event_indices', 'polarity', 'controller_indices', 'controlled_indices', 'trigger_indices']
    )
    eval_ds.to_pandas()

    num_labels = len(relation_types)
    config = AutoConfig.from_pretrained(transformer_name, num_labels = num_labels)
    model = (BertForSequenceClassification.from_pretrained(transformer_name, config=config))
    
    num_epochs = 2
    batch_size = 24 # change this for smaller data sets
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
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )    

    print("\n", train_ds, "\n")

    train_output = trainer.train()

    print("> Train Output:\n", train_output, "\n")

    test_ds = ds['test'].map(
        tokenize,
        batched=True,
        remove_columns=['event_indices', 'polarity', 'controller_indices', 'controlled_indices', 'trigger_indices']
    )
    test_ds.to_pandas()

    output = trainer.predict(test_ds)

    print("test output: ", output)

    y_true = output.label_ids
    y_pred = np.argmax(output.predictions, axis=-1)
    target_names = relation_types
    print(classification_report(y_true, y_pred, target_names=target_names))

    print("y_true: ", y_true)
    print("y_pred: ", y_pred)

    print("\nConfusion Matrix:")
    cm = metrics.multilabel_confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0,1,2])
    print(cm)

    
def tokenize(examples):
    output = tokenizer(examples['sentence_tokens'], is_split_into_words=True, truncation=True)
    return output
            


def add_entity_markers(data):
    count = 0
    for e in data['regulations']:
        og_sentence = e['sentence_tokens']
        new_sentence = copy.deepcopy(e['sentence_tokens'])

        controller_indices = e['controller_indices']
        controlled_indices = e['controlled_indices']

        controller_start = controller_indices[0]
        controller_end = controller_indices[-1]

        controlled_start = controlled_indices[0]
        controlled_end = controlled_indices[-1]

        if (controller_start < controlled_start):
            new_sentence.insert(controller_start, "[E1]")
            new_sentence.insert(controller_end + 1, "[/E1]" )
            new_sentence.insert(controlled_start+2, "[E2]")
            new_sentence.insert(controlled_end+3, "[/E2]")

        else:
            new_sentence.insert(controlled_start, "[E1]")
            new_sentence.insert(controlled_end + 1, "[/E1]" )
            new_sentence.insert(controller_start+2, "[E2]")
            new_sentence.insert(controller_end+3, "[/E2]")
        e['sentence_tokens'] = new_sentence
        # print("\n--------")
        # print("Regulations")
        # print(controller_indices)
        # print(controlled_indices)
        # print("og sentence: ", og_sentence, "\n")
        # print("new sentence: ", new_sentence, "\n")
        # print()


    for e in data['hardInstances']:
        og_sentence = e['sentence_tokens']
        new_sentence = copy.deepcopy(e['sentence_tokens'])
        indices = e['entities_indices']
        num_entities = len(indices)
        offset = 0
        for i in range(0, num_entities):
            if (offset == 0):
                start = indices[i][0] 
                end = indices[i][-1]
                offset += 1 
            else: 
                start = indices[i][0] + offset + 1
                end = indices[i][0] + offset + 2
                offset += 2

            new_sentence.insert(start, "[E{}]".format(i+1))
            new_sentence.insert(end+1, "[/E{}]".format(i+1))
        e['sentence_tokens'] = new_sentence

        # print("\n--------")
        # print("Hard Instances")
        # print(indices, "\n")
        # print("og sentence: ", og_sentence, "\n")
        # print("new sentence: ", new_sentence, "\n")
        # print()

    for e in data['withoutRegulations']:
        og_sentence = e['sentence_tokens']
        new_sentence = copy.deepcopy(e['sentence_tokens'])
        indices = e['entities_indices']
        num_entities = len(indices)
        offset = 0
        for i in range(0, num_entities):
            if (offset == 0):
                start = indices[i][0] 
                end = indices[i][-1]
                offset += 1 
            else: 
                start = indices[i][0] + offset + 1
                end = indices[i][0] + offset + 2
                offset += 2

            new_sentence.insert(start, "[E{}]".format(i+1))
            new_sentence.insert(end+1, "[/E{}]".format(i+1))
        e['sentence_tokens'] = new_sentence

        # print("\n--------")
        # print("Without Regulations")
        # print(indices, "\n")
        # print("og sentence: ", og_sentence, "\n")
        # print("new sentence: ", new_sentence, "\n")
        # print()


    return data



def read_test_data():
    json_data = []
    f = open("test_data.json")
    data = json.load(f)
    # print(data)

    for e in data:
        if (e.get('type')):
            e['label'] = e.pop('type')
            if (e['label'] == "Negative_activation"):
                e['label'] = 0
            elif (e['label'] == "Positive_activation"):
                e['label'] = 1
            elif (e['label'] == "Negative_regulation"):
                e['label'] = 0
            elif (e['label'] == "Positive_regulation"):
                e['label'] = 1
        else:
            e['label'] = 2
    if (configuration != 1):
        add_entity_markers(data)
    return data
 

def read_data():
    """ 
    Read data from sample_training_data folder and remove duplicates.
    """
    data = {
        "regulations" : [],
        "hardInstances": [],
        "withoutRegulations": [],
        "emptySentences": []
        }
    
    # directory1 = 'sample_training_data'
    # directory2 = 'negative_training_data'

    directory1 = 'test_sample'
    directory2 = 'test_negative'

    negative_count = 0
    positive_count = 0
    none_count = 0

    count = 0

    for filename in os.listdir(directory1):
        if filename.endswith('.json'):
            with open(os.path.join(directory1, filename)) as f:
                file = json.load(f)
                if (len(file) > 0):
                    for e in file:
                        if (e['type'] == "Negative_activation" or e['type'] == "Negative_regulation"):
                                e['label'] = 0
                                negative_count += 1
                        elif (e['type'] == "Positive_activation" or e['type'] == "Positive_regulation"):
                                e['label'] = 1
                                positive_count += 1
                
                        data['regulations'].append(e)

    for filename in os.listdir(directory2):
        if filename.endswith('.json'):
            with open(os.path.join(directory2, filename)) as f:
                file = json.load(f)
                if (len(file) > 0): 
                    regulations = file['regulations']
                    hard_instances = file['hardInstances']
                    without_regulations = file['withoutRegulations']
                    empty_sentences = file['emptySentences']

                    for e in regulations:
                        if (e['type']):
                            if (e['type'] == "Negative_activation" or e['type'] == "Negative_regulation"):
                                e['label'] = 0
                                negative_count += 1
                            
                            elif (e['type'] == "Positive_activation" or e['type'] == "Positive_regulation"):
                                e['label'] = 1
                                positive_count += 1
                            
                            else:
                                raise Exception("Unknown label in regulations data!")
                            data['regulations'].append(e)
                            count += 1
                        else:
                            raise Exception("Unknown label in regulations data!")

                    for e in hard_instances:
                        e['label'] = 0
                        data['hardInstances'].append(e)
                        count += 1
                        negative_count += 1 

                    for e in without_regulations:
                        e['label'] = 2
                        none_count += 1
                        data['withoutRegulations'].append(e)
                        count += 1

                    for e in empty_sentences:
                        e['label'] = 2
                        none_count += 1
                        data['emptySentences'].append(e)
                        count += 1 
                    

    # print("Positive: ", positive_count)
    # print("Negative: ", negative_count)
    # print("None: ", none_count)
    
    # TODO remove duplicates & shuffle 
    
    if (configuration != 1):
        data = add_entity_markers(data)
    else:
        data = data
    return data


def pretty_print(list):
    for e in list:
        print(e, "\n")

# https://github.com/huggingface/transformers/blob/65659a29cf5a079842e61a63d57fa24474288998/src/transformers/models/bert/modeling_bert.py#L1486
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False) # what should add_pooling_layer be?
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
        entity_indexes = dict()
        for i in range(0,len(input_ids)):
            tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
            if "[E1]" in tokens:
                entity_indexes["[E1]"] = i
            if "[/E1]" in tokens:
                entity_indexes["[/E1]"] = i
            if "[E2]" in tokens:
                entity_indexes["[E2]"] = i
            if "[/E2]" in tokens:
                entity_indexes["[/E2]"] = i

        embeddings = outputs.last_hidden_state

        e1start = embeddings[entity_indexes.get("[E1]")]
        e1end = embeddings[entity_indexes.get("[/E1]")]
        e2start = embeddings[entity_indexes.get("[E2]")]
        e2end = embeddings[entity_indexes.get("[/E2]")]
     
        labels = torch.FloatTensor(labels)

        concatenated = torch.concat((e1start, e1end, e2start, e2end))
        concatenated = self.dropout(concatenated) 
        logits = self.classifier(concatenated) # linear layer? 
        # print(labels)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            inputs = logits
            targets = concatenated
            print(inputs.shape)
            print(targets.shape)
            loss = loss_fn(logits, targets)
            

    

def maxpool(data):
    config = AutoConfig.from_pretrained(transformer_name, num_labels = 3, output_hidden_states=True)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[E1]','[/E1]', '[E2]', '[/E2]']})
    model = BertForSequenceClassification.from_pretrained(transformer_name, config=config)
    model.resize_token_embeddings(len(tokenizer))

    count = 0
    for item in data:
        label = item['label']
        count += 1
        sentence = item['sentence_tokens']
        tokens = tokenizer(sentence, padding=True, return_tensors='pt')

        output = model.forward(input_ids=tokens['input_ids'], 
                               attention_mask=tokens['attention_mask'], 
                               token_type_ids=tokens['token_type_ids'],
                               labels=label
                               )
        # hidden_states = output.hidden_states

        if (count > 1):
            break
        exit(0)


def compute_metrics(eval_pred):
    y_true = eval_pred.label_ids
    y_pred = np.argmax(eval_pred.predictions, axis=-1)
    return {'accuracy': accuracy_score(y_true, y_pred)}

def init():
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
main()