import os, json
import pandas as pd
import random
import math
import torch
import numpy as np
from tqdm.notebook import tqdm
from datasets import Dataset, DatasetDict, load_metric
# from clulab.clu_tokenizer import CluTokenizer
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
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

transformer_name = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(transformer_name)

configuration = 2

def main():
    init()
    data = read_data()
    
    # Take the first 60% as train, next 20% as development, last 20% as test (Don't use test for now)
    train_list, eval_list = train_test_split(data, train_size=0.6)

    # create train dataset
    train_df = pd.DataFrame(train_list)
    eval_df = pd.DataFrame(eval_list)
    # test_df = pd.DataFrame(test_list)

    # print(f'train rows: {len(train_df.index):,}')
    # print(f'eval rows: {len(dev_df.index):,}')
    # print(f'test rows: {len(test_df.index):,}')

    ds = DatasetDict()
    ds['train'] = Dataset.from_pandas(train_df)
    ds['validation'] = Dataset.from_pandas(eval_df)
    # ds['test'] = Dataset.from_pandas(test_df)


    train_ds = ds['train'].map(
        tokenize, batched=True,
        remove_columns=['event_indices', 'polarity', 'controller_indices', 'controlled_indices', 'trigger_indices']
    )   
    #maybe add entity markers after tokenization???

    eval_ds = ds['validation'].map(
        tokenize,
        batched=True,
        remove_columns=['event_indices', 'polarity', 'controller_indices', 'controlled_indices', 'trigger_indices']
    )
    eval_ds.to_pandas()

    num_labels = 3 # TODO make this not hard coded
    config = AutoConfig.from_pretrained(transformer_name, num_labels = num_labels)
    model = (BertForSequenceClassification.from_pretrained(transformer_name, config=config))
    
    # -- [10] --
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
        # label_names=['input_ids', 'token_type_ids', 'attention_mask', 'label']
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
    print("\n", train_ds, "\n")
    train_output = trainer.train()
    print("~ Train Output:\n", train_output, "\n")

    # -- [14] --
    predictions = trainer.predict(eval_ds)

    print("~ Prediction output for eval_ds:\n", predictions, "\n")
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=-1)
    target_names = ['Positive_activation', 'Negative_activation', 'No_relation']

    print(classification_report(y_true, y_pred, target_names=target_names))

    # get confusion matrix
    print("Confusion Matrix:")
    cm = metrics.multilabel_confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0,1,2,3,4])

    print(cm)
"""
TN -> [0,0]
FN -> [1,0]
TP -> [1,1]
FP -> [0,1]

In multilabel confusion matrix, the count of true negatives is at 00, 
false negatives is at 10, true positives is at 11 and false positives is at 01.
"""

    
def tokenize(examples):
    output = tokenizer(examples['sentence_tokens'], is_split_into_words=True, truncation=True)
    return output



def add_entity_markers(text):
    for i in range(0, len(text)): 
        og_sentence = text[i]['sentence_tokens']
        new_sentence = []
        if (('controller_indices' in text[i]) & ('controlled_indices' in text[i])):
            controller_indices = text[i]['controller_indices']
            controlled_indices = text[i]['controlled_indices']

            controller_start = controller_indices[0]
            controller_end = controller_indices[len(controller_indices)-1] - 1

            controlled_start = controlled_indices[0]
            controlled_end = controlled_indices[len(controlled_indices)-1] - 1

            entity_count = 1
            for j in range(0, len(og_sentence)):
                if (j == controller_start):
                    new_sentence.append("[E{}]".format(entity_count))
                    new_sentence.append(og_sentence[j])
                    if (j == controller_end):
                        new_sentence.append("[/E{}]".format(entity_count))
                        entity_count += 1
                        continue
                    continue

                if (j == controller_end):
                    new_sentence.append(og_sentence[j])
                    new_sentence.append("[/E{}]".format(entity_count))
                    entity_count += 1
                    continue
                
                if (j == controlled_start):
                    new_sentence.append("[E{}]".format(entity_count))
                    new_sentence.append(og_sentence[j])
                    if (j == controlled_end):
                        new_sentence.append("[/E{}]".format(entity_count))
                        entity_count += 1
                        continue
                    continue

                if (j == controlled_end):
                    new_sentence.append(og_sentence[j])
                    new_sentence.append("[/E{}]".format(entity_count))
                    entity_count += 1
                    continue
            
                if (og_sentence[j].strip() == "."):
                    new_sentence.append("[SEP]")
                else:
                    new_sentence.append(og_sentence[j])
                
            # print("\n---------------------")
            # print("CONTROLLER INDICES: ", controller_indices)
            # print("CONTROLLED INDICES: ", controlled_indices)
            # label_1 = ""
            # if (text[i]['label'] == 0):
            #     label_1 = "Negative_activation"
            # elif (text[i]['label'] == 1):
            #     label_1 = "Positive_activation"
            # print("LABEL: ", label_1)

            # print("\nORIGINAL: ", og_sentence)
            # print("\nNEW:      ", new_sentence)



        if (('entity_one_indices' in text[i]) & ('entity_two_indices' in text[i])):
            e1_indices = text[i]['entity_one_indices']
            e2_indices = text[i]['entity_two_indices']

            e1_start = e1_indices[0]
            e1_end = e1_indices[len(e1_indices)-1] - 1

            e2_start = e2_indices[0]
            e2_end = e2_indices[len(e2_indices)-1] - 1

            entity_count = 1
            for j in range(0, len(og_sentence)):
                if (j == e1_start):
                    new_sentence.append("[E{}]".format(entity_count))
                    new_sentence.append(og_sentence[j])
                    if (j == e1_end):
                        new_sentence.append("[/E{}]".format(entity_count))
                        entity_count += 1
                        continue
                    continue

                if (j == e1_end):
                    new_sentence.append(og_sentence[j])
                    new_sentence.append("[/E{}]".format(entity_count))
                    entity_count += 1
                    continue
                
                if (j == e2_start):
                    new_sentence.append("[E{}]".format(entity_count))
                    new_sentence.append(og_sentence[j])
                    if (j == e2_end):
                        new_sentence.append("[/E{}]".format(entity_count))
                        entity_count += 1
                        continue
                    continue

                if (j == e2_end):
                    new_sentence.append(og_sentence[j])
                    new_sentence.append("[/E{}]".format(entity_count))
                    entity_count += 1
                    continue
            
                if (og_sentence[j].strip() == "."):
                    new_sentence.append("[SEP]")
                else:
                    new_sentence.append(og_sentence[j])
            # print("\n---------------------\n")
            # print("E1 INDICES: ", e1_indices)
            # print("E2 INDICES: ", e2_indices)
            # if (text[i]['label'] == 2):
            #     print("LABEL: No_relation")
            # else:
            #     print("LABEL:", text[i]['label'])
            # print("\nORIGINAL: ", og_sentence)
            # print("\nNEW:      ", new_sentence)

    return text






def read_data():
    """ 
    Read data from sample_training_data folder and remove duplicates.
    """
    json_data = []
    directory1 = 'sample_training_data'
    directory2 = 'negative_training_data'

    count = 0 # for testing with smaller sample sizes
    for filename in os.listdir(directory1):
        if filename.endswith('.json'):
            with open(os.path.join(directory1, filename)) as f:
                data = json.load(f)
                if (len(data) > 0):
                    json_data.append(data)
                    if (len(json_data) > 75): # for testing
                        break

    for filename in os.listdir(directory2):
        if filename.endswith('.json'):
            with open(os.path.join(directory2, filename)) as f:
                data = json.load(f)
                if (len(data) > 0):
                    json_data.append(data)
                    count += 1
                    if (len(json_data) > 100): # for testing
                        break
               
    list_no_dups = remove_duplicates(json_data)
    random.shuffle(list_no_dups)

    # 0: Negative_activation 
    # 1: Postive_activation
    # 2: No_relation
    for e in list_no_dups:
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
        data = add_entity_markers(list_no_dups)
    else:
        data = list_no_dups
    return data

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