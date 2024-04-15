import os, json
import pandas as pd
import random
import torch
import numpy as np
import torch.utils
from tqdm.notebook import tqdm
from datasets import Dataset, DatasetDict
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
from sklearn.model_selection import train_test_split
import copy
import os
# from transformers import DataCollatorForTokenClassification

'''
Configurations: 
Configuration 1: Classifies the sentences using the [CLS] embedding
Configuration 2: Classifies the sentences using the [CLS] embedding and adds 4 entitity markers:
    begin controller [E1], end controller [/E1], begin controlled [E2], end controlled [/E2]
Configuration 3: Classifies the sentences using the [CLS] embedding and adds 4 entitity markers. 
    This configuration max pools the data in the forward function.
'''
transformer_name = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(transformer_name)

# TODO make configuration a command line option
configuration = 3
classes = ['Negative_activation', 'Positive_activation', 'No_relation']

# TOKENIZERS_PARALLELISM=False
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

def remove_duplicates(data):
    seen = set()
    unique_data = []
    for item in data:
        item_str = json.dumps(item, sort_keys=True)
        if item_str not in seen:
            seen.add(item_str)
            unique_data.append(item)

    return unique_data

def main():
    init()
    tokenizer.add_special_tokens({'additional_special_tokens': ['[E1]','[/E1]', '[E2]', '[/E2]']})
    
    data = read_data("training_data")
    # data = read_data("test_data")

    values = data.values()
    data_list = []
    for e in values:
        for f in e:
            data_list.append(f)
            # if (len(data_list) > 400): # just for testing
            #     break
    data = data_list
    data = remove_duplicates(data)
    random.shuffle(data) 
    # data = generate_random_dataset(data, 300) # for testing with smaller datasets
    label_0 = 0
    label_1 = 0
    label_2 = 0

    for e in data:
        # print("---")
        # print(e['sentence_tokens'])
        # try:
        #     if (e['controller_indices']):
        #         print("controller: ", e['controller_indices'])
        #     if (e['controlled_indices']):
        #         print("controlled: ", e['controlled_indices'])
            
        #     if (e['entities_indices']):
        #         print("entitites: ", e['entities_indices'])
        #     print("\n")
        # except KeyError:
        #     pass
        


        if (e['label'] == 0):
            label_0 += 1
        elif (e['label'] == 1):
            label_1 += 1
            # print(e,"\n")
        elif (e['label'] == 2):
            label_2 += 1

    print("\nlabel 0: ", label_0)
    print("label 1: ", label_1)
    print("label 2: ", label_2, "\n")
    # exit(0)
    # Take the first 60% as train, next 20% as development, last 20% as test 
    train_df, eval_df = train_test_split(data, train_size=0.6)
    eval_df, test_df = train_test_split(eval_df, train_size=0.5)

    print(f'train rows: {len(train_df):,}')
    print(f'eval rows: {len(eval_df):,}')
    print(f'test rows: {len(test_df):,}')
    # print("test: ", test_df)
    train_df = pd.DataFrame(train_df)
    eval_df = pd.DataFrame(eval_df)
    test_df = pd.DataFrame(test_df)

    ds = DatasetDict()
    ds['train'] = Dataset.from_pandas(train_df)
    ds['validation'] = Dataset.from_pandas(eval_df)
    ds['test'] = Dataset.from_pandas(test_df)
    
    train_ds = ds['train'].map(
        tokenize, batched=True, 
        remove_columns=['event_indices', 'polarity', 'controller_indices', 'controlled_indices', 
                        'trigger_indices', 'type', 'sentence_tokens']
        # remove_columns=['event_indices', 'polarity', 'controller_indices', 'controlled_indices', 
        #                 'trigger_indices', 'type', 'sentence_tokens', 'rule', 'rule_name']
    )   
    print("\ntrain_ds: ", train_ds)

    eval_ds = ds['validation'].map(
        tokenize,
        batched=True,
        remove_columns=['event_indices', 'polarity', 'controller_indices', 'controlled_indices', 
                        'trigger_indices', 'type', 'sentence_tokens']
    )
    eval_ds.to_pandas()
    print("\neval_ds: ", eval_ds)

    config = AutoConfig.from_pretrained(transformer_name, num_labels = len(classes))
    model = (BertForSequenceClassification.from_pretrained(transformer_name, config=config))
    model.resize_token_embeddings(len(tokenizer))  # adjust for entity markers being added to tokenizer dictionary

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
    ) 
    
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )  
    train_output = trainer.train()
    # print("Train Output:\n", train_output, "\n")

    test_ds = ds['test'].map(
        tokenize,
        batched=True,
        remove_columns=['event_indices', 'polarity', 'controller_indices', 'controlled_indices', 
                        'trigger_indices', 'type', 'sentence_tokens']
    )
    test_ds.to_pandas()

    output = trainer.predict(test_ds)
    # print("test output: ", output)

    y_true = output.label_ids
    y_pred = np.argmax(output.predictions, axis=-1)
    print(classification_report(y_true, y_pred, target_names=classes, labels=[0,1,2]))
    print("y_true: ", y_true)
    print("y_pred: ", y_pred)

    print("\nConfusion Matrix:")
    cm = metrics.multilabel_confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0,1,2])
    print(cm)

'''
This function deletes sentences that are too long when tokenized from the dataset.
When a tokenized sentence array has a size >= 512, it breaks the forward function when training.
'''
def prune_ds(ds):
    indexes_to_delete = []
    for i in range(0, len(ds)):
        e = ds[i]
        if (len(e['input_ids']) >= 512):
            indexes_to_delete.append(i)
    indexes_to_delete.reverse()
    for index in indexes_to_delete:
        ds = np.delete(ds, index)
    return ds
    

'''
Generates a random dataset from training data to be used for testing.
new_size: the size of the new dataset to be generated
'''
def generate_random_dataset(ds, new_size):
    ds_size = len(ds)
    start = random.randint(0, ds_size-new_size)
    new_ds = []
    for i in range(start, start+new_size):
        new_ds.append(ds[i])
    return new_ds


def tokenize(examples):
    output = tokenizer(examples['sentence_tokens'], is_split_into_words=True, truncation=True)
    return output


'''
Add entity markers to the sentence tokens.
'''
def add_entity_markers(data):
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
            new_sentence.insert(controlled_start + 2, "[E2]")
            new_sentence.insert(controlled_end + 3, "[/E2]")
        else:
            new_sentence.insert(controlled_start, "[E1]")
            new_sentence.insert(controlled_end + 1, "[/E1]" )
            new_sentence.insert(controller_start + 2, "[E2]")
            new_sentence.insert(controller_end + 3, "[/E2]")
        e['sentence_tokens'] = new_sentence

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

    return data


def read_data(directory):
    """ 
    Read data from training_data and add the appropiate labels.

    regulations: Sentences that have a regulation. 
        The entities in here are the particpants of a regulation event.
    hardInstances: Sentences that have a regulation, but the entities provided are 
        not participants in the regulation event. These are all negatives. 
    withoutRegulations: Sentences that contain entities but no regulation events. 
    """
    data = {
        "regulations" : [],
        "hardInstances": [],
        "withoutRegulations": [],
        "emptySentences": []
        }


    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename)) as f:
                file = json.load(f)
                if (len(file) > 0): 
                    regulations = file['regulations']
                    hard_instances = file['hardInstances']
                    without_regulations = file['withoutRegulations']

                    # ignore empty sentences
                    # empty_sentences = file['emptySentences'] 

                    for e in regulations:
                        if (e['type']):
                            if (e['type'] == "Negative_activation" or e['type'] == "Negative_regulation"):
                                e['label'] = 0
                            elif (e['type'] == "Positive_activation" or e['type'] == "Positive_regulation"):
                                e['label'] = 1
                            else:
                                raise Exception("Unknown label in regulations data!")
                            data['regulations'].append(e)
                        else:
                            raise Exception("Unknown label in regulations data!")
   
                    for e in hard_instances:
                        # if (len(entity_indices) < 2):
                        #     # ignore sentences with only one entity
                        #     pass
                        if (len(e['entities_indices']) == 2):
                            e['label'] = 0
                            data['hardInstances'].append(e)
                        # else:
                            # There are multiple entities, so create all possible pairs and classify them individually
                            # for i in range(0, len(e['entities_indices']) - 1):
                            #     new_entry = {
                            #         "sentence_tokens": e['sentence_tokens'], 
                            #         "entities_indices": [e['entities_indices'][i], e['entities_indices'][i+1]]}
                            #     new_entry['label'] = 0
                            #     data['hardInstances'].append(new_entry)
                            
                    for e in without_regulations:
                    
                        # if (len(entity_indices) < 2):
                        #     # ignore sentences with only one entity
                        #     pass 
                        if (len(e['entities_indices']) == 2):
                            # print(e['sentence_tokens'])
                            e['label'] = 2
                            data['withoutRegulations'].append(e)
                        # else:
                        #     # There are multiple entities, so create all possible pairs and classify them individually
                        #     for i in range(0, len(e['entities_indices'])-1):
                        #         new_entry = {
                        #             "sentence_tokens": e['sentence_tokens'], 
                        #             "entities_indices": [e['entities_indices'][i], e['entities_indices'][i+1]]}
                        #         new_entry['label'] = 2
                        #         data['withoutRegulations'].append(new_entry)
    # print()
    # for e in data:
    #     print(e, "\n")
    # print()
    if (configuration != 1):
        data = add_entity_markers(data)
    else:
        data = data
    # data = prune_data(data)
    return data


def prune_data(data):
    label_0 = 0 
    label_1 = 0
    label_2 = 0

    newRegulations = []
    for i in range(0, len(data['regulations'])):
        e = data['regulations'][i]
        tokens = tokenizer.tokenize(e['sentence_tokens'], is_split_into_words=True, truncation=True)     
        if (len(tokens) < 512):
            newRegulations.append(e)
        if (e['label'] == 0):
            label_0 += 1
        elif (e['label'] == 1):
            label_1 += 1
        else: 
            label_2 += 1
    data['regulations'] = newRegulations
    
    newHardInstances = []
    for i in range(0, len(data['hardInstances'])):
        e = data['hardInstances'][i]
        tokens = tokenizer.tokenize(e['sentence_tokens'], is_split_into_words=True, truncation=True) 
        del e['entities_indices']
        if (len(tokens) < 512):
            newHardInstances.append(e)
        if (e['label'] == 0):
            label_0 += 1
        elif (e['label'] == 1):
            label_1 += 1
        else: 
            label_2 += 1
    data['hardInstances'] = newHardInstances
    
    newWithoutRegulations = []
    for i in range(0, len(data['withoutRegulations'])):
        e = data['withoutRegulations'][i]
        tokens = tokenizer.tokenize(e['sentence_tokens'], is_split_into_words=True, truncation=True) 
        del e['entities_indices'] 
        if (len(tokens) < 512):
            newWithoutRegulations.append(e)   
        if (e['label'] == 0):
            label_0 += 1
        elif (e['label'] == 1):
            label_1 += 1
        else: 
            label_2 += 1
    data['withoutRegulations'] = newWithoutRegulations
    
    # print("label 0: ", label_0)
    # print("label 1: ", label_1)
    # print("label 2: ", label_2)
    return data


    
def maxpool(e1, e2, e3, e4):
    # combine embeddings
    # print("individual embedding shape: ", e1.shape)
    concatenated = torch.stack((e1, e2, e3, e4))
    maxpooled = torch.max_pool1d(concatenated, kernel_size=4)
    maxpooled = maxpooled.view(-1)
    return maxpooled
    
     


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
        # print("\n\n--- FORWARDING ---")
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        if (configuration == 3):
            embeddings = outputs.last_hidden_state
            final_vector = torch.empty(0)
            for i in range(0, len(input_ids)):
                tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
                entity_indexes = dict()
                for j in range(0, len(tokens)):
                    if (tokens[j] == "[E1]"):
                        entity_indexes["[E1]"] = j
                    if (tokens[j] == "[/E1]"):
                        entity_indexes["[/E1]"] = j
                    if (tokens[j] == "[E2]"):
                        entity_indexes["[E2]"] = j
                    if (tokens[j] == "[/E2]"):
                        entity_indexes["[/E2]"] = j

                if (entity_indexes.get("[E2]") == None or entity_indexes.get("[/E2]") == None):
                    raise Exception("Input is missing one or more entities.")
                
                # get each embedding for each entity marker
                e1 = embeddings[i][entity_indexes.get("[E1]")]
                e2 = embeddings[i][entity_indexes.get("[/E1]")]
                e3 = embeddings[i][entity_indexes.get("[E2]")]
                e4 = embeddings[i][entity_indexes.get("[/E2]")] 

                returned = maxpool(e1, e2, e3, e4) 
                # print("returned shape: ", returned.shape)
                final_vector = torch.cat((returned, final_vector))
                # print("final_vec shape rn: ", final_vector.shape)
                # print()

            final_vector = final_vector.view(len(input_ids), -1)  
            # print("final vector shape: ", final_vector.shape)
 
            sequence_output = self.dropout(final_vector)
            logits = self.classifier(sequence_output)

            loss = None
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss()
                inputs = logits.view(-1, self.num_labels)
                targets = labels.view(-1)
                loss = loss_fn(inputs, targets)            
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions
            )
        else:
            # print("CONFIG 1 or 2")
            cls_outputs = outputs.last_hidden_state[:, 0, :]
            # print("cls_outputs shape: ", cls_outputs.shape)
            cls_outputs = self.dropout(cls_outputs)
            # print("cls_outputs shape (after dropout): ", cls_outputs.shape)

            logits = self.classifier(cls_outputs)
            # print("logits shape: ", logits.shape)
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